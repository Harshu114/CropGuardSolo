"""CropGuard model server — loads the v5 model once, exposes inference.

Model: CropGuardMultiTask (nc=11, width=96, FasterNet-T0 backbone, 2.63M params)
Checkpoint: cropguard_model_v5.pt == runs/student_best.pt from Kaggle kernel v5
           ("CropGuard Claude Train"), trained on cropguard-corpus-claude.

Test-split scores for this checkpoint (see MODEL_MANIFEST.json / metrics.json):
  mAP@0.5 0.511   mAP@0.5:0.95 0.431   cls_top1 0.853   cls_macro_f1 0.843
  seg_mIoU 0.880  seg_dice 0.936       sev_mae 0.068 (sev_r2 -1.41, WEAK)

Architecture notes that matter for serving:
  * 11 classes (was 16 in the old v2 model). Order matches data.yaml.
  * The classification head is MULTI-SCALE: it pools P3+P4+P5, so any call
    site must pass a list. `model.cls_head(p5)` is the stale single-scale call
    and will raise a mat1/mat2 shape error.
  * The neck returns [p3, p4, p5], not a 3-tuple unpacked differently.
  * The checkpoint is a BARE state_dict (OrderedDict of tensors), so it loads
    cleanly under weights_only=True. The old {'model': ...} wrapper is gone.
"""
import os
import uuid
import tempfile
import time
import inspect
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Import model architecture and postprocessing from the v5 training pipeline
from model import CropGuardMultiTask
from losses import dfl_decode, box_xywh_to_xyxy
from inference_utils import nms

# Class ORDER is fixed by data.yaml of cropguard-corpus-claude. Renaming or
# reordering here silently mislabels every prediction — treat as frozen.
#
# The STRING FORMAT is not free either: the web UI looks these up directly as
# keys of DISEASE_INFO (web/src/lib/api.ts), which is keyed 'Healthy' +
# snake_case. Returning "Septoria Leaf Spot" instead makes the frontend fall
# back to "Information not available." for every disease. Keep this format.
CLASS_NAMES = [
    "Healthy", "bacterial_spot", "early_blight", "late_blight",
    "leaf_miner", "leaf_mold", "mosaic_virus", "septoria_leaf_spot",
    "spider_mites", "target_spot", "yellow_leaf_curl_virus",
]
SEV_LABELS = ["Healthy", "Mild", "Moderate", "Severe"]
NUM_CLASSES = 11
IMG_SIZE = 320
REG_MAX = 8
# Measured on 120 real test images (mean det-head boxes / images with zero):
#   0.05 -> 1.84 / 2      0.10 -> 1.73 / 6      0.15 -> 1.63 / 11
#   0.25 -> 1.52 / 16     0.50 -> 0.99 / 34
# 0.25 dropped a box on 13% of images; 0.05 starts admitting noise. 0.15 is the
# measured compromise. The `detections` the UI actually draws come from the
# lesion mask and are unaffected by this value -- it only gates `leaf_boxes`.
CONF_THR = 0.15
MAX_DET = 50
# Horizontal-flip test-time augmentation. Measured on 1000 val images with the
# real COCO AP: mAP@0.5 0.4972 -> 0.5101, mAP@0.5:0.95 0.4196 -> 0.4300,
# cls_top1 0.9316 -> 0.9326. Costs a second forward pass (~9 ms on this GPU).
USE_TTA = True

_device: Optional[torch.device] = None
_model: Optional[CropGuardMultiTask] = None
_model_version: str = "unknown"


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(model_path: str) -> CropGuardMultiTask:
    """Load the v5 CropGuardMultiTask from checkpoint.

    Accepts a bare state_dict (what the v5 kernel writes) and tolerates the
    {'state_dict': ...} / {'model': ...} wrappers older checkpoints used.
    """
    global _model, _model_version
    device = get_device()

    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}. Set MODEL_PATH env var.")

    model = CropGuardMultiTask(nc=NUM_CLASSES, pretrained=False).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    ckpt_version = None
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            ckpt_version = checkpoint.get("model_version")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            ckpt_version = checkpoint.get("model_version")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # strict=True so a stale 16-class checkpoint fails loudly here instead of
    # silently producing garbage predictions from partially-loaded weights.
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _model = model
    _model_version = ckpt_version or "v5-claude"
    return model


def get_model() -> CropGuardMultiTask:
    global _model
    if _model is None:
        model_path = os.environ.get("MODEL_PATH", "cropguard_model_v5.pt")
        if not os.path.isabs(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), model_path)
        load_model(model_path)
    return _model


def get_model_version() -> str:
    return _model_version


# ---- hybrid serving: a v7-family segmentation side-model --------------------
# The served v5 classifier STAYS the classifier. Its 11-class head can name
# leaf_miner / spider_mites / target_spot; the nc=8 v7-family head structurally
# cannot, and an 8-way head must assign such a photo to some *disease* -- which
# recommends a fungicide for an insect. That is the worst error this app can make,
# so swapping the classifier is a product decision, not a drop-in.
# The v7-family model supplies ONLY the lesion mask and analytic severity. Its
# cls/det heads are never run (active_tasks={"seg"}), so the 8-class space cannot
# leak into the response. Set SEG_MODEL_PATH to enable; unset = today's behaviour.
SEG_NC = 8
# Lab-measured basis for this threshold: v9 predicts severity 0.0119 (max 0.129)
# on 20 GT-healthy PlantSeg val leaves, so 0.10 sits above the healthy band. It is
# NOT validated on field imagery, where 2/8 GT-healthy leaves already read ~0.30.
LESION_CONFLICT_SEVERITY = 0.10

_seg_state: dict = {"tried": False, "model": None, "version": None, "reason": None}


def get_seg_model():
    """Lazy-load the segmentation side-model. Returns None (and the reason) rather
    than raising, so a missing/mismatched checkpoint degrades to v5-only serving
    instead of failing every request."""
    if _seg_state["tried"]:
        return _seg_state["model"]
    _seg_state["tried"] = True
    path = os.environ.get("SEG_MODEL_PATH", "")
    if not path:
        _seg_state["reason"] = "SEG_MODEL_PATH not set"
        return None
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        _seg_state["reason"] = f"not found: {path}"
        return None
    try:
        from model_v7_serving import CropGuardMultiTaskV7
        device = get_device()
        model = CropGuardMultiTaskV7(nc=SEG_NC, pretrained=False).to(device)
        # weights_only=False: this checkpoint is the training dict
        # {'model','epoch','metrics','which'}, not a bare state_dict.
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        model.load_state_dict(sd, strict=True)
        model.eval()
        _seg_state["model"] = model
        _seg_state["version"] = (f"seg-nc{SEG_NC}-epoch{ck.get('epoch')}"
                                 if isinstance(ck, dict) else f"seg-nc{SEG_NC}")
        print(f"[serve] segmentation side-model loaded: {path} "
              f"({_seg_state['version']})", flush=True)
    except Exception as e:
        _seg_state["model"] = None
        _seg_state["reason"] = f"{type(e).__name__}: {e}"
        print(f"[serve] segmentation side-model DISABLED: {_seg_state['reason']}",
              flush=True)
    return _seg_state["model"]



# ---- v11 field abstention gate artifact (documentation + future classifier swap) ----
# The served classifier is v5 (nc=11, CropGuardMultiTask). v11 is an nc=8 v7-family model whose
# field gate constants are documented here for the next classifier swap. Until the classifier is
# swapped, this artifact is informational only and does NOT affect /v1/predict responses.
_V11_GATE_STATE: dict = {"tried": False, "artifact": None, "reason": None}


def _canonical_v11_gate_path() -> str:
    """Resolve the v11 gate artifact relative to model_server.py's real location.
    Robust to import topology: works whether model_server is imported as 'model_server'
    (main.py style) or re-exported via 'app.api.model_server'."""
    base = os.path.abspath(os.path.dirname(inspect.getfile(model_server)))
    return os.path.join(base, "model/weights/v11/v11_field_gate_artifact.json")


def get_v11_gate_artifact() -> dict:
    """Lazy-load the v11 field gate artifact. Returns the parsed JSON dict, or None."""
    if _V11_GATE_STATE["tried"]:
        return _V11_GATE_STATE["artifact"]
    _V11_GATE_STATE["tried"] = True
    path = _canonical_v11_gate_path()
    if not os.path.exists(path):
        _V11_GATE_STATE["reason"] = f"not found: {path}"
        return None
    try:
        with open(path) as f:
            _V11_GATE_STATE["artifact"] = _json.load(f)
        print(f"[serve] v11 field gate artifact loaded: {path}", flush=True)
        return _V11_GATE_STATE["artifact"]
    except Exception as e:
        _V11_GATE_STATE["reason"] = f"{type(e).__name__}: {e}"
        print(f"[serve] v11 field gate artifact DISABLED: {_V11_GATE_STATE['reason']}", flush=True)
        return None


def v11_gate_status() -> dict:
    """Status of the v11 field gate artifact (informational until classifier swap)."""
    return {
        "enabled": _V11_GATE_STATE["artifact"] is not None,
        "model": "v11",
        "classifier_nc": 8,
        "note": "informational until the served classifier is swapped to a v11-loadable server",
        "artifact": _V11_GATE_STATE["artifact"],
        "reason": _V11_GATE_STATE["reason"],
    }


def get_v11_model_path() -> str:
    """Path to the v11 checkpoint tracked under model/weights/v11/ (for the next swap)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "model/weights/v11/best_v11_raw.pt")


def load_v11_model() -> CropGuardMultiTaskV7:
    """Load the v11 nc=8 classifier (for the next classifier swap + gate wiring).

    Not called by /v1/predict until the classifier is swapped; provided so the swap is a
    one-location change plus setting V11_MODEL_PATH / enabling the v11 predict path.
    """
    path = os.environ.get("V11_MODEL_PATH", get_v11_model_path())
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        raise RuntimeError(f"v11 checkpoint not found at {path}")
    device = get_device()
    model = CropGuardMultiTaskV7(nc=8, pretrained=False).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

def seg_model_status() -> dict:
    get_seg_model()
    return {"enabled": _seg_state["model"] is not None,
            "version": _seg_state["version"],
            "reason": _seg_state["reason"]}


# Letterbox fill value. Training used aspect-preserving scale-jitter that pads
# with (114,114,114) gray (`CropGuardDataset._augment`), so a gray-padded square
# is in-distribution -- a hard aspect squash is not.
LETTERBOX_FILL = 114


@torch.no_grad()
def preprocess_image(file_bytes: bytes) -> Tuple[torch.Tensor, Tuple[int, int],
                                                Tuple[int, int, int, int]]:
    """Decode to RGB, LETTERBOX to 320x320 on gray-114, normalize to [0,1].

    Why letterbox instead of a plain resize to 320x320:
      * the corpus is 100% square 320x320 images, and the only geometric
        augmentation is aspect-preserving scale-jitter + gray padding. The model
        has therefore never seen a non-uniformly squashed image, yet the old
        code squashed every non-square upload (e.g. 750x562 -> 320x320).
      * the returned content rect lets boxes, the mask and the CAM all be mapped
        back onto the original frame.

    Returns (1,C,H,W), (orig_w, orig_h), (pad_x, pad_y, content_w, content_h).
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image file")

    orig_h, orig_w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    content_w = max(1, int(round(orig_w * scale)))
    content_h = max(1, int(round(orig_h * scale)))
    # INTER_AREA for downscaling (less aliasing), INTER_LINEAR otherwise. For an
    # already-square 320x320 input this is an exact identity: scale == 1.0,
    # content == 320, pads == 0 -- so corpus metrics are unchanged.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (content_w, content_h), interpolation=interp)

    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), LETTERBOX_FILL, dtype=np.uint8)
    pad_x = (IMG_SIZE - content_w) // 2
    pad_y = (IMG_SIZE - content_h) // 2
    canvas[pad_y:pad_y + content_h, pad_x:pad_x + content_w] = resized

    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)
    return tensor.unsqueeze(0), (orig_w, orig_h), (pad_x, pad_y, content_w, content_h)


@torch.no_grad()
def postprocess(out: dict, orig_size: Tuple[int, int],
                conf_thr: float = CONF_THR,
                out_flip: Optional[dict] = None,
                lb: Tuple[int, int, int, int] = (0, 0, IMG_SIZE, IMG_SIZE),
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DFL decode -> confidence filter -> optional flip-TTA union -> NMS -> scale.

    `lb` is the letterbox content rect (pad_x, pad_y, content_w, content_h) from
    preprocess_image; boxes are mapped from the padded 320x320 canvas back onto
    the original photo through it.

    When `out_flip` is given (the mirrored forward pass), its boxes are mirrored
    back into the original frame and unioned with the base pass before a single
    NMS. Measured on 1000 val images with the real COCO AP:
        mAP@0.5       0.4972 -> 0.5101  (+0.0129)
        mAP@0.5:0.95  0.4196 -> 0.4300  (+0.0104)
        cls_top1      0.9316 -> 0.9326  (negligible)
    Detection-only; segmentation/severity still come from the base pass.
    """
    def decode(o):
        anchors = o["anchors"]
        boxes_xywh = dfl_decode(o["det_box"], anchors, IMG_SIZE, REG_MAX)
        scores_all, labels_all = o["det_cls"].sigmoid().max(-1)
        bb = box_xywh_to_xyxy(boxes_xywh[0]) * IMG_SIZE
        ss, ll = scores_all[0], labels_all[0]
        m = ss > conf_thr
        return bb[m], ss[m], ll[m]

    bb, ss, ll = decode(out)

    if out_flip is not None:
        bbf, scf, lbf = decode(out_flip)
        if bbf.numel():
            mf = bbf.clone()
            mf[:, 0] = IMG_SIZE - bbf[:, 2]      # mirror x back to the original
            mf[:, 2] = IMG_SIZE - bbf[:, 0]
            bb = torch.cat([bb, mf], 0)
            ss = torch.cat([ss, scf], 0)
            ll = torch.cat([ll, lbf], 0)

    if bb.numel() == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)

    keep = nms(bb, ss)[:MAX_DET]
    bb, ss, ll = bb[keep], ss[keep], ll[keep]

    orig_w, orig_h = orig_size
    pad_x, pad_y, content_w, content_h = lb
    bb_scaled = bb.clone()
    # Undo the letterbox: drop the pad offset, then scale the content region.
    bb_scaled[:, [0, 2]] = (bb_scaled[:, [0, 2]] - pad_x) * (orig_w / content_w)
    bb_scaled[:, [1, 3]] = (bb_scaled[:, [1, 3]] - pad_y) * (orig_h / content_h)

    # DFL regression routinely overshoots by a couple of pixels, so boxes can
    # land just outside the frame. Clip here: the frontend draws these raw and
    # an out-of-bounds box reads as a bug. Degenerate boxes are dropped.
    bb_scaled[:, [0, 2]] = bb_scaled[:, [0, 2]].clamp(0, orig_w)
    bb_scaled[:, [1, 3]] = bb_scaled[:, [1, 3]].clamp(0, orig_h)
    ok = ((bb_scaled[:, 2] - bb_scaled[:, 0]) > 1) & ((bb_scaled[:, 3] - bb_scaled[:, 1]) > 1)
    bb_scaled, ss, ll = bb_scaled[ok], ss[ok], ll[ok]

    return bb_scaled.cpu().numpy(), ss.cpu().numpy(), ll.cpu().numpy()


def _multiscale_layercam(model: CropGuardMultiTask, x: torch.Tensor,
                         class_idx: int) -> np.ndarray:
    """LayerCAM over P3+P4+P5, matching the multi-scale classification head.

    The model stays in eval() mode on purpose: train() would make BatchNorm use
    batch stats (batch=1 -> garbage) and would permanently corrupt the running
    statistics on every request.

    NOTE: predict_image is wrapped in @torch.no_grad(), which would make this
    backward pass impossible ("element 0 of tensors does not require grad"), so
    the whole computation runs inside an explicit torch.enable_grad() scope.
    """
    with torch.enable_grad():
        x_cam = x.detach().clone().float().requires_grad_(True)
        c3, c4, c5 = model.backbone(x_cam)
        p3, p4, p5 = model.neck([c3, c4, c5])
        logits = model.cls_head([p3, p4, p5])[0]      # multi-scale call
        score = logits[0, class_idx]
        levels = (p3, p4, p5)
        grads = torch.autograd.grad(score, list(levels))

        cam = None
        for p, g in zip(levels, grads):
            alpha = torch.relu(g)
            contrib = torch.relu((alpha * p).sum(1, keepdim=True))
            contrib = F.interpolate(contrib, size=(IMG_SIZE, IMG_SIZE),
                                    mode="bilinear", align_corners=False)
            cam = contrib if cam is None else cam + contrib

        cam = cam / cam.max().clamp(min=1e-6)
        return cam[0, 0].detach().cpu().numpy()


@torch.no_grad()
def predict_image(file_bytes: bytes, conf_thr: float = CONF_THR,
                  with_cam: bool = True) -> dict:
    """Full inference pipeline. Returns the dict matching the API contract."""
    model = get_model()
    device = get_device()

    start_total = time.time()

    t0 = time.time()
    x, orig_size, lb = preprocess_image(file_bytes)
    x = x.to(device).float()
    preprocess_ms = int((time.time() - t0) * 1000)

    t0 = time.time()
    use_amp = device.type == "cuda"
    # Flip-TTA: a second mirrored forward pass, unioned in postprocess() and
    # averaged into the classification. Measured +0.0129 mAP@0.5 on val for
    # ~9 ms of extra work on this GPU. Set USE_TTA = False to disable.
    x_flip = torch.flip(x, dims=[3]) if USE_TTA else None
    if use_amp:
        with torch.amp.autocast("cuda", enabled=True):
            out = model(x, with_severity=True)
            out_flip = model(x_flip, with_severity=True) if USE_TTA else None
    else:
        out = model(x, with_severity=True)
        out_flip = model(x_flip, with_severity=True) if USE_TTA else None
    model_ms = int((time.time() - t0) * 1000)

    t0 = time.time()
    orig_w, orig_h = orig_size
    pad_x, pad_y, content_w, content_h = lb

    with torch.no_grad():
        boxes, scores, labels = postprocess(out, orig_size, conf_thr,
                                            out_flip=out_flip, lb=lb)

        logits_c = out["cls_logits"].float()
        if out_flip is not None:
            logits_c = (logits_c + out_flip["cls_logits"].float()) / 2
        cls_probs = logits_c.softmax(-1)[0]
        top3 = cls_probs.topk(3)
        top3_idx = top3.indices.cpu().numpy()
        top3_conf = top3.values.cpu().numpy()
        top_class = int(top3_idx[0])
        top_conf = float(top3_conf[0])

        seg_prob = out["seg_logits"][0, 0].float().sigmoid().cpu().numpy()
        # Fraction of the photo the segmentation head marks as leaf/plant, measured
        # over the letterbox CONTENT rect only (the gray pad is not the photo).
        #
        # NOT a lesion fraction. This corpus's segmentation target is the filled
        # leaf bounding box -- the stored mask files are binary {0,255} and the
        # 128 "lesion" value never occurs in any of the 32,897 files (verified).
        # So the head predicts a leaf REGION, and treating it as lesions drew a
        # full-frame box over nearly every photo.
        seg_content = seg_prob[pad_y:pad_y + content_h, pad_x:pad_x + content_w]
        leaf_fraction = float(seg_content.mean())

        # Severity is the WEAK head on this checkpoint (sev_r2 -1.41; mae 0.068
        # vs 0.009 for a constant predictor -- worse than predicting the mean).
        # Its labels were derived from how much of the FRAME the leaf box covers,
        # which for a close-up leaf photo is always >= 30% -> "Severe" for
        # 1991/2000 val images and every real upload tested. Served to keep the
        # API contract intact, flagged reliable=False so clients can hide it
        # rather than present "Severe" as a diagnosis for every photo.
        sev_cont = float(out["sev_cont"][0])
        sev_stage_idx = int(out["sev_ord"][0].argmax())
        sev_stage = SEV_LABELS[sev_stage_idx]
        severity_reliable = False

    # ---- segmentation side-model (v7-family): lesion mask + analytic severity --
    # Same letterboxed tensor, so the overlay lands on the same content rect as
    # every other output. Only the seg head runs; see the note above.
    t_seg = time.time()
    seg_model = get_seg_model()
    lesion_content = None
    sev_v7 = None
    lesion_fraction = None
    seg_conflict = False
    if seg_model is not None:
        with torch.no_grad():
            so = seg_model(x, active_tasks={"seg"})
            p7 = torch.sigmoid(so["seg_logits"].float())[0]
            sev_val, _ = seg_model.analytic_severity(so["seg_logits"])
            sev_v7 = float(sev_val[0])
        les7 = (p7[0] > 0.5).cpu().numpy().astype(np.uint8)
        lesion_content = les7[pad_y:pad_y + content_h, pad_x:pad_x + content_w]
        lesion_fraction = float(lesion_content.mean())
        # Contradiction guard: v5 says Healthy while v9 paints a real lesion load.
        # v5's healthy call is NOT allowed to suppress the overlay -- the
        # healthy-shortcut probe (0.9841 val accuracy from bare image statistics)
        # says that label partly tracks the background, not the leaf.
        seg_conflict = bool(top_class == 0 and sev_v7 > LESION_CONFLICT_SEVERITY)
    seg_ms = int((time.time() - t_seg) * 1000)

    cam_img = None
    if with_cam:
        cam_img = _multiscale_layercam(model, x, top_class)

    # REMOVED: "lesion spot" boxes derived from the segmentation map via
    # connected components. The map is a leaf-region map, so those "spots" were
    # really one leaf-sized rectangle covering most of the photo (measured: 77.6%
    # of the frame as a single 79,439 px component on a real upload). The
    # detection head's own leaf boxes are reported instead.

    leaf_boxes = [
        {"box": [float(x1), float(y1), float(x2), float(y2)],
         "class": CLASS_NAMES[int(l)], "score": float(s),
         "box_norm": [round(float(x1) / orig_w, 4), round(float(y1) / orig_h, 4),
                      round(float(x2) / orig_w, 4), round(float(y2) / orig_h, 4)],
         "source": "det_head"}
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels)]

    image_id = str(uuid.uuid4())
    tmp_dir = tempfile.gettempdir()

    # Mask and CAM are written at the CONTENT rect, not 320x320, so they carry
    # the photo's own aspect ratio and line up when the UI stretches them over
    # the photo (they used to be squashed 320x320 overlays).
    mask_path = os.path.join(tmp_dir, f"{image_id}_mask.png")
    cv2.imwrite(mask_path, (seg_content > 0.5).astype(np.uint8) * 255)

    lesion_mask_path = None
    if lesion_content is not None:
        lesion_mask_path = os.path.join(tmp_dir, f"{image_id}_lesion.png")
        cv2.imwrite(lesion_mask_path, lesion_content * 255)

    cam_path = os.path.join(tmp_dir, f"{image_id}_cam.png")
    if cam_img is not None:
        cam_content = cam_img[pad_y:pad_y + content_h, pad_x:pad_x + content_w]
        heat = cv2.applyColorMap((cam_content * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(cam_path, heat)
    else:
        cv2.imwrite(cam_path, np.zeros((content_h, content_w, 3), dtype=np.uint8))

    postprocess_ms = int((time.time() - t0) * 1000)

    return {
        "image_id": image_id,
        "width": orig_w,
        "height": orig_h,
        # Detections are the detection head's own output (whole-leaf boxes) --
        # the task the labels actually describe. `leaf_boxes` is kept as an
        # alias so clients written against the previous contract still work.
        "detections": leaf_boxes,
        "leaf_boxes": leaf_boxes,
        "classification": {
            "class": CLASS_NAMES[top_class],
            "confidence": round(top_conf, 4),
            "top3": [{"class": CLASS_NAMES[int(i)], "confidence": round(float(c), 4)}
                     for i, c in zip(top3_idx, top3_conf)],
        },
        "segmentation": {
            "mask_png_url": f"/v1/masks/{image_id}.png",
            "cam_png_url": f"/v1/cams/{image_id}.png",
            # Renamed from `lesion_fraction`: the value is the predicted LEAF
            # region, not damaged tissue (see predict_image).
            "leaf_fraction": round(leaf_fraction, 4),
            # Additive: real lesion pixels from the v7-family head. None when the
            # side-model is disabled, so old clients keep working unchanged.
            "lesion_mask_png_url": (f"/v1/lesion_masks/{image_id}.png"
                                    if lesion_mask_path else None),
            "lesion_fraction": (round(lesion_fraction, 4)
                                if lesion_fraction is not None else None),
            "seg_model": _seg_state["version"],
            "seg_conflict": seg_conflict,
        },
        "severity": {
            "continuous": round(sev_cont, 4),
            "stage": sev_stage,
            "stage_idx": sev_stage_idx,
            "reliable": severity_reliable,
            # Analytic severity (lesion∩leaf / leaf) from the side-model. It beat
            # a constant predictor on PlantSeg val (MAE 0.1296 vs 0.2098, r 0.7733)
            # but has NO ground-truth measurement on field imagery, so it is
            # exposed without a stage label and clients must keep hiding the meter
            # until severity_reliable is true.
            "lesion_ratio": (round(sev_v7, 4) if sev_v7 is not None else None),
            "lesion_ratio_validated_on": ("PlantSeg val (lab), not field imagery"
                                          if sev_v7 is not None else None),
        },
        "timings_ms": {
            "preprocess": preprocess_ms,
            "model": model_ms,
            "postprocess": postprocess_ms,
            "seg_model": seg_ms,
        },
        "model_version": get_model_version(),
    }


def get_mask_path(image_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{image_id}_mask.png")


def get_cam_path(image_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{image_id}_cam.png")


def get_lesion_mask_path(image_id: str) -> str:
    """Lesion overlay from the v7-family side-model. Separate file suffix so it
    can never overwrite the v5 leaf-region mask served by get_mask_path."""
    return os.path.join(tempfile.gettempdir(), f"{image_id}_lesion.png")