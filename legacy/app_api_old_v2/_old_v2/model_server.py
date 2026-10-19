"""CropGuard model server — loads model once, exposes inference function."""
import os
import uuid
import tempfile
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import UploadFile

# Import model architecture and postprocessing from training pipeline
from model import CropGuardMultiTask, _anchors
from losses import dfl_decode, box_xywh_to_xyxy
from inference_utils import nms

# Constants from training
CLASS_NAMES = [
    "Healthy", "bacterial_spot", "early_blight", "late_blight",
    "leaf_miner", "leaf_mold", "magnesium_deficiency",
    "mosaic_virus", "nitrogen_deficiency", "potassium_deficiency",
    "septoria_leaf_spot", "spider_mites", "spotted_wilt_virus",
    "target_spot", "tomato", "yellow_leaf_curl_virus"
]
SEV_LABELS = ["Healthy", "Mild", "Moderate", "Severe"]
IMG_SIZE = 320
REG_MAX = 8
CONF_THR = 0.25  # matches training eval; 0.1 leaked hundreds of weak boxes
MAX_DET = 50

_device: Optional[torch.device] = None
_model: Optional[CropGuardMultiTask] = None          # primary: v2 (det + cls)
_model_aux: Optional[CropGuardMultiTask] = None      # aux: v1 (seg + severity)
_model_version: str = "unknown"


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_model(model_path: str, role: str = "primary") -> CropGuardMultiTask:
    """Load CropGuardMultiTask from checkpoint. Handles bare state_dict / {'model':...} / {'state_dict':...}."""
    global _model, _model_version, _model_aux
    device = get_device()

    model = CropGuardMultiTask(nc=16, pretrained=False).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    ckpt_version = None
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            ckpt_version = checkpoint.get("model_version")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            ckpt_version = checkpoint.get("model_version")
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    if role == "aux":
        _model_aux = model
    else:
        _model = model
        _model_version = ckpt_version or "cropguard-final"
    return model


def get_model() -> CropGuardMultiTask:
    global _model
    if _model is None:
        model_path = os.environ.get("MODEL_PATH", "/app/cropguard_model_final.pt")
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found at {model_path}. Set MODEL_PATH env var.")
        load_model(model_path)
    return _model


def get_aux_model() -> Optional[CropGuardMultiTask]:
    """v1 checkpoint for seg+severity (dice 0.60 vs v2's 0.31). Optional:
    falls back to the primary model for everything if absent."""
    global _model_aux
    if _model_aux is None:
        aux_path = os.environ.get(
            "MODEL_PATH_AUX",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "cropguard_model_v1_backup.pt"))
        if aux_path and os.path.exists(aux_path):
            try:
                load_model(aux_path, role="aux")
                print(f"[aux] seg+sev ensemble with {aux_path}")
            except Exception as e:
                print(f"[aux] load failed ({e}); single-model mode")
    return _model_aux


def get_model_version() -> str:
    return _model_version


@torch.no_grad()
def preprocess_image(file_bytes: bytes) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load image, convert to RGB, resize to 320x320, normalize to [0,1]. Returns (tensor CHW, (orig_w, orig_h))."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image file")

    orig_h, orig_w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to 320x320 bilinear
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0,1] - NO mean/std
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1)  # CHW

    return tensor.unsqueeze(0), (orig_w, orig_h)


@torch.no_grad()
def postprocess(out: dict, orig_size: Tuple[int, int], conf_thr: float = CONF_THR) -> Tuple[list, list, list]:
    """Run postprocessing: decode DFL, filter, NMS, scale to original image size."""
    anchors = out["anchors"]
    boxes_xywh = dfl_decode(out["det_box"], anchors, IMG_SIZE, REG_MAX)
    scores_all, labels_all = out["det_cls"].sigmoid().max(-1)

    bb = box_xywh_to_xyxy(boxes_xywh[0]) * IMG_SIZE
    ss, ll = scores_all[0], labels_all[0]
    m = ss > conf_thr

    if m.sum() == 0:
        return [], [], []

    bb, ss, ll = bb[m], ss[m], ll[m]
    keep = nms(bb, ss)[:MAX_DET]
    bb, ss, ll = bb[keep], ss[keep], ll[keep]

    # Scale to original image size
    orig_w, orig_h = orig_size
    scale_x = orig_w / IMG_SIZE
    scale_y = orig_h / IMG_SIZE
    bb_scaled = bb.clone()
    bb_scaled[:, [0, 2]] *= scale_x
    bb_scaled[:, [1, 3]] *= scale_y

    return bb_scaled.cpu().numpy(), ss.cpu().numpy(), ll.cpu().numpy()


def predict_image(file_bytes: bytes, conf_thr: float = CONF_THR, with_cam: bool = True) -> dict:
    """Full inference pipeline. Returns dict matching API contract."""
    model = get_model()
    device = get_device()

    start_total = time.time()

    # Preprocess
    t0 = time.time()
    x, orig_size = preprocess_image(file_bytes)
    x = x.to(device).float()  # Force float32
    preprocess_ms = int((time.time() - t0) * 1000)

    # Model forward. Ensemble mode: primary (v2) does det+cls, aux (v1) does
    # seg+severity -- v1's seg head is measurably better (dice 0.60 vs 0.31).
    t0 = time.time()
    use_amp = device.type == "cuda"
    aux = get_aux_model()
    if use_amp:
        with torch.amp.autocast("cuda", enabled=True):
            out = model(x, with_severity=(aux is None))
            aux_out = aux(x, with_severity=True) if aux is not None else None
    else:
        out = model(x, with_severity=(aux is None))
        aux_out = aux(x, with_severity=True) if aux is not None else None
    seg_src = aux_out if aux is not None else out
    model_ms = int((time.time() - t0) * 1000)

    # Postprocess (no gradients needed)
    with torch.no_grad():
        boxes, scores, labels = postprocess(out, orig_size, conf_thr)

        # Classification
        cls_logits = out["cls_logits"]
        cls_probs = cls_logits.softmax(-1)[0]
        top3_idx = cls_probs.topk(3).indices.cpu().numpy()
        top3_conf = cls_probs.topk(3).values.cpu().numpy()
        top_class = int(top3_idx[0])
        top_conf = float(top3_conf[0])

        # Segmentation (aux/v1 when ensemble is active)
        seg_logits = seg_src["seg_logits"]
        seg_prob = seg_logits[0, 0].sigmoid().cpu().numpy()
        lesion_fraction = float(seg_prob.mean())

        # Severity (aux/v1 when ensemble is active)
        sev_cont = float(seg_src["sev_cont"][0])
        sev_stage_idx = int(seg_src["sev_ord"][0].argmax())
        sev_stage = SEV_LABELS[sev_stage_idx]

    # LayerCAM on the classification pathway (re-run neck with gradients).
    # The model STAYS in eval() mode: train() would flip BatchNorm to batch
    # statistics (batch=1 -> garbage features) and permanently pollute the
    # running stats on every request. CAM runs in fp32 for stable gradients.
    cam_img = None
    cam_class_idx = None
    if with_cam:
        x_cam = x.detach().clone().requires_grad_(True)
        c3, c4, c5 = model.backbone(x_cam)
        _, _, p5_grad = model.neck([c3, c4, c5])
        logits = model.cls_head(p5_grad)[0]
        class_idx = top_class
        score = logits[0, class_idx]
        grads = torch.autograd.grad(score, p5_grad)[0]
        alpha = F.relu(grads)
        cam = F.relu((alpha * p5_grad).sum(1, keepdim=True))
        cam = cam / cam.max().clamp(min=1e-6)
        cam_img = cam[0, 0].detach().cpu().numpy()
        cam_class_idx = class_idx

    postprocess_ms = int((time.time() - t0) * 1000)

    # ---- Lesion boxes FROM THE MASK (spot-level, not whole-leaf) ----------
    # The det head boxes whole leaves (that's how the training labels were
    # drawn), so derive tight per-spot boxes from connected mask regions.
    # Coordinates stay in 320-space; scaled to the original image below.
    boxes, scores, labels = postprocess(out, orig_size, conf_thr)
    mask_320 = (seg_prob > 0.5).astype(np.uint8)
    if mask_320.any():
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask_320, 8)
        spot_boxes = []
        for i in range(1, n):
            px = stats[i, cv2.CC_STAT_AREA]
            if px < 12:            # ignore specks
                continue
            bx, by, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            # pad 2px so spots aren't clipped
            bx, by = max(bx - 2, 0), max(by - 2, 0)
            bw = min(bw + 4, IMG_SIZE - bx)
            bh = min(bh + 4, IMG_SIZE - by)
            spot_boxes.append((bx, by, bw, bh, px))
        spot_boxes.sort(key=lambda b: -b[4])   # biggest lesions first
        orig_w, orig_h = orig_size
        sx, sy = orig_w / IMG_SIZE, orig_h / IMG_SIZE
        spots = [{
            "box": [round(b[0] * sx, 1), round(b[1] * sy, 1),
                    round((b[0] + b[2]) * sx, 1), round((b[1] + b[3]) * sy, 1)],
            "box_norm": [round(b[0] / IMG_SIZE, 4), round(b[1] / IMG_SIZE, 4),
                         round((b[0] + b[2]) / IMG_SIZE, 4),
                         round((b[1] + b[3]) / IMG_SIZE, 4)],
            "class": CLASS_NAMES[top_class],
            "score": round(min(0.95, 0.5 + 0.5 * float(seg_prob[b[1]:b[1]+b[3], b[0]:b[0]+b[2]].mean())), 2),
            "source": "lesion_mask",
        } for b in spot_boxes[:12]]
    else:
        spots = []

    # Display boxes = mask spots when the mask found lesions; det-head boxes
    # kept as "leaf_boxes" for reference only.
    det_list = []
    for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels):
        det_list.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "class": CLASS_NAMES[int(l)], "score": float(s),
            "box_norm": [float(x1) / orig_size[0], float(y1) / orig_size[1],
                         float(x2) / orig_size[0], float(y2) / orig_size[1]],
            "source": "det_head"})
    display_boxes = spots if spots else det_list

    # Save mask and CAM to temp files (return URLs)
    image_id = str(uuid.uuid4())
    tmp_dir = tempfile.gettempdir()

    # Save mask PNG (320x320, binary)
    mask_png = (seg_prob > 0.5).astype(np.uint8) * 255
    mask_path = os.path.join(tmp_dir, f"{image_id}_mask.png")
    cv2.imwrite(mask_path, mask_png)

    # Save CAM PNG (320x320, heatmap)
    cam_path = os.path.join(tmp_dir, f"{image_id}_cam.png")
    if cam_img is not None:
        cam_up = cv2.resize(cam_img, (IMG_SIZE, IMG_SIZE))
        heat = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(cam_path, heat)
    else:
        # Dummy transparent PNG
        cv2.imwrite(cam_path, np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    # Detections = spot-level boxes from the lesion mask (source-tagged);
    # whole-leaf det-head boxes preserved separately as leaf_boxes.
    detections = display_boxes
    leaf_boxes = [
        {"box": [float(x1), float(y1), float(x2), float(y2)],
         "class": CLASS_NAMES[int(l)], "score": float(s),
         "box_norm": [float(x1) / orig_size[0], float(y1) / orig_size[1],
                      float(x2) / orig_size[0], float(y2) / orig_size[1]]}
        for (x1, y1, x2, y2), s, l in zip(boxes, scores, labels)]

    # Classification top3
    top3 = [
        {"class": CLASS_NAMES[int(idx)], "confidence": float(conf)}
        for idx, conf in zip(top3_idx, top3_conf)
    ]

    return {
        "image_id": image_id,
        "width": orig_size[0],
        "height": orig_size[1],
        "detections": detections,
        "leaf_boxes": leaf_boxes,
        "classification": {
            "class": CLASS_NAMES[top_class],
            "confidence": float(top_conf),
            "top3": top3
        },
        "segmentation": {
            "mask_png_url": f"/v1/masks/{image_id}.png",
            "lesion_fraction": float(lesion_fraction)
        },
        "severity": {
            "continuous": float(sev_cont),
            "stage": sev_stage,
            "stage_idx": int(sev_stage_idx)
        },
        "cam_png_url": f"/v1/cams/{image_id}.png",
        "timings_ms": {
            "preprocess": preprocess_ms,
            "model": model_ms,
            "postprocess": postprocess_ms
        },
        "model_version": get_model_version() + ("+v1seg" if aux is not None else "")
    }


def get_mask_path(image_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{image_id}_mask.png")


def get_cam_path(image_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{image_id}_cam.png")