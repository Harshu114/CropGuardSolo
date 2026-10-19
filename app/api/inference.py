"""Inference + LayerCAM explainability.

Usage: python inference.py --image path.jpg --ckpt student_best.pt [--out out_dir]
Produces: annotated boxes + class + severity estimate + pseudo-mask overlay +
LayerCAM heatmap overlay PNG.
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import CropGuardMultiTask
from losses import dfl_decode, box_xywh_to_xyxy

CLASS_NAMES = ["healthy", "bacterial_spot", "early_blight", "late_blight",
               "leaf_miner", "leaf_mold", "mosaic_virus",
               "septoria_leaf_spot", "spider_mites",
               "target_spot", "yellow_leaf_curl_virus"]
SEV_LABELS = ["Healthy", "Mild", "Moderate", "Severe"]


@torch.no_grad()
def postprocess(out, img_size, conf_thr=0.25, max_det=50, reg_max=8):
    from eval import nms
    anchors = out["anchors"]
    boxes_xywh = dfl_decode(out["det_box"], anchors, img_size, reg_max)
    scores_all, labels_all = out["det_cls"].sigmoid().max(-1)
    bb = box_xywh_to_xyxy(boxes_xywh[0]) * img_size
    ss, ll = scores_all[0], labels_all[0]
    m = ss > conf_thr
    if m.sum() == 0:
        return torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    bb, ss, ll = bb[m], ss[m], ll[m]
    keep = nms(bb, ss)[:max_det]
    return bb[keep], ss[keep], ll[keep]


def layercam(model, x, class_idx=None):
    """LayerCAM on P5 features using the classification head logits (eqs 12-13)."""
    feats = {}
    hooks = []

    def make_hook(key):
        def hook(mod, inp, o):
            feats[key] = o
        return hook

    # Capture all three neck levels: the classification head now fuses
    # P3+P4+P5 (multi-scale, BUGFIX #6) and no longer accepts a single map.
    for i, key in enumerate(("p3", "p4", "p5")):
        hooks.append(model.neck.out_proj[i].register_forward_hook(make_hook(key)))
    model.eval()
    _ = model(x, with_severity=False)
    for h in hooks:
        h.remove()

    p5 = feats["p5"]
    logits = model.cls_head([feats["p3"], feats["p4"], feats["p5"]])[0]
    if class_idx is None:
        class_idx = int(logits.argmax(-1))
    score = logits[0, class_idx]
    grads = torch.autograd.grad(score, p5, retain_graph=False)[0]      # 1,C,h,w
    alpha = F.relu(grads)
    cam = (alpha * p5).sum(1, keepdim=True)
    cam = F.relu(cam)
    cam = cam / cam.max().clamp(min=1e-6)
    return cam[0, 0].detach().cpu().numpy(), class_idx


def overlay(image_bgr, mask, color=(0, 0, 255), alpha=0.4):
    ov = image_bgr.copy()
    m = mask > 0.5
    ov[m] = (1 - alpha) * ov[m] + alpha * np.array(color, dtype=np.float32)
    return ov


def draw_boxes(image_bgr, boxes, scores, labels):
    for bb, sc, lb in zip(boxes.cpu().numpy(), scores.cpu().numpy(),
                          labels.cpu().numpy()):
        x1, y1, x2, y2 = map(int, bb)
        name = CLASS_NAMES[int(lb)] if 0 <= int(lb) < len(CLASS_NAMES) else str(lb)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"{name} {sc:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return image_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ckpt", default="/kaggle/working/runs/student_best.pt")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="/kaggle/working/predictions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropGuardMultiTask(nc=11, pretrained=False).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    import cv2 as _cv2
    img0 = _cv2.imread(args.image)
    rgb = _cv2.cvtColor(img0, _cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = F.interpolate(x[None], size=(args.size, args.size),
                      mode="bilinear", align_corners=False).to(device)

    with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        out = model(x, with_severity=True)
    boxes, scores, labels = postprocess(out, args.size, args.conf)
    mask = (out["seg_logits"][0, 0].sigmoid() > 0.5).float().cpu().numpy()
    sev = float(out["sev_cont"][0])
    stage = int(out["sev_ord"][0].argmax())
    cam, cam_cls = layercam(model, x)

    # render on the 320x320 canvas so mask + boxes (320-space) align with pixels
    vis_src = cv2.resize(rgb, (args.size, args.size))
    vis = vis_src.astype(np.float32) / 255.0
    vis = overlay(vis, mask, color=(0, 0, 255), alpha=0.35)
    vis = (vis.clip(0, 1) * 255).astype(np.uint8)   # cv2 drawing needs uint8
    vis = draw_boxes(vis, boxes, scores, labels)
    cam_up = cv2.resize(cam, (args.size, args.size))
    heat = cv2.applyColorMap((cam_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blend = (0.5 * vis.astype(np.float32) / 255.0 + 0.5 * heat).clip(0, 1)

    top = CLASS_NAMES[cam_cls] if 0 <= cam_cls < 11 else str(cam_cls)
    panel = vis.copy()
    cv2.putText(panel, f"cls: {top} | sev: {sev:.2f} ({SEV_LABELS[stage]})",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    cv2.imwrite(os.path.join(args.out, f"{stem}_pred.jpg"),
                cv2.cvtColor((panel * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out, f"{stem}_layercam.jpg"),
                cv2.cvtColor((blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"severity={sev:.3f} stage={SEV_LABELS[stage]} class={top} "
          f"dets={len(boxes)} -> {args.out}")


if __name__ == "__main__":
    main()
