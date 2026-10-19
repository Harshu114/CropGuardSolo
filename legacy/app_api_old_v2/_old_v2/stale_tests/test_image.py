#!/usr/bin/env python3
"""Test CropGuard on any image — no server needed.

Usage:
  python test_image.py path/to/photo.jpg
  python test_image.py path/to/photo.jpg --save

Prints a plain-English diagnosis and (with --save) writes an annotated copy
(detection boxes + lesion mask overlay) next to your photo.
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import CropGuardMultiTask                 # noqa: E402
from losses import dfl_decode, box_xywh_to_xyxy      # noqa: E402
from eval import nms                                 # noqa: E402

CLASS_NAMES = ["Healthy", "bacterial_spot", "early_blight", "late_blight",
               "leaf_miner", "leaf_mold", "magnesium_deficiency",
               "mosaic_virus", "nitrogen_deficiency", "potassium_deficiency",
               "septoria_leaf_spot", "spider_mites", "spotted_wilt_virus",
               "target_spot", "tomato", "yellow_leaf_curl_virus"]
SEV_LABELS = ["Healthy", "Mild", "Moderate", "Severe"]
SIZE, REG_MAX, NC = 320, 8, 16


def load_model(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    net = CropGuardMultiTask(nc=NC, pretrained=False)
    net.load_state_dict(sd)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default=os.path.join(HERE, "cropguard_model_final.pt"))
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--save", action="store_true", help="write annotated image")
    args = ap.parse_args()

    net = load_model(args.ckpt)

    bgr = cv2.imread(args.image)
    if bgr is None:
        sys.exit(f"cannot read image: {args.image}")
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1)[None]

    with torch.no_grad():
        out = net(x, with_severity=True)

    # ---- classification (the headline) ----
    probs = out["cls_logits"][0].softmax(-1)
    top = torch.topk(probs, 3)
    print("\n=== CropGuard diagnosis ===")
    print(f"image: {os.path.basename(args.image)}  ({W}x{H})")
    print(f"diagnosis: {CLASS_NAMES[int(top.indices[0])]} "
          f"({float(top.values[0]) * 100:.1f}% confident)")
    for i in range(1, 3):
        print(f"   alt {i}: {CLASS_NAMES[int(top.indices[i])]} "
              f"({float(top.values[i]) * 100:.1f}%)")

    # ---- severity ----
    sev_c = float(out["sev_cont"][0])
    sev_s = SEV_LABELS[int(out["sev_ord"][0].argmax())]
    print(f"severity: {sev_s} (score {sev_c:.2f})")

    # ---- detections ----
    anchors = out["anchors"]
    boxes_xywh = dfl_decode(out["det_box"], anchors, SIZE, REG_MAX)
    sc, lb = out["det_cls"][0].sigmoid().max(-1)
    m = sc > args.conf
    print(f"detections (conf > {args.conf}): {int(m.sum())}")
    if m.any():
        bb = box_xywh_to_xyxy(boxes_xywh[0][m]) * SIZE
        keep = nms(bb, sc[m])[:20]
        for k in keep.tolist():
            bx = bb[k]
            sx, sy = W / SIZE, H / SIZE
            print(f"   {CLASS_NAMES[int(lb[m][k])]:24s} score "
                  f"{float(sc[m][k]):.2f} box px "
                  f"[{int(bx[0] * sx)}, {int(bx[1] * sy)}, "
                  f"{int(bx[2] * sx)}, {int(bx[3] * sy)}]")

    # ---- segmentation ----
    lesion = float((out["seg_logits"][0].sigmoid() > 0.5).float().mean())
    print(f"lesion area: {lesion * 100:.1f}% of the frame")

    if args.save:
        vis = bgr.copy()  # uint8 BGR for OpenCV drawing
        mask = out["seg_logits"][0, 0].sigmoid().cpu().numpy()
        mask = cv2.resize(mask, (W, H)) > 0.5
        overlay_color = np.array([255, 0, 0], dtype=np.uint8)  # red lesion tint
        vis[mask] = (0.5 * vis[mask].astype(np.float32)
                     + 0.5 * overlay_color).astype(np.uint8)
        if m.any():
            for k in nms(bb, sc[m])[:20].tolist():
                sx, sy = W / SIZE, H / SIZE
                p1 = (int(bb[k][0] * sx), int(bb[k][1] * sy))
                p2 = (int(bb[k][2] * sx), int(bb[k][3] * sy))
                cv2.rectangle(vis, p1, p2, (0, 165, 255), 2)
                cv2.putText(vis, f"{CLASS_NAMES[int(lb[m][k])]} "
                            f"{float(sc[m][k]):.2f}", (p1[0], max(20, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        outp = os.path.splitext(args.image)[0] + "_cropguard.jpg"
        cv2.imwrite(outp, vis)
        print(f"annotated copy saved: {outp}")

    print("\nNOTE: research model, not a substitute for expert advice.")


if __name__ == "__main__":
    main()
