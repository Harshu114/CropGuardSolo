#!/usr/bin/env python3
"""Head-to-head field test: v2 (current API model) vs v3-hybrid candidate.
Loads both checkpoints directly, predicts the 6 ground-truth photos, prints
top-1 per photo + agreement with the filename ground truth."""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

API_DIR = os.path.dirname(os.path.abspath(__file__))
KAGGLE_DIR = os.path.abspath(os.path.join(API_DIR, "..", "..", "CropGuardOrganized", "KaggleTrainingPipeline"))
sys.path.insert(0, KAGGLE_DIR)
from model import CropGuardMultiTask  # noqa: E402

CLASS_NAMES = ["Healthy", "bacterial_spot", "early_blight", "late_blight",
               "leaf_miner", "leaf_mold", "magnesium_deficiency",
               "mosaic_virus", "nitrogen_deficiency", "potassium_deficiency",
               "septoria_leaf_spot", "spider_mites", "spotted_wilt_virus",
               "target_spot", "tomato", "yellow_leaf_curl_virus"]
IMG = 320

TRUTH = {
    "spotdisease.jpg": "bacterial_spot",
    "lateblight1.jpg": "late_blight",
    "tomato-late-blight.webp": "late_blight",
    "tomato-early-blight.webp": "early_blight",
    "tomato-septoria-leaf-spot.webp": "septoria_leaf_spot",
    "leaf1.jpg": "leaf_miner",
}

CKPTS = {
    "v2 (live)": os.path.join(API_DIR, "cropguard_model_final.pt"),
    "v3-hybrid": os.path.abspath(os.path.join(KAGGLE_DIR, "v3_hybrid", "student_best.pt")),
}


def load_model(path):
    m = CropGuardMultiTask(nc=16, pretrained=False)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck.get("model", ck))
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def predict(m, path):
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(cv2.resize(rgb, (IMG, IMG)).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    out = m(x, active_tasks={"cls", "det"})
    logits = out["cls_logits"][0]
    conf, idx = logits.softmax(0).max(0)
    seg = out.get("seg_logits")
    n_spots = 0
    if seg is not None:
        mask = (torch.sigmoid(seg[0, 0]) > 0.5).cpu().numpy().astype(np.uint8)
        if mask.any():
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            n_spots = sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12)
    return CLASS_NAMES[int(idx)], float(conf), n_spots


def main():
    device = "cpu"  # keep GPU free; 6 photos
    models = {}
    for name, path in CKPTS.items():
        models[name] = load_model(path).to(device)
        print(f"loaded {name}: {os.path.basename(path)}")

    dl = os.path.expanduser("~/Downloads")
    photos = [p for p in TRUTH if os.path.exists(os.path.join(dl, p))]
    print(f"\n{len(photos)}/6 photos found\n")
    wins = {k: 0 for k in CKPTS}
    for p in photos:
        truth = TRUTH[p]
        line = f"{p:34} truth={truth:18}"
        for name, m in models.items():
            r = predict(m, os.path.join(dl, p))
            ok = r and r[0] == truth
            if ok:
                wins[name] += 1
            line += f" | {name}: {r[0]} {r[1]:.2f} {'OK' if ok else 'X'}"
        print(line)
    print()
    for name, w in wins.items():
        print(f"{name}: {w}/{len(photos)} field correct")


if __name__ == "__main__":
    main()
