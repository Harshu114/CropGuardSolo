#!/usr/bin/env python3
"""Test 2-model cls ensemble (v2 + v3-hybrid) on the 6 field photos.
Reports per-photo: each model's top1, prob-avg top1, logit-avg top1."""
import os
import sys

import cv2
import numpy as np
import torch

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
    "v2": os.path.join(API_DIR, "cropguard_model_final.pt"),
    "v3": os.path.abspath(os.path.join(KAGGLE_DIR, "v3_hybrid", "student_best.pt")),
}


def load_model(path):
    m = CropGuardMultiTask(nc=16, pretrained=False)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck.get("model", ck))
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def probs(m, path):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(cv2.resize(rgb, (IMG, IMG)).astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    out = m(x, active_tasks={"cls"})
    return out["cls_logits"][0].softmax(0), out["cls_logits"][0]


def main():
    models = {k: load_model(v) for k, v in CKPTS.items()}
    dl = os.path.expanduser("~/Downloads")
    wins = {"v2": 0, "v3": 0, "prob-ens": 0, "logit-ens": 0}
    for p, truth in TRUTH.items():
        fp = os.path.join(dl, p)
        if not os.path.exists(fp):
            continue
        P, L = {}, {}
        for k, m in models.items():
            P[k], L[k] = probs(m, fp)
        t2 = {k: CLASS_NAMES[int(v.argmax())] for k, v in P.items()}
        pe = (P["v2"] + P["v3"]) / 2
        le = L["v2"] + L["v3"]
        tpe, tle = CLASS_NAMES[int(pe.argmax())], CLASS_NAMES[int(le.argmax())]
        for key, pred in (("v2", t2["v2"]), ("v3", t2["v3"]),
                          ("prob-ens", tpe), ("logit-ens", tle)):
            wins[key] += int(pred == truth)
        print(f"{p:34} truth={truth:18} v2={t2['v2']:18} v3={t2['v3']:18} "
              f"prob={tpe:18} logit={tle}")
    print()
    for k, w in wins.items():
        print(f"{k:10}: {w}/6")


if __name__ == "__main__":
    main()
