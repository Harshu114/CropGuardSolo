#!/usr/bin/env python3
"""DIAGNOSTIC: per-class accuracy + confusion matrix of the v5 checkpoint on its
OWN held-out test split (cropguard-corpus-claude/test, 1286 images).

Decisive question: is bacterial_spot / early_blight / septoria actually confused
by the model on data it was trained on, or do the failures only happen on the
user's downloaded web images (a domain/appearance gap)?

Uses the exact serving forward pass (flip-TTA averaged logits) so the numbers
are comparable to what the website produces.

Usage: python _diag_perclass.py [root]
"""
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

import model_server as ms
from dataset import CropGuardDataset, collate

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "cropguard_model_v5.pt")
ROOT = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/logiclocus/HarshalPrivate/Projects/collegeProjects/"
    "ClaudeCropGuard/datasetclaude")

NAMES = ["healthy", "bacterial_spot", "early_blight", "late_blight",
         "leaf_miner", "leaf_mold", "mosaic_virus", "septoria_leaf_spot",
         "spider_mites", "target_spot", "yellow_leaf_curl_virus"]

torch.set_grad_enabled(False)


def main() -> int:
    ds = CropGuardDataset(ROOT, split="test", size=ms.IMG_SIZE, augment=False)
    print(f"test images: {len(ds)}   root={ROOT}")

    model = ms.load_model(CKPT)
    device = ms.get_device()
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate)

    cm = np.zeros((11, 11), dtype=int)      # cm[true, pred]
    n_skipped = 0
    for bi, batch in enumerate(loader):
        x = batch["image"].to(device)
        y = batch["cls"]
        valid = y >= 0
        n_skipped += int((~valid).sum())
        if valid.sum() == 0:
            continue
        xf = torch.flip(x, dims=[3])
        lo = model(x, with_severity=False)["cls_logits"]
        lf = model(xf, with_severity=False)["cls_logits"]
        logits = 0.5 * (lo + lf)            # same averaging as serving TTA
        pred = logits.argmax(1).cpu()
        yv = y[valid]
        pv = pred[valid]
        for t, p in zip(yv.tolist(), pv.tolist()):
            cm[t, p] += 1
        if (bi + 1) % 20 == 0:
            print(f"  ...{min((bi+1)*16, len(ds))}/{len(ds)}")

    total = cm.sum()
    correct = np.trace(cm)
    print(f"\noverall top-1: {correct}/{total} = {correct/total:.4f}   "
          f"(skipped background/images without boxes: {n_skipped})")

    print(f"\n{'cls':<2} {'name':<24} {'n':>5} {'correct':>8} {'acc':>7}   "
          f"top confusions (true -> predicted)")
    for t in range(11):
        n = cm[t].sum()
        if n == 0:
            print(f"{t:<2} {NAMES[t]:<24} {0:>5}        -       -   (none in test split)")
            continue
        c = cm[t, t]
        others = [(cm[t, p], p) for p in range(11) if p != t and cm[t, p] > 0]
        others.sort(reverse=True)
        conf = ", ".join(f"{NAMES[p]} {v}" for v, p in others[:4]) or "-"
        print(f"{t:<2} {NAMES[t]:<24} {n:>5} {c:>8} {c/n:>7.3f}   {conf}")

    print("\nfull confusion matrix (rows=true, cols=predicted), "
          "columns = " + " ".join(f"{i}:{NAMES[i][:6]}" for i in range(11)))
    for t in range(11):
        if cm[t].sum() == 0:
            continue
        print(f"  {t:>2} " + " ".join(f"{v:>7}" for v in cm[t]))

    worst = sorted(
        ((cm[t, t] / cm[t].sum() if cm[t].sum() else 1.0, t, cm[t].sum())
         for t in range(11)),
    )
    print("\nworst classes by accuracy:")
    for acc, t, n in worst[:5]:
        print(f"  {NAMES[t]:<24} acc={acc:.3f}  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
