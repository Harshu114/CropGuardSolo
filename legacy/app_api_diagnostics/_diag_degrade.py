#!/usr/bin/env python3
"""DIAGNOSTIC: does image degradation reproduce the user's failure rate?

The user's downloaded web images are far smaller than the corpus (148x148 / 4-10 KB
vs 320x320). This takes the corpus TEST split (where v5 scores 0.861 top-1) and
re-renders it progressively worse, then re-measures top-1.

If accuracy collapses at the resolution the user's images have, then the failures
are an input-quality problem, not a model-capability problem.

Builds its own small dataset under the project tree (NOT /tmp -- that gets wiped):
  _diag_small/{variant}/test/images/*.jpg   + test/labels symlinked to the corpus

Usage: python _diag_degrade.py
"""
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import model_server as ms
from dataset import CropGuardDataset, collate

HERE = Path(__file__).resolve().parent
CKPT = HERE / "cropguard_model_v5.pt"
CORPUS = Path("/home/logiclocus/HarshalPrivate/Projects/collegeProjects/"
              "ClaudeCropGuard/datasetclaude")
WORK = HERE / "_diag_small"

VARIANTS = [
    ("native", None),          # untouched 320x320, baseline
    ("s224", 224),
    ("s160", 160),             # ~ the user's 148x148 thumbnails
    ("s96", 96),
    ("s160_jpg50", 160),       # + heavy JPEG compression
]

torch.set_grad_enabled(False)


def build(variant: str, side: int, jpeg_q: int = 95) -> Path:
    """Render the corpus test split at a new size; return the dataset root."""
    root = WORK / variant
    img_dir = root / "test" / "images"
    lbl_dir = root / "test" / "labels"
    if root.exists():
        shutil.rmtree(root)
    img_dir.mkdir(parents=True)
    src_lbl = CORPUS / "labels" / "test"
    lbl_dir.symlink_to(src_lbl.resolve())

    for p in (CORPUS / "images" / "test").iterdir():
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB")
        if side:
            im = im.resize((side, side), Image.BICUBIC)
        im.save(img_dir / (p.stem + ".jpg"), "JPEG", quality=jpeg_q)
    return root


def accuracy(root: Path, model) -> float:
    ds = CropGuardDataset(str(root), split="test", size=ms.IMG_SIZE,
                          augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate)
    device = ms.get_device()
    ok = tot = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["cls"]
        v = y >= 0
        if v.sum() == 0:
            continue
        lo = model(x, with_severity=False)["cls_logits"]
        lf = model(torch.flip(x, dims=[3]), with_severity=False)["cls_logits"]
        pred = (0.5 * (lo + lf)).argmax(1).cpu()[v]
        ok += int((pred == y[v]).sum())
        tot += int(v.sum())
    return ok / max(tot, 1)


def main() -> int:
    model = ms.load_model(str(CKPT))
    print(f"model={ms.get_model_version()}  IMG_SIZE={ms.IMG_SIZE}\n")
    print(f"{'variant':<12} {'render':<14} {'top1':>7}")
    print("-" * 36)
    for name, side in VARIANTS:
        q = 50 if name.endswith("jpg50") else 95
        root = build(name, side, q)
        acc = accuracy(root, model)
        desc = "320x320 q95" if not side else f"{side}x{side} q{q}"
        print(f"{name:<12} {desc:<14} {acc:>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
