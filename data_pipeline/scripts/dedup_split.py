"""Dedup + duplicate-aware splitting (expert: worth it).

PlantSeg/PlantVillage are scraped from the web, so near-duplicate images (same
leaf, several shots; re-uploads; mirrored copies) leak across train/val/test and
inflate every score. This script:

  1. computes a perceptual hash (dHash, 64-bit) per image,
  2. groups images whose hashes are within a Hamming radius (duplicates),
  3. assigns whole GROUPS to one split, so a duplicate can never straddle the
     train/test boundary,
  4. writes images/{train,val,test} + masks_lesion/{...} + labels/{...} in the
     layout the trainer expects.

Usage:
    python3 data_pipeline/scripts/dedup_split.py \
        --in crawlguard_v7 --out cropguard_v7_split \
        --val-frac 0.15 --test-frac 0.15 --hamming 6
Dry run (report only):
    python3 data_pipeline/scripts/dedup_split.py --in cropguard_v7 --report
"""
import argparse
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(img, size=8):
    """64-bit difference hash: robust to scaling/compression, cheap."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def group_images(paths, radius):
    """Union-find over hash neighbourhoods -> list of groups (index lists)."""
    hashes = []
    for p in paths:
        try:
            hashes.append(dhash(Image.open(p)))
        except Exception:
            hashes.append(None)
    parent = list(range(len(paths)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(len(paths)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(paths)):
            if hashes[j] is None:
                continue
            if hamming(hashes[i], hashes[j]) <= radius:
                union(i, j)
    groups = defaultdict(list)
    for i in range(len(paths)):
        groups[find(i)].append(i)
    return list(groups.values()), hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    img_root = os.path.join(args.inp, "images")
    if not os.path.isdir(img_root):
        raise SystemExit(f"no images/ under {args.inp}")
    # gather images from any split layout the corpus already has
    paths = []
    for dirpath, _, fnames in os.walk(img_root):
        if os.path.basename(dirpath) == "negatives":
            continue
        for f in sorted(fnames):
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(dirpath, f))
    print(f"{len(paths)} images")
    groups, hashes = group_images(paths, args.hamming)
    n_dup = sum(len(g) - 1 for g in groups)
    print(f"{len(groups)} unique groups; {n_dup} near-duplicate images "
          f"({n_dup / max(len(paths), 1):.1%}) at hamming<={args.hamming}")
    big = sorted((len(g) for g in groups), reverse=True)[:5]
    print(f"largest duplicate groups: {big}")
    if args.report:
        return

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(groups))
    n_val = int(len(groups) * args.val_frac)
    n_test = int(len(groups) * args.test_frac)
    split_of = {}
    for k, gi in enumerate(order):
        split = "train"
        if k < n_val:
            split = "val"
        elif k < n_val + n_test:
            split = "test"
        for i in groups[gi]:
            split_of[i] = split

    for split in ("train", "val", "test"):
        for sub in ("images", "masks_lesion", "masks_leaf", "labels"):
            os.makedirs(os.path.join(args.out, sub, split), exist_ok=True)
    counts = defaultdict(int)
    for i, p in enumerate(paths):
        split = split_of[i]
        stem = os.path.splitext(os.path.basename(p))[0]
        counts[split] += 1
        shutil.copy2(p, os.path.join(args.out, "images", split,
                                     os.path.basename(p)))
        for sub, ext in (("masks_lesion", ".png"), ("masks_leaf", ".png"),
                         ("labels", ".txt")):
            src = os.path.join(args.inp, sub, "")
            match = None
            for dirpath, _, fnames in os.walk(os.path.join(args.inp, sub)):
                if stem + ext in fnames:
                    match = os.path.join(dirpath, stem + ext)
                    break
            if match:
                shutil.copy2(match, os.path.join(args.out, sub, split,
                                                 stem + ext))
    print("split:", dict(counts))
    print(f"wrote -> {args.out} (duplicates kept inside one split)")


if __name__ == "__main__":
    main()
