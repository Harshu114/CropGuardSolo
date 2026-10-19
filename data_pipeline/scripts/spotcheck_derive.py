"""Spot-check the PlantSeg->CropGuard derivation by eye.

Requirement: before trusting the derived corpus, render N random test images with
their lesion mask, leaf mask and the derived severity, and confirm the masks land
where they should. A rectangle-shaped "lesion" or a leaf mask that does not cover
the leaf means the derivation is wrong -- and that must be caught before any
number derived from it is reported.

Severity is recomputed here from the masks (lesion_px / leaf_px), NOT read from a
CSV, so this script is the independent check of the derivation rather than a
restatement of it. Grade cutoffs match configs/class_map.yaml.

  python3 data_pipeline/scripts/spotcheck_derive.py \
      --root ~/tmp/cgpack/cropguard_v7_split --split test --n 50 \
      --out ~/tmp/cgcheck
"""
import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw

# configs/class_map.yaml -> severity.grades (area-based scale)
GRADE_CUTOFFS = (0.05, 0.25)          # <5% mild, 5-25% moderate, >25% severe


def grade_of(sev):
    if sev < GRADE_CUTOFFS[0]:
        return 1, "mild"
    if sev < GRADE_CUTOFFS[1]:
        return 2, "moderate"
    return 3, "severe"


def load_mask(path, size):
    with Image.open(path) as im:
        m = im.convert("L").resize(size, Image.NEAREST)
    return np.asarray(m) > 127


def overlay(img, lesion, leaf):
    """image + leaf tinted green + lesion tinted red, lesion drawn on top."""
    base = np.asarray(img.convert("RGB")).astype(np.float32)
    out = base.copy()
    green = np.array([40, 200, 60], np.float32)
    red = np.array([240, 40, 40], np.float32)
    a = 0.45
    out[leaf] = out[leaf] * (1 - a) + green * a
    out[lesion] = out[lesion] * (1 - a) + red * a
    return Image.fromarray(out.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_dir = os.path.join(args.root, "images", args.split)
    les_dir = os.path.join(args.root, "masks_lesion", args.split)
    leaf_dir = os.path.join(args.root, "masks_leaf", args.split)
    names = sorted(os.listdir(img_dir))
    rng = random.Random(args.seed)
    pick = rng.sample(names, min(args.n, len(names)))
    os.makedirs(args.out, exist_ok=True)

    s = args.size
    tiles, stats = [], {"lesion_in_leaf": [], "leaf_cov": [], "sev": [],
                        "rect_like": []}
    for i, name in enumerate(pick):
        stem = os.path.splitext(name)[0]
        img = Image.open(os.path.join(img_dir, name)).convert("RGB").resize(
            (s, s), Image.BILINEAR)
        les = load_mask(os.path.join(les_dir, stem + ".png"), (s, s))
        leaf = load_mask(os.path.join(leaf_dir, stem + ".png"), (s, s))
        ov = overlay(img, les, leaf)

        les_px, leaf_px = int(les.sum()), int(leaf.sum())
        sev = les_px / max(leaf_px, 1)
        g, glabel = grade_of(sev)
        # fraction of lesion pixels that also lie inside the leaf mask: the
        # lesion belongs to the leaf, so this should be ~1.0.
        in_leaf = float((les & leaf).sum()) / max(les_px, 1) if les_px else 1.0
        leaf_cov = leaf_px / (s * s)
        # rectangle-ness: area vs bounding-box area. A filled detection box
        # scores 1.0 at every component; a real lesion blob does not.
        if les_px:
            ys, xs = np.where(les)
            bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            rect_like = les_px / (bh * bw)
        else:
            rect_like = 0.0
        stats["lesion_in_leaf"].append(in_leaf)
        stats["leaf_cov"].append(leaf_cov)
        stats["sev"].append(sev)
        stats["rect_like"].append(rect_like)

        # four panels, exactly what was asked for: the image, the LESION mask,
        # the LEAF mask, and the two combined -- plus the derived severity.
        tile = Image.new("RGB", (s * 4, s + 22), (18, 18, 18))
        tile.paste(img, (0, 22))
        tile.paste(Image.fromarray((les * 255).astype(np.uint8)).convert("RGB"),
                   (s, 22))
        tile.paste(Image.fromarray((leaf * 255).astype(np.uint8)).convert("RGB"),
                   (2 * s, 22))
        tile.paste(ov, (3 * s, 22))
        d = ImageDraw.Draw(tile)
        for j, cap in enumerate(("image", "lesion", "leaf", "lesion+leaf")):
            d.text((j * s + 4, 10), cap, fill=(150, 200, 255))
        txt = (f"{stem}  sev={sev:.3f} grade={g}({glabel})  "
               f"leaf_cov={leaf_cov:.2f} lesion_in_leaf={in_leaf:.2f} "
               f"rect={rect_like:.2f}")
        d.text((4, 0), txt, fill=(255, 255, 255))
        tiles.append(tile)
        tile.save(os.path.join(args.out, f"{i:02d}_{stem}.png"))

    n = len(tiles)
    cols = 2
    rows = (n + cols - 1) // cols
    th, tw = tiles[0].size
    sheet = Image.new("RGB", (tw * cols, th * rows), (10, 10, 10))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(os.path.join(args.out, "_contact_sheet.png"))

    def mean(k):
        return float(np.mean(stats[k])) if stats[k] else float("nan")
    print(f"rendered {n} overlays -> {args.out}")
    print(f"  mean lesion-in-leaf : {mean('lesion_in_leaf'):.3f}  (want ~1.0)")
    print(f"  mean leaf coverage  : {mean('leaf_cov'):.3f}")
    print(f"  mean severity       : {mean('sev'):.3f}")
    print(f"  mean rect-likeness  : {mean('rect_like'):.3f}  (want <<1.0; "
          f"~1.0 means masks are just detection boxes)")
    n_rect = sum(1 for r in stats["rect_like"] if r > 0.9)
    print(f"  images that are rectangle-like (>0.9): {n_rect}/{n}")


if __name__ == "__main__":
    main()
