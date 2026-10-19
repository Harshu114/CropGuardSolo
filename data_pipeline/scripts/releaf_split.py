"""Recompute masks_leaf/ for a derived split with the fixed leaf segmenter.

The split already contains the source images and the lesion masks, so a broken
leaf mask can be repaired in place without re-running the whole (Kaggle) derive.

    python3 data_pipeline/scripts/releaf_split.py --root ~/tmp/cgpack/cropguard_v7_split

Prints, per split, the before/after lesion-in-leaf overlap and the severity range
so the repair is visible rather than assumed.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_dataset import grabcut_leaf_mask, leaf_quality  # noqa: E402


def stats(root, split):
    img_d = os.path.join(root, "images", split)
    les_d = os.path.join(root, "masks_lesion", split)
    lf_d = os.path.join(root, "masks_leaf", split)
    inleaf, sev, cov = [], [], []
    for n in sorted(os.listdir(img_d)):
        stem = os.path.splitext(n)[0]
        l = cv2.imread(os.path.join(les_d, stem + ".png"), 0)
        f = cv2.imread(os.path.join(lf_d, stem + ".png"), 0)
        if l is None or f is None:
            continue
        lb, fb = l > 127, f > 127
        lp = int(lb.sum())
        if lp == 0:
            continue
        inleaf.append(float((lb & fb).sum()) / lp)
        sev.append(lp / max(int(fb.sum()), 1))
        cov.append(leaf_quality(f)[0])
    return (np.mean(inleaf), np.percentile(sev, [50, 95, 100]),
            np.mean(cov), len(inleaf))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    for split in args.splits.split(","):
        img_d = os.path.join(args.root, "images", split)
        les_d = os.path.join(args.root, "masks_lesion", split)
        lf_d = os.path.join(args.root, "masks_leaf", split)
        names = sorted(os.listdir(img_d))
        print(f"[{split}] {len(names)} images", flush=True)
        n_fixed = n_same = 0
        for i, n in enumerate(names):
            stem = os.path.splitext(n)[0]
            # PIL, not cv2: the derive wrote the masks against PIL's (no-EXIF)
            # orientation, but cv2.imread silently rotates by EXIF, which made
            # image and mask shapes disagree and produced transposed output.
            rgb = np.asarray(Image.open(os.path.join(img_d, n)).convert("RGB"))
            lesion = cv2.imread(os.path.join(les_d, stem + ".png"), 0)
            if lesion is not None and lesion.shape[:2] != rgb.shape[:2]:
                lesion = cv2.resize(lesion, (rgb.shape[1], rgb.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            new = grabcut_leaf_mask(rgb, lesion)
            out = os.path.join(lf_d, stem + ".png")
            old = cv2.imread(out, 0)
            if old is not None and (old > 127).sum() == (new > 127).sum():
                n_same += 1
            else:
                n_fixed += 1
            cv2.imwrite(out, new)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(names)}", flush=True)
        a = stats(args.root, split)
        print(f"[{split}] fixed={n_fixed} unchanged={n_same}")
        print(f"[{split}] mean lesion-in-leaf = {a[0]:.3f}  "
              f"sev  p50/p95/max = {a[1].round(3)}  mean leaf_cov = {a[2]:.3f}")


if __name__ == "__main__":
    main()
