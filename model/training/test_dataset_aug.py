"""CPU test for dataset.py geometric augmentation.

Two parts:
  A) SYNTHETIC: an image containing a bright square whose lesion and leaf masks
     cover exactly the same square. Run the augmentation (flip x scale) and
     assert the square's centroid in the image and in BOTH masks agree within
     2 px. This is exactly what was broken before: masks were left untransformed
     (or the flip was hardcoded off), so seg targets were misaligned.
  B) REAL: 20 train samples through the full __getitem__ with random aug;
     mean fraction of lesion pixels falling inside the leaf mask must be >= 0.9.

Usage:  python model/training/test_dataset_aug.py [corpus_root]
Exit code 0 = all PASS.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import CropGuardDataset  # noqa: E402

W, H = 120, 80            # deliberately non-square to catch x/y transposition
X1, X2 = 30, 70
Y1, Y2 = 20, 60
TOL = 2.0

failures = []


def centroid(a):
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def expected_centroid(flip, s, cx, cy):
    """Where the square's centre must land: same math the boxes use."""
    if flip:
        cx = W - 1 - cx
    if abs(s - 1.0) > 1e-3:
        if s < 1.0:                      # resize then centre-pad
            cx = cx * s + (W - W * s) / 2.0
            cy = cy * s + (H - H * s) / 2.0
        else:                            # resize then centre-crop
            cx = cx * s - (W * s - W) / 2.0
            cy = cy * s - (H * s - H) / 2.0
    return cx, cy


def make_synth():
    arr = np.zeros((H, W, 3), np.uint8)
    arr[Y1:Y2, X1:X2] = 255
    img = Image.fromarray(arr)
    m = np.zeros((H, W), np.uint8)
    m[Y1:Y2, X1:X2] = 255
    return img, m.copy(), m.copy()


def part_a(ds):
    print("=== PART A: synthetic square, image vs lesion vs leaf ===")
    cx0, cy0 = (X1 + X2 - 1) / 2.0, (Y1 + Y2 - 1) / 2.0
    for flip in (False, True):
        for s in (1.0, 0.8):
            img, les, leaf = make_synth()
            img2, _boxes, les2, leaf2, _c = ds._augment(
                img, [], les, leaf, None, flip=flip, scale=s)
            # threshold ABOVE the 114 grey pad colour, so the pad is not
            # counted as part of the bright square
            ci = centroid(np.asarray(img2.convert("L")) > 200)
            cl = centroid(les2 > 127)
            cf = centroid(leaf2 > 127)
            ex, ey = expected_centroid(flip, s, cx0, cy0)
            ok = True
            why = []
            for name, c in (("image", ci), ("lesion", cl), ("leaf", cf)):
                if c is None:
                    ok, why = False, why + [f"{name} empty"]
                    continue
                d = np.hypot(c[0] - ex, c[1] - ey)
                if d > TOL:
                    ok = False
                    why.append(f"{name} off by {d:.2f}px (got {c}, want {(ex, ey)})")
            # masks must also agree with each other
            if cl and cf and np.hypot(cl[0] - cf[0], cl[1] - cf[1]) > TOL:
                ok = False
                why.append("lesion vs leaf disagree")
            if cl and ci and np.hypot(cl[0] - ci[0], cl[1] - ci[1]) > TOL:
                ok = False
                why.append("lesion vs image disagree")
            print(f"  flip={flip!s:5s} scale={s:<4} "
                  f"img={None if ci is None else tuple(round(v,1) for v in ci)} "
                  f"les={None if cl is None else tuple(round(v,1) for v in cl)} "
                  f"leaf={None if cf is None else tuple(round(v,1) for v in cf)} "
                  f"-> {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(f"A flip={flip} scale={s}: {'; '.join(why)}")


def part_b(root):
    print("=== PART B: 20 real train samples, lesion inside leaf ===")
    ds = CropGuardDataset(root, split="train", size=320, augment=True)
    fracs, used = [], 0
    for i in range(min(20, len(ds))):
        s = ds[i]
        les = s["mask"].numpy() > 0.5
        leaf = s["leaf"].numpy()
        valid = leaf >= 0.0
        if les.sum() < 1 or not valid.any():
            continue
        inside = les & (leaf > 0.5)
        fracs.append(inside.sum() / les.sum())
        used += 1
    if not fracs:
        print("  FAIL: no sample with lesion pixels found")
        failures.append("B: no lesion pixels")
        return
    mean = float(np.mean(fracs))
    ok = mean >= 0.9 and min(fracs) >= 0.5
    print(f"  n={used}  mean lesion-inside-leaf={mean:.4f}  min={min(fracs):.4f}"
          f"  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"B: mean={mean:.4f} min={min(fracs):.4f}")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "CROPGUARD_DATA", os.path.expanduser("~/tmp/cgpack/cropguard_v7_split"))
    ds = CropGuardDataset.__new__(CropGuardDataset)   # geometry only, no dirs
    ds.size = 320
    part_a(ds)
    part_b(root)
    print("=" * 60)
    if failures:
        print(f"OVERALL: FAIL ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
