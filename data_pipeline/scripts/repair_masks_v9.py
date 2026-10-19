"""Repair the v7 corpus masks -> v9 corpus (single source of truth).

Two defects found on 2026-10-12 (see RUNLOG):
  1. LESION masks are 1-px polygon OUTLINES (prepare_plantseg.py poly_to_mask
     walked edges, never filled interiors). Fix: dilate -> flood-fill outside
     from border -> region = dilated | interior -> erode back.
  2. LEAF masks (grabcut) are HEALTHY-TISSUE-ONLY: necrotic areas are holes /
     bays, so lesion (a disease region) is mostly OUTSIDE the leaf mask. The
     leaf channel target and the severity denominator (lesion_px/leaf_px) both
     need the WHOLE leaf. Fix: fill_holes(leaf) | lesion_filled.

Output corpus layout: masks_lesion/, masks_leaf/ (real PNGs), images/, labels/
(symlinks to the v7 corpus by default). The v7 corpus is never modified.

Usage:
    python3 data_pipeline/scripts/repair_masks_v9.py \
        --v7 /home/logiclocus/tmp/cgpack/cropguard_v7_split \
        --out /home/logiclocus/tmp/cgpack/cropguard_v9_fixed
"""
import argparse
import os

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

SPLITS = ("train", "val", "test")
DEFAULT_V7 = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
DEFAULT_OUT = "/home/logiclocus/tmp/cgpack/cropguard_v9_fixed"


def fill_outline(b):
    """b: binary uint8 (0/1) -> solid region mask.

    Correct order for 1px open-ish curves: dilate (seal gaps) -> flood-fill the
    OUTSIDE from the border -> region = dilated | interior -> erode back.
    (morphological closing alone is ~identity on 1px lines, it does not seal.)
    """
    if b.sum() == 0:
        return b, 0
    for ksize in (3, 5, 7):
        k = np.ones((ksize, ksize), np.uint8)
        bd = cv2.dilate(b, k)
        ff = bd.copy()
        h, w = ff.shape
        if ff[0, 0] == 1:
            ys, xs = np.where(ff == 0)
            if len(ys) == 0:
                return cv2.erode(bd, k), ksize
            x0, y0 = int(xs[0]), int(ys[0])
        else:
            x0, y0 = 0, 0
        m2 = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, m2, (x0, y0), 1)
        filled_d = bd | (1 - ff)
        interior_gain = int(filled_d.sum()) - int(bd.sum())
        if interior_gain > 0.05 * int(bd.sum()) or ksize == 7:
            filled = cv2.erode(filled_d, k)
            if int((filled & b).sum()) == 0:
                filled = filled | b
            return filled, ksize
    return b, 7


def load_exclusions(v7):
    """leaf_exclusion_list.txt lives at the corpus root, one level up, or the
    repo root (where dataset.py's own search finds it on the laptop)."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    for cand in (os.path.join(v7, "leaf_exclusion_list.txt"),
                 os.path.join(os.path.dirname(v7), "leaf_exclusion_list.txt"),
                 os.path.join(repo, "leaf_exclusion_list.txt")):
        if os.path.isfile(cand):
            with open(cand) as f:
                return {l.strip() for l in f if l.strip()}, cand
    return set(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v7", default=DEFAULT_V7)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    v7, out = args.v7, args.out

    os.makedirs(out, exist_ok=True)
    for d in ("images", "labels"):
        p = os.path.join(out, d)
        if not os.path.exists(p):
            os.symlink(os.path.join(v7, d), p)

    excl, excl_path = load_exclusions(v7)
    print(f"leaf exclusions: {len(excl)} stems ({excl_path})")

    def label_cls(split, stem):
        p = os.path.join(v7, "labels", split, stem + ".txt")
        if not os.path.exists(p):
            return -1
        cs = []
        for line in open(p):
            q = line.split()
            if len(q) >= 5:
                try:
                    cs.append(int(float(q[0])))
                except ValueError:
                    pass
        return int(np.bincount(cs, minlength=11).argmax()) if cs else -1

    for split in SPLITS:
        src_les = os.path.join(v7, "masks_lesion", split)
        src_leaf = os.path.join(v7, "masks_leaf", split)
        dst_les = os.path.join(out, "masks_lesion", split)
        dst_leaf = os.path.join(out, "masks_leaf", split)
        os.makedirs(dst_les, exist_ok=True)
        os.makedirs(dst_leaf, exist_ok=True)

        n = k3 = leaf_changed = 0
        ratios, inside = [], []
        for fn in sorted(os.listdir(src_les)):
            m = cv2.imread(os.path.join(src_les, fn), 0)
            if m is None:
                continue
            b = (m > 127).astype(np.uint8)
            filled, which = fill_outline(b)
            if which:
                k3 += 1
            if b.sum():
                ratios.append(float(filled.sum()) / float(b.sum()))
            cv2.imwrite(os.path.join(dst_les, fn), filled * 255)

            lp = os.path.join(src_leaf, fn)
            leaf = cv2.imread(lp, 0)
            if leaf is None:
                leaf_fix = filled.copy()
            else:
                lf = binary_fill_holes(leaf > 127)
                leaf_fix = (lf | (filled > 0)).astype(np.uint8)
                if not np.array_equal(leaf_fix, (leaf > 127)):
                    leaf_changed += 1
            cv2.imwrite(os.path.join(dst_leaf, fn), leaf_fix * 255)
            if filled.sum():
                inside.append(float((filled & leaf_fix).sum()) / filled.sum())
            n += 1

        r = np.array(ratios) if ratios else np.array([0.0])
        i_ = np.array(inside) if inside else np.array([0.0])
        print(f"{split}: n={n}  lesion filled/outline median={np.median(r):.1f}x  "
              f"leaf changed={leaf_changed}  lesion-inside-leaf(union) "
              f"mean={i_.mean():.4f} min={i_.min():.4f}")
        assert i_.min() > 0.999, f"{split}: lesion must be inside leaf after union"
        assert np.median(r) >= 5.0, f"{split}: fill did not grow outlines"

    print("\n=== severity (lesion_filled / leaf_union), dataset.py rules ===")
    # Protocol parity with the trainer's val severity metric: label-valid stems
    # only (label audit), leaf-excluded stems emit -1 and are skipped, every
    # other stem carries sev_cont = clip(lesion_px/leaf_px) (0.0 when !has:
    # healthy or no lesion), so healthy images count as 0.0 in the MAE set.
    for split in SPLITS:
        vals_all, vals_ex = [], []
        for fn in sorted(os.listdir(os.path.join(out, "masks_lesion", split))):
            stem = os.path.splitext(fn)[0]
            cls = label_cls(split, stem)
            if cls not in range(8):          # label audit exclusion
                continue
            leaf = cv2.imread(os.path.join(out, "masks_leaf", split, fn), 0)
            leaf_px = int((leaf > 127).sum()) if leaf is not None else 320 * 320
            mn = cv2.imread(os.path.join(out, "masks_lesion", split, fn), 0)
            ln = int((mn > 127).sum()) if mn is not None else 0
            has = leaf_px > 0 and (ln > 0 or cls > 0) and True  # cls >= 0 holds
            sev = float(np.clip(ln / max(leaf_px, 1), 0.0, 1.0)) if has else 0.0
            vals_all.append(sev)
            if stem not in excl:             # leaf exclusion -> sev_cont = -1
                vals_ex.append(sev)
        v = np.array(vals_ex)                # the trainer's actual set
        print(f"{split:5s} n={len(v)} (label-valid, leaf-excl removed) "
              f"mean={v.mean():.4f} p50={np.median(v):.4f} "
              f"baselineMAE(mean)={float(np.abs(v - v.mean()).mean()):.4f}  "
              f"[incl-leaf-excl n={len(vals_all)}]")
    print("\nv9 corpus written:", out)


if __name__ == "__main__":
    main()
