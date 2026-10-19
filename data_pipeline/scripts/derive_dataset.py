"""Derive every label for a single-dataset CropGuard corpus (v7).

Input  : images/ + masks/ where masks are LESION masks (255 = lesion, e.g. the
         polygons PlantSeg provides). Images are named
         `<host>_<disease>_<n>.jpg`.
Output : images/, masks_lesion/, masks_leaf/, labels/, severity_<split>.csv,
         data.yaml -- so one training run covers all heads.

Label sources:
  classification  disease name -> CropGuard class via class_map.yaml (VERIFIED;
                  unmapped names are skipped loudly, never guessed)
  detection       connected components of the LESION mask -> YOLO boxes
  segmentation    lesion mask (given) + leaf mask (derived)
  severity        lesion_px / leaf_px, ONLY on close-up single-leaf images
                  (expert: the ratio is ill-defined on wide shots). Others get
                  -1 = "not assessed" instead of a meaningless number.
  negatives       optional non-tomato hosts -> "not_tomato" set for the
                  abstention / OOD head

Leaf mask backend: `grabcut` (default, no extra deps) or `sam` (better, needs
the segment_anything package + checkpoint; cache the output, it is slow).

Usage:
    python3 data_pipeline/scripts/derive_dataset.py \
        --in plantseg_tomato --out cropguard_v7 \
        --negatives-in plantseg_other_hosts
Self-test (no data needed):
    python3 data_pipeline/scripts/derive_dataset.py --self-test
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter

import cv2
import numpy as np
import yaml
from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MAP = os.path.join(_ROOT, "data_pipeline", "configs", "class_map.yaml")


def parse_names(file_name):
    """'tomato_bacterial_leaf_spot_12.jpg' -> ('tomato', 'bacterial_leaf_spot')."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if "_" not in stem:
        return None, None
    host, rest = stem.split("_", 1)
    disease = re.sub(r"(_google)?_\d+$", "", rest)
    disease = re.sub(r"[^a-z0-9]+", "_", disease.lower()).strip("_")
    return host, (disease or None)


def load_map(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    classes = [cfg["classes"][i] for i in sorted(cfg["classes"])]
    return cfg, classes


def verify_plantseg_classes(annos_dir, cfg, classes):
    """Print every PlantSeg disease name that this map does not cover.

    Reading the host names out of an annotation JSON/CSV listing is enough to
    catch a mismatch before a 5-hour run. Returns (mapped, unmapped) counters.
    """
    names = set()
    for split in ("train", "val", "test"):
        for ext in (".json", ".csv", ".txt"):
            p = os.path.join(annos_dir, f"annotation_{split}{ext}")
            if not os.path.exists(p):
                continue
            text = open(p, errors="ignore").read()
            for m in re.findall(r"([a-z0-9]+_[a-z0-9_]+_\d+)\.jpg", text):
                _, dis = parse_names(m)
                if dis:
                    names.add(dis)
    mapped = {n for n in names if n in cfg["plantseg_to_cropguard"]}
    unmapped = names - mapped
    print(f"class map check: {len(mapped)} mapped, {len(unmapped)} unmapped")
    if unmapped:
        print("  UNMAPPED (add them to class_map.yaml or they will be skipped):")
        for n in sorted(unmapped):
            print(f"    - {n}")
    return len(mapped), len(unmapped)


# ------------------------------------------------------------ leaf masks ----
def grabcut_leaf_mask(rgb, lesion=None, iters=3):
    """OpenCV GrabCut for the leaf region (no extra deps).

    `lesion` (the PlantSeg annotation, uint8 0/255) is used TWO ways, both
    deliberate:

      * as a CERTAIN-foreground seed  -- expert annotation says this pixel is a
        diseased spot on a tomato leaf, so it is leaf tissue, full stop;
      * as a guaranteed union        -- the returned mask ALWAYS contains the
        lesion, because a lesion is part of a leaf.

    Why this is not optional: seeding only from green pixels makes GrabCut mark
    necrotic (brown/yellow) tissue as background, so the leaf mask was missing
    the very region that defines severity = lesion_px / leaf_px. On the derived
    test split 95/130 images had lesion pixels OUTSIDE the leaf mask (median
    overlap 0.64) and severity reached 4.04 -- lesion 4x the leaf, physically
    impossible. Unioning the annotation back in fixes the denominator.
    """
    h, w = rgb.shape[:2]
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    green = ((g > r + 10) & (g > b + 10) & (g > 50)).astype(np.uint8)
    if green.sum() < 200:
        green = ((g > r) & (g > b)).astype(np.uint8)
    les = None
    if lesion is not None:
        les = (lesion > 127).astype(np.uint8)
    m = np.zeros((h, w), np.uint8)
    m[green > 0] = cv2.GC_PR_FGD
    if les is not None:
        m[les > 0] = cv2.GC_FGD          # annotation wins over the green test
    m[: max(2, h // 40), :] = cv2.GC_BGD
    m[-max(2, h // 40):, :] = cv2.GC_BGD
    m[:, : max(2, w // 40)] = cv2.GC_BGD
    m[:, -max(2, w // 40):] = cv2.GC_BGD
    try:
        cv2.grabCut(img, m, None, np.zeros((1, 65), np.float64), 5, iters,
                    cv2.GC_INIT_WITH_MASK)
        out = np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        out = (green > 0).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.where(lab == keep, 255, 0).astype(np.uint8)
    if les is not None:
        # union AFTER component selection, so a lesion that sits off the main
        # green blob (heavy necrosis) is not discarded as a stray component
        out = np.where((out > 0) | (les > 0), 255, 0).astype(np.uint8)
    return out


def sam_leaf_mask(rgb, sam_ckpt, device="cuda"):
    """Segment-anything leaf mask: mask with the most green coverage."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    key = "vit_b" if "vit_b" in os.path.basename(sam_ckpt) else "vit_h"
    sam = sam_model_registry[key](checkpoint=sam_ckpt).to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=16, pred_iou_thresh=0.86,
        stability_score_thresh=0.90, min_mask_region_area=200)
    masks = gen.generate(rgb)
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    green = (g > r + 10) & (g > b + 10)
    best, best_score = None, -1.0
    for m in masks:
        mm = m["segmentation"]
        score = (mm & green).sum() / max(green.sum(), 1)
        if score > best_score:
            best_score, best = score, mm
    if best is None:
        return grabcut_leaf_mask(rgb)
    return best.astype(np.uint8) * 255


def leaf_quality(leaf_mask):
    """(coverage, n_components) -- the close-up test inputs."""
    m = (leaf_mask > 127).astype(np.uint8)
    coverage = float(m.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    big = int((stats[1:, cv2.CC_STAT_AREA] > 0.02 * m.size).sum()) if n > 1 else 0
    return coverage, big


# ------------------------------------------------------ boxes from masks ----
def boxes_from_lesion_mask(lesion, min_area=24, max_components=8):
    comps = (lesion > 127).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(comps, 8)
    h, w = lesion.shape[:2]
    rows = []
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:max_components]:
        x, y, bw, bh, area = stats[i + 1]
        if area < min_area:
            continue
        rows.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return rows


def mask_to_boxes(mask, min_area=24, max_components=8):
    """Public helper: predicted-mask -> boxes. Same code the detector's
    YOLO boxes are derived from, so inference-time 'boxes for free' uses the
    identical geometry (expert: the detector is largely redundant when boxes
    come from masks)."""
    return boxes_from_lesion_mask(mask, min_area, max_components)


def grade_of(sev, cfg):
    """Map a severity ratio to the configured area-based grade (1/2/3)."""
    g = cfg["severity"]["grades"]
    if sev < g[1]:
        return 1
    if sev < g[2]:
        return 2
    return 3


# ------------------------------------------------------------------ main ----
def self_test():
    img = np.full((240, 320, 3), (60, 120, 40), np.uint8)
    cv2.ellipse(img, (160, 120), (110, 70), 25, 0, 360, (30, 90, 30), -1)
    cv2.circle(img, (150, 110), 28, (40, 40, 70), -1)
    les = np.zeros((240, 320), np.uint8)
    cv2.circle(les, (150, 110), 28, 255, -1)
    leaf = grabcut_leaf_mask(img)
    cov, ncomp = leaf_quality(leaf)
    print(f"grabcut: coverage {cov:.2%}, components {ncomp} -> "
          f"{'close-up' if cov >= 0.55 and ncomp <= 1 else 'NOT close-up'}")
    rows = boxes_from_lesion_mask(les)
    print(f"boxes: {len(rows)} rows, first {rows[0] if rows else None}")
    cfg, _ = load_map(DEFAULT_MAP)
    sev = float((les > 0).sum()) / max(float((leaf > 0).sum()), 1)
    print(f"severity {sev:.3f} -> grade {grade_of(sev, cfg)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="corpus root with images/ + masks/")
    ap.add_argument("--out", default="cropguard_v7")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--leaf-backend", choices=["grabcut", "sam"], default="grabcut")
    ap.add_argument("--sam-ckpt", default=None)
    ap.add_argument("--min-area", type=int, default=24)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--negatives-in", default=None,
                    help="dir of non-tomato host images -> not_tomato negatives")
    ap.add_argument("--max-negatives-per-host", type=int, default=None)
    ap.add_argument("--check-annos", default=None,
                    help="PlantSeg annotations dir: verify class mapping only")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    cfg, classes = load_map(args.map)
    if args.self_test:
        sys.exit(self_test())
    if args.check_annos:
        _, unmapped = verify_plantseg_classes(args.check_annos, cfg, classes)
        sys.exit(1 if unmapped else 0)
    if not args.inp:
        ap.error("--in is required unless --self-test/--check-annos")

    name2id = {n: i for i, n in enumerate(classes)}
    print(f"{len(classes)} classes: {classes}")

    skipped = Counter()
    for split in args.splits.split(","):
        out_img = os.path.join(args.out, "images", split)
        out_les = os.path.join(args.out, "masks_lesion", split)
        out_lf = os.path.join(args.out, "masks_leaf", split)
        out_lbl = os.path.join(args.out, "labels", split)
        for d in (out_img, out_les, out_lf, out_lbl):
            os.makedirs(d, exist_ok=True)

        sev_rows = []
        img_dir = os.path.join(args.inp, "images", split)
        mask_dir = os.path.join(args.inp, "masks", split)
        if not os.path.isdir(img_dir):
            print(f"  [warn] no images for split {split!r}")
            continue
        files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(IMG_EXTS))
        n_close = 0
        for f in files:
            stem = os.path.splitext(f)[0]
            _, disease = parse_names(f)
            cg = cfg["plantseg_to_cropguard"].get(disease) if disease else None
            if cg is None or cg not in name2id:
                skipped[disease] += 1
                continue
            mp = os.path.join(mask_dir, stem + ".png")
            if not os.path.exists(mp):
                skipped["no_mask"] += 1
                continue
            img = np.asarray(Image.open(os.path.join(img_dir, f)).convert("RGB"))
            lesion = cv2.imread(mp, 0)
            if lesion.shape[:2] != img.shape[:2]:
                lesion = cv2.resize(lesion, img.shape[1::-1],
                                    interpolation=cv2.INTER_NEAREST)
            leaf = (sam_leaf_mask(img, args.sam_ckpt)
                    if args.leaf_backend == "sam" and args.sam_ckpt
                    else grabcut_leaf_mask(img, lesion))
            # INVARIANT: a lesion is part of a leaf, so the leaf mask must
            # contain the lesion. Enforced here (backend-independent) because
            # severity = lesion_px / leaf_px is nonsense otherwise -- the
            # earlier pure-GrabCut masks missed necrotic tissue and produced
            # severity values above 1.0.
            if (lesion > 127).any():
                leaf = np.where((leaf > 127) | (lesion > 127), 255, 0).astype(
                    np.uint8)

            rows = boxes_from_lesion_mask(lesion, args.min_area)
            cls_id = name2id[cg]
            with open(os.path.join(out_lbl, stem + ".txt"), "w") as fh:
                for cx, cy, bw, bh in rows:
                    fh.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            import shutil
            shutil.copy2(os.path.join(img_dir, f), os.path.join(out_img, f))
            cv2.imwrite(os.path.join(out_les, stem + ".png"),
                        np.where(lesion > 127, 255, 0).astype(np.uint8))
            cv2.imwrite(os.path.join(out_lf, stem + ".png"), leaf)

            # severity only on close-up single leaves (expert correction)
            cov, ncomp = leaf_quality(leaf)
            scfg = cfg["severity"]
            if cov >= scfg["min_leaf_coverage"] and ncomp <= scfg["max_leaf_components"]:
                leaf_px = float((leaf > 127).sum())
                sev = float((lesion > 127).sum()) / max(leaf_px, 1.0)
                sev = min(1.0, sev)
                grade = grade_of(sev, cfg)
                n_close += 1
            else:
                sev, grade = scfg["unassessed_value"], 0
            sev_rows.append((stem, sev, grade, cls_id))

        with open(os.path.join(args.out, f"severity_{split}.csv"), "w",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["image_id", "severity", "grade", "class_id"])
            for row in sev_rows:
                w.writerow([row[0], f"{row[1]:.6f}", row[2], row[3]])
        print(f"  {split}: {len(sev_rows)} images ({n_close} severity-assessed)")

    # ---- free negatives: other hosts -> not_tomato (abstention/OOD) ---------
    neg_cfg = cfg.get("negatives", {})
    max_per_host = args.max_negatives_per_host or neg_cfg.get("max_images_per_host", 200)
    if args.negatives_in and neg_cfg.get("enabled", True):
        out_neg = os.path.join(args.out, "images", "negatives")
        os.makedirs(out_neg, exist_ok=True)
        hosts = Counter()
        import shutil
        for dirpath, _, fnames in os.walk(args.negatives_in):
            for f in fnames:
                if not f.lower().endswith(IMG_EXTS):
                    continue
                host, _ = parse_names(f)
                if host in (None, "tomato"):
                    continue
                if hosts[host] >= max_per_host:
                    continue
                hosts[host] += 1
                shutil.copy2(os.path.join(dirpath, f),
                             os.path.join(out_neg, f))
        print(f"  negatives: {sum(hosts.values())} images over {len(hosts)} hosts")

    with open(os.path.join(args.out, "data.yaml"), "w") as fh:
        yaml.safe_dump({"names": classes, "nc": len(classes),
                        "train": "images/train", "val": "images/val",
                        "test": "images/test",
                        "negatives": "images/negatives",
                        "not_tomato_label": neg_cfg.get("label", "not_tomato")},
                       fh, sort_keys=False)
    if skipped:
        print("skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.most_common()))
    print(f"wrote corpus -> {args.out}")


if __name__ == "__main__":
    main()
