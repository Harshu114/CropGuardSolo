"""Convert PlantSeg into the CropGuard corpus layout, deriving all four heads.

Why this dataset: the current corpus's masks are filled *leaf bounding boxes*
(binary {0,255}; the "128 = lesion" value never occurs in any of the 32,897
masks), so the segmentation head learns a leaf region and severity (derived from
leaf area) collapses to "Severe" for almost everything. PlantSeg ships
expert-annotated polygons of *diseased tissue*, in the wild, with tomato among its
34 host plants.

From PlantSeg's COCO annotations this derives the labels each head needs:

  classification  disease name (from the image's folder/name; PlantSeg's own
                  `categories` array is empty, so the name is the only label)
  detection       the COCO `bbox` of each lesion -> YOLO label rows
  segmentation    the COCO polygon -> a rendered binary lesion mask
  severity        polygon area / image area -> a real, area-based severity target

Get the data first (3 GB, Zenodo, md5 c321381894575e5dca83686d125fe2cd):
    curl -L -o plantseg.zip https://zenodo.org/records/13762907/files/plantseg.zip
    unzip plantseg.zip -d plantseg
    git clone https://github.com/tqwei05/PlantSeg   # for coco_annotations/

Then:
    python3 data_pipeline/scripts/prepare_plantseg.py \
        --root plantseg --annos PlantSeg/coco_annotations \
        --out ./plantseg_tomato --plant tomato

Self-test the label derivation without the 3 GB of images (uses the annotation
JSONs only):
    python3 data_pipeline/scripts/prepare_plantseg.py --annos PlantSeg/coco_annotations --self-test
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

SPLITS = ("train", "val", "test")


def parse_names(file_name, plant):
    """'tomato_bacterial_leaf_spot_12.jpg' -> ('tomato', 'bacterial_leaf_spot')."""
    stem = os.path.splitext(os.path.basename(file_name))[0]
    if "_" not in stem:
        return None, None
    host, rest = stem.split("_", 1)
    disease = re.sub(r"(_google)?_\d+$", "", rest)          # strip trailing index
    disease = re.sub(r"[^a-z0-9]+", "_", disease.lower()).strip("_")
    if not disease:
        return host, None
    return host, disease


def load_split(annos_dir, split, plant):
    path = os.path.join(annos_dir, f"annotation_{split}.json")
    if not os.path.exists(path):
        # Do not fail silently: a missing file previously produced "0 images"
        # with no explanation, which reads as "this plant has no data".
        print(f"  [warn] missing {path} -- skipping split {split!r}", file=sys.stderr)
        return [], Counter()
    try:
        data = json.load(open(path))
    except json.JSONDecodeError as e:
        # A truncated download (interrupted curl) otherwise aborts the whole run
        # with an opaque traceback; name the likely cause instead.
        print(f"  [warn] {path} is not valid JSON ({e}). "
              f"size={os.path.getsize(path)}B -- truncated download? skipping split "
              f"{split!r}", file=sys.stderr)
        return [], Counter()
    by_image = defaultdict(list)
    for a in data.get("annotations", []):
        by_image[a["image_id"]].append(a)

    rows = []
    classes = Counter()
    for im in data.get("images", []):
        host, disease = parse_names(im["file_name"], plant)
        if plant and host != plant:
            continue
        if disease is None:
            continue
        anns = by_image.get(im["id"], [])
        if not anns:
            continue
        classes[disease] += 1
        rows.append({
            "file_name": im["file_name"], "w": im["width"], "h": im["height"],
            "disease": disease, "anns": anns,
        })
    return rows, classes


def poly_to_mask(anns, w, h):
    """Render COCO polygons/sub-RLE into a binary uint8 lesion mask (FILLED).

    The v7 corpus was built when this function only walked the polygon EDGES
    (1px outline strokes, no interior), which made every lesion target a lacy
    scribble -- unlearnable at model resolution and meaningless for IoU/severity
    (see RUNLOG "v8 handover audit").  The interior is now filled with an
    even-odd scanline pass; the boundary is still drawn for crisp edges.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        seg = a.get("segmentation")
        if isinstance(seg, list):
            for poly in seg:
                if len(poly) < 6:
                    continue
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    n = max(int(max(abs(x2 - x1), abs(y2 - y1))), 1)
                    xs = np.linspace(x1, x2, n, dtype=np.float32)
                    ys = np.linspace(y1, y2, n, dtype=np.float32)
                    for x, y in zip(xs, ys):
                        xi = int(min(max(x, 0), w - 1))
                        yi = int(min(max(y, 0), h - 1))
                        mask[yi, xi] = 255
                # interior: even-odd scanline fill (pure numpy, no cv2 needed)
                ymin = max(int(np.floor(pts[:, 1].min())), 0)
                ymax = min(int(np.ceil(pts[:, 1].max())), h - 1)
                for yy in range(ymin, ymax + 1):
                    yc = yy + 0.5
                    xs_cross = []
                    for i in range(len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[(i + 1) % len(pts)]
                        if (y1 <= yc < y2) or (y2 <= yc < y1):
                            t = (yc - y1) / (y2 - y1)
                            xs_cross.append(x1 + t * (x2 - x1))
                    xs_cross.sort()
                    for k in range(0, len(xs_cross) - 1, 2):
                        xa = int(min(max(np.ceil(xs_cross[k]), 0), w - 1))
                        xb = int(min(max(np.floor(xs_cross[k + 1]), 0), w - 1))
                        if xb >= xa:
                            mask[yy, xa:xb + 1] = 255
        # (COCO RLE would go here; PlantSeg ships polygons.)
    return mask


def bbox_to_yolo(a, w, h):
    """COCO bbox [x, y, bw, bh] -> normalised (cx, cy, bw, bh) in [0,1]."""
    x, y, bw, bh = a["bbox"]
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    return [min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
            min(max(bw / w, 0.0), 1.0), min(max(bh / h, 0.0), 1.0)]


def self_test(annos_dir, plant):
    """Exercise label derivation on the annotation JSONs (no images needed)."""
    print(f"self-test: annos={annos_dir} plant={plant}\n")
    all_classes, grand = Counter(), 0
    for split in SPLITS:
        rows, classes = load_split(annos_dir, split, plant)
        grand += len(rows)
        all_classes.update(classes)
        boxes = sum(len(r["anns"]) for r in rows)
        print(f"  {split:5s}: {len(rows):5d} {plant} images, {boxes:6d} lesion boxes, "
              f"{len(classes):2d} diseases")
        if rows:
            r = rows[0]
            r["anns"][0].setdefault("bbox", [0, 0, 1, 1])
            m = poly_to_mask(r["anns"], min(r["w"], 256), min(r["h"], 256))
            cov = (m > 0).mean()
            y = bbox_to_yolo(r["anns"][0], r["w"], r["h"])
            print(f"         e.g. {r['file_name'][:44]:46s} disease={r['disease']}")
            print(f"              mask coverage={cov:.3%}  yolo row="
                  f"{[round(v, 4) for v in y]}")
    print(f"\n  {plant} totals: {grand} images, {len(all_classes)} diseases")
    for k, v in all_classes.most_common(12):
        print(f"     {v:>5}  {k}")
    return grand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="extracted PlantSeg dir (contains images/)")
    ap.add_argument("--annos", required=True, help="COCO annotations dir")
    ap.add_argument("--out", default="./plantseg_corpus")
    ap.add_argument("--plant", default="tomato")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.root:
        self_test(args.annos, args.plant)
        return

    # Collect classes first so indices are stable across splits.
    per_split, classes = {}, {}
    for split in SPLITS:
        rows, c = load_split(args.annos, split, args.plant)
        per_split[split] = rows
        classes[split] = c
    names = sorted(set().union(*[set(c) for c in classes.values()]))
    print(f"{len(names)} {args.plant} diseases across splits")
    if not names:
        sys.exit("no images matched -- check --plant and the annotation files")

    import cv2  # imported late so --self-test works without opencv
    for split in SPLITS:
        for sub in ("images", "labels", "masks"):
            os.makedirs(os.path.join(args.out, sub, split), exist_ok=True)
        sev = []
        n_mask = 0
        for r in per_split[split]:
            src = None
            for cand in glob.glob(os.path.join(args.root, "**", r["file_name"]),
                                  recursive=True):
                src = cand
                break
            if src is None:
                continue
            stem = os.path.splitext(os.path.basename(r["file_name"]))[0]
            import shutil
            shutil.copy2(src, os.path.join(args.out, "images", split, stem + ".jpg"))

            idx = names.index(r["disease"])
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
                for a in r["anns"]:
                    if "bbox" not in a:
                        continue
                    cx, cy, bw, bh = bbox_to_yolo(a, r["w"], r["h"])
                    f.write(f"{idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            m = poly_to_mask(r["anns"], r["w"], r["h"])
            cv2.imwrite(os.path.join(args.out, "masks", split, stem + ".png"), m)
            n_mask += 1
            area = sum(a.get("area", 0.0) for a in r["anns"])
            sev.append((stem, min(area / float(r["w"] * r["h"]), 1.0)))

        with open(os.path.join(args.out, f"severity_{split}.csv"), "w") as f:
            f.write("image_id,lesion_area_ratio\n")
            for stem, ratio in sev:
                f.write(f"{stem},{ratio:.6f}\n")
        print(f"  {split}: {n_mask} images, {len(sev)} severity rows")

    import yaml
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        yaml.safe_dump({"names": names, "nc": len(names),
                        "train": "images/train", "val": "images/val",
                        "test": "images/test"}, f, sort_keys=False)
    print(f"\nwrote corpus -> {args.out}")


if __name__ == "__main__":
    main()
