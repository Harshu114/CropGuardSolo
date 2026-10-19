"""Build the Flutter app's sample assets from REAL photographs and REAL model output.

The previous assets in mobile/assets/ were drawn programmatically with PIL
(`create_demo_images.py` / `create_masks.py`), so every "diagnosis" the app showed
was fiction: a hand-painted leaf with hand-painted rectangles standing in for a
lesion mask. This script replaces them with:

  * photos taken from the real annotated corpus, and
  * masks / heatmaps produced by actually running cropguard_model_v5 on those
    photos.

Ground truth comes from the YOLO label files (labels/<split>/*.txt), NOT from
filenames -- filenames are unreliable here (the `healthy` class is stored as
`GHLB2-Leaf-*`, and `Target` never appears at all). Only images that contain
exactly ONE class are used, because the "class" of a multi-class collage image is
not well defined.

It also prints the measured confidence / detected-region / affected-area numbers
so `mobile/lib/main.dart` (which previously hardcoded invented values like
confidence 0.912) can use real ones.

Usage:
    CROPGUARD_DATA=/path/to/corpus python3 mobile/generate_real_assets.py
"""
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "app", "api"))
import model_server as ms  # noqa: E402

DATA = os.environ.get("CROPGUARD_DATA", "/kaggle/input/cropguard-corpus-claude")
SPLIT = os.environ.get("CROPGUARD_SPLIT", "test")
MAX_PER_CLASS = int(os.environ.get("MAX_CANDIDATES", "40"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_IMG = os.path.join(HERE, "assets", "images")
OUT_MASK = os.path.join(HERE, "assets", "masks")
OUT_HEAT = os.path.join(HERE, "assets", "heatmaps")
SIZE = 400

# asset stems to generate (matched to model_server.CLASS_NAMES case-insensitively)
WANTED = ["late_blight", "early_blight", "bacterial_spot", "healthy"]


def model_class_name(dataset_name):
    """Map a data.yaml class name to the exact string model_server emits.

    These differ in case (`healthy` vs `Healthy`), and comparing the raw
    data.yaml name against the prediction silently rejects every match.
    """
    for c in ms.CLASS_NAMES:
        if c.lower() == dataset_name.lower():
            return c
    return None


def single_class_index(root, split):
    """stem -> class index, for images containing exactly one class."""
    names = yaml.safe_load(open(os.path.join(root, "data.yaml")))["names"]
    out = {}
    for lp in glob.glob(os.path.join(root, "labels", split, "*.txt")):
        idxs = set()
        for line in open(lp):
            parts = line.split()
            if parts:
                idxs.add(int(float(parts[0])))
        if len(idxs) == 1:
            out[os.path.splitext(os.path.basename(lp))[0]] = idxs.pop()
    return out, names


def main():
    for d in (OUT_IMG, OUT_MASK, OUT_HEAT):
        os.makedirs(d, exist_ok=True)

    stem2cls, names = single_class_index(DATA, SPLIT)
    by_cls = defaultdict(list)
    for stem, idx in stem2cls.items():
        by_cls[idx].append(stem)
    print(f"corpus {DATA}/{SPLIT}: {len(stem2cls)} single-class images")
    print("  " + ", ".join(f"{names[i]}:{len(by_cls[i])}"
                           for i in range(len(names)) if by_cls[i]))

    ms.load_model(os.path.abspath(os.path.join(HERE, "..", "app", "api",
                                              "cropguard_model_v5.pt")))
    print(f"\nmodel: {ms.get_model_version()}\n")

    imgdir = os.path.join(DATA, "images", SPLIT)
    summary = []
    for stem in WANTED:
        if stem not in names:
            print(f"  !! {stem} not in data.yaml"); continue
        want = model_class_name(stem)
        if want is None:
            print(f"  !! {stem}: no matching entry in model_server.CLASS_NAMES")
            continue
        idx = names.index(stem)
        best = None
        n_scored = 0
        for cand in sorted(by_cls[idx])[:MAX_PER_CLASS]:
            p = os.path.join(imgdir, cand + ".jpg")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                r = ms.predict_image(f.read(), with_cam=True)
            n_scored += 1
            if r["classification"]["class"] != want:
                continue
            conf = r["classification"]["confidence"]
            if best is None or conf > best[2]:
                best = (p, r, conf)
        if best is None:
            print(f"  !! {stem}: model predicted {want} for none of "
                  f"{n_scored} scored candidates")
            continue

        path, r, conf = best
        with open(path, "rb") as f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(OUT_IMG, f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

        m = cv2.resize(cv2.imread(ms.get_mask_path(r["image_id"]), cv2.IMREAD_GRAYSCALE),
                       (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        c = cv2.resize(cv2.imread(ms.get_cam_path(r["image_id"]), cv2.IMREAD_COLOR),
                       (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(OUT_MASK, f"{stem}_mask.png"), m)
        cv2.imwrite(os.path.join(OUT_HEAT, f"{stem}_heatmap.png"), c)

        summary.append((stem, conf, len(r["detections"]),
                        r["segmentation"]["leaf_fraction"],
                        r["severity"]["stage"]))
        print(f"  {stem:16s} <- {os.path.basename(path)[:44]:46s} "
              f"conf={conf:.3f} det={len(r['detections'])} "
              f"leaf={r['segmentation']['leaf_fraction']:.3f}")

    print("\n--- measured values for mobile/lib/main.dart ---")
    for stem, conf, ndet, leaf, sev in summary:
        print(f"  {stem:16s} confidence={conf:.3f}  detectedRegions={ndet}  "
              f"affectedArea={leaf * 100:.1f}  severity={sev}")


if __name__ == "__main__":
    main()
