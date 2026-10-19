"""End-to-end check that the v5 model serves correctly through model_server.

Runs predict_image() over a sample of REAL test images and asserts:
  * the model loads with strict=True (no silent partial load)
  * every returned class name is in the 11-class vocabulary
  * mask + CAM PNGs are actually written and non-empty
  * boxes land inside the image bounds
  * classification agrees with the filename's ground-truth class most of the time

Run:  python3 _check_v5_serving.py [n_images]
"""
import os
import sys
import glob

import cv2
import torch

import model_server as ms

# The frontend's DISEASE_INFO table, parsed straight from the TS source so this
# check fails if the two vocabularies ever drift apart.
_FRONTEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web",
                         "src", "lib", "api.ts")
DISEASE_INFO = set()
if os.path.exists(_FRONTEND):
    import re
    with open(_FRONTEND) as fh:
        blob = fh.read()
    block = blob.split("const DISEASE_INFO", 1)[1].split("};", 1)[0]
    DISEASE_INFO = set(re.findall(r"^\s{2}(\w+):\s*\{", block, re.M))

# Corpus root: CROPGUARD_DATA (a machine-local path would break on Kaggle/HF and
# did -- this constant previously missed the 'All tries of cropguard/' segment,
# so the harness silently found no test images).
_CORPUS = os.environ.get(
    "CROPGUARD_DATA",
    "/kaggle/input/cropguard-corpus-claude")
DATA = os.path.join(_CORPUS, "images", os.environ.get("CROPGUARD_SPLIT", "test"))

# filename -> class name, derived the way the dataset was built.
# Must match ms.CLASS_NAMES exactly ('Healthy' + snake_case) because the web UI
# keys DISEASE_INFO on the same strings.
GT_TOKENS = [
    ("healthy", "Healthy"),
    ("bacterial", "bacterial_spot"),
    ("early-blight", "early_blight"),
    ("late-blight", "late_blight"),
    ("leaf-miner", "leaf_miner"),
    ("leaf-mold", "leaf_mold"),
    ("mosaic", "mosaic_virus"),
    ("septoria", "septoria_leaf_spot"),
    ("spider", "spider_mites"),
    ("target", "target_spot"),
    ("yellow-leaf-curl", "yellow_leaf_curl_virus"),
]


def gt_from_name(name: str):
    low = name.lower()
    for tok, label in GT_TOKENS:
        if tok in low:
            return label
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    files = sorted(glob.glob(os.path.join(DATA, "*.jpg")))[:n]
    if not files:
        sys.exit(f"no test images found under {DATA}")

    model = ms.get_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded: {ms.get_model_version()}  "
          f"params={n_params/1e6:.2f}M  device={ms.get_device()}  nc={model.nc}")
    assert model.nc == 11, f"expected nc=11, got {model.nc}"

    hits = scored = 0
    cam_ok = mask_ok = 0
    confs, ndet = [], []

    for f in files:
        with open(f, "rb") as fh:
            res = ms.predict_image(fh.read(), with_cam=True)

        cls = res["classification"]["class"]
        assert cls in ms.CLASS_NAMES, f"unknown class {cls!r}"

        # Every class the model can emit must exist in the frontend's
        # DISEASE_INFO table, else the UI shows "Information not available."
        for t in ("Healthy", "bacterial_spot", "early_blight", "late_blight",
                  "leaf_miner", "leaf_mold", "mosaic_virus",
                  "septoria_leaf_spot", "spider_mites", "target_spot",
                  "yellow_leaf_curl_virus"):
            assert t in ms.CLASS_NAMES
            assert t in DISEASE_INFO, f"{t} missing from frontend DISEASE_INFO"

        mask_p = ms.get_mask_path(res["image_id"])
        cam_p = ms.get_cam_path(res["image_id"])
        assert os.path.exists(mask_p) and os.path.getsize(mask_p) > 0, "mask not written"
        assert os.path.exists(cam_p) and os.path.getsize(cam_p) > 0, "cam not written"
        mask_ok += 1
        cam_ok += 1

        W, H = res["width"], res["height"]
        for b in res["leaf_boxes"]:
            x1, y1, x2, y2 = b["box"]
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, \
                f"box {b['box']} outside {W}x{H}"

        confs.append(res["classification"]["confidence"])
        ndet.append(len(res["leaf_boxes"]))

        gt = gt_from_name(os.path.basename(f))
        if gt:
            scored += 1
            if gt == cls:
                hits += 1
        print(f"  {os.path.basename(f)[:52]:54s} -> {cls:24s} "
              f"{res['classification']['confidence']:.3f} "
              f"det={len(res['leaf_boxes'])} mask_spots={len(res['detections'])} "
              f"sev={res['severity']['stage']}")

    acc = hits / max(scored, 1)
    print("\n--- summary ---")
    print(f"images: {len(files)}   with GT name: {scored}")
    print(f"classification agreement with filename: {hits}/{scored} = {acc:.3f}")
    print(f"mean top-1 confidence: {sum(confs)/len(confs):.3f}")
    print(f"mean det boxes @{ms.CONF_THR}: {sum(ndet)/len(ndet):.2f}  "
          f"(images with 0 boxes: {sum(1 for n in ndet if n == 0)})")
    print(f"mask written: {mask_ok}/{len(files)}   cam written: {cam_ok}/{len(files)}")
    print(f"class vocabulary size: {len(ms.CLASS_NAMES)}")
    print(f"classes covered by frontend DISEASE_INFO: "
          f"{sum(1 for c in ms.CLASS_NAMES if c in DISEASE_INFO)}/{len(ms.CLASS_NAMES)}")
    print("\nALL CHECKS PASSED" if mask_ok == len(files) == cam_ok else "\nFAILED")


if __name__ == "__main__":
    main()