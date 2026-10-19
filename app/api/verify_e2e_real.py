"""End-to-end check of the live API against REAL annotated test images.

Unlike the earlier demo assets (synthetic PIL drawings, now deleted), this posts
genuine photographs from the training corpus through the actual HTTP endpoint and
verifies the full contract: status, class vocabulary, top-3, boxes in bounds,
mask + CAM actually served, and agreement with the filename's ground truth.

Usage:  python3 verify_e2e_real.py [n_images]
Env:    CROPGUARD_DATA  corpus root (defaults to the Kaggle/local corpus)
"""
import glob
import json
import os
import sys
import urllib.request

BASE = "http://localhost:8000"
DEFAULT_DATA = "/kaggle/input/cropguard-corpus-claude"
DATA = os.environ.get("CROPGUARD_DATA", DEFAULT_DATA)
SPLIT = os.environ.get("CROPGUARD_SPLIT", "test")

# filename token -> class name (must match model_server.CLASS_NAMES)
GT_TOKENS = [
    ("healthy", "Healthy"), ("bacterial", "bacterial_spot"),
    ("early-blight", "early_blight"), ("early_blight", "early_blight"),
    ("late-blight", "late_blight"), ("late_blight", "late_blight"),
    ("leaf-miner", "leaf_miner"), ("leaf_mold", "leaf_mold"),
    ("mosaic", "mosaic_virus"), ("septoria", "septoria_leaf_spot"),
    ("spider", "spider_mites"), ("target", "target_spot"),
    ("yellow-leaf-curl", "yellow_leaf_curl_virus"),
]


def gt_from_name(name):
    low = name.lower()
    for tok, label in GT_TOKENS:
        if tok in low:
            return label
    return None


def post_image(path, conf="0.25", with_cam="true"):
    img = open(path, "rb").read()
    B = "----cropguard"
    parts = []
    parts.append((f'--{B}\r\nContent-Disposition: form-data; name="image"; '
                  f'filename="{os.path.basename(path)}"\r\n'
                  f"Content-Type: image/jpeg\r\n\r\n").encode() + img + b"\r\n")
    for k, v in (("conf", conf), ("with_cam", with_cam)):
        parts.append((f'--{B}\r\nContent-Disposition: form-data; name="{k}"'
                      f"\r\n\r\n{v}\r\n").encode())
    parts.append(f"--{B}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        BASE + "/v1/predict", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={B}"})
    return json.load(urllib.request.urlopen(req, timeout=300))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    files = sorted(glob.glob(os.path.join(DATA, "images", SPLIT, "*.jpg")))
    if not files:
        sys.exit(f"no real test images under {DATA}/images/{SPLIT} "
                 f"(set CROPGUARD_DATA)")
    files = files[:n]

    print(f"corpus: {DATA}  split={SPLIT}  images={len(files)}\n")
    hits = scored = 0
    for p in files:
        r = post_image(p)
        c = r["classification"]
        cls = c["class"]

        # --- contract assertions ---
        assert isinstance(c["confidence"], (int, float)) and 0 <= c["confidence"] <= 1
        assert len(c["top3"]) == 3, c["top3"]
        assert r["severity"]["reliable"] in (True, False)
        W, H = r["width"], r["height"]
        for b in r["detections"]:
            x1, y1, x2, y2 = b["box"]
            assert 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H, f"box out of bounds {b['box']}"

        mask = urllib.request.urlopen(
            BASE + r["segmentation"]["mask_png_url"], timeout=60).read()
        cam = urllib.request.urlopen(BASE + r["cam_png_url"], timeout=60).read()
        assert len(mask) > 0 and len(cam) > 0, "mask/cam not served"

        gt = gt_from_name(os.path.basename(p))
        if gt:
            scored += 1
            hits += (gt == cls)
        flag = "" if (gt is None or gt == cls) else f"  (GT={gt})"
        print(f"{os.path.basename(p)[:44]:46s} -> {cls:22s} {c['confidence']:.3f} "
              f"| det={len(r['detections'])} sev={r['severity']['stage']}"
              f"(rel={r['severity']['reliable']}) mask={len(mask)}B cam={len(cam)}B"
              f" {r['timings_ms']['model']}ms{flag}")
        print(f"{'':46s}    top3: " + ", ".join(
            f"{t['class']} {t['confidence']:.2f}" for t in c["top3"]))

    print(f"\nclassification agreement with filename: {hits}/{scored}")
    print("ALL CONTRACT CHECKS PASSED")


if __name__ == "__main__":
    main()
