#!/usr/bin/env python3
"""DIAGNOSTIC: isolate the effect of input aspect-ratio handling.

Training data is 100% square 320x320 (cropguard-corpus-claude), so the v5 model
has only ever seen 1:1 crops. Serving (model_server.preprocess_image) does a
plain cv2.resize(rgb,(320,320)) -- a hard aspect-ratio STRETCH.

This feeds each user image through the real serving code three ways:
  A stretch   : resize to 320x320 ignoring aspect  (what production does now)
  B centercrop: crop the largest centred square, then resize to 320
  C letterbox : resize longest side to 320, pad the rest with gray 114
                (gray-114 padding IS in-distribution: dataset._augment pads
                 with exactly (114,114,114) during scale jitter)

Because each variant is pre-rendered to an exact 320x320 square, the server's
own resize becomes a no-op and the ONLY variable is aspect handling.

Usage: python _diag_aspect.py
"""
import io
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import model_server as ms

HERE = Path(__file__).resolve().parent
CKPT = HERE / "cropguard_model_v5.pt"
DL = Path(os.path.expanduser("~/Downloads"))

# (filename, expected class or None if not a v5 class)
CASES = [
    ("bacterialspot.jpg", "bacterial_spot"),
    ("bacterialspot1.jpg", "bacterial_spot"),
    ("lateblight.jpg", "late_blight"),
    ("lateblight1.jpg", "late_blight"),
    ("lateblight2.jpg", "late_blight"),
    ("magnesiumdeficiency.jpg", None),
    ("magnesiumdeficiency1.jpg", None),
    ("nitrogendeficiency.jpg", None),
    ("nitrogendeficiency1.jpg", None),
    ("test1leaf.jpg", None),
    ("test2leaf.jpg", None),
    ("tomato-early-blight.webp", "early_blight"),
    ("tomato-late-blight.webp", "late_blight"),
    ("tomato-septoria-leaf-spot.webp", "septoria_leaf_spot"),
]

S = ms.IMG_SIZE


def to_bytes(img_rgb: np.ndarray) -> bytes:
    """Encode an already-square 320x320 RGB array as PNG bytes."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return buf.tobytes()


def variant_stretch(bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(bgr, (S, S), interpolation=cv2.INTER_LINEAR)


def variant_centercrop(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(bgr[y:y + s, x:x + s], (S, S), interpolation=cv2.INTER_LINEAR)


def variant_letterbox(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = S / max(h, w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    r = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((S, S, 3), 114, np.uint8)
    y, x = (S - nh) // 2, (S - nw) // 2
    canvas[y:y + nh, x:x + nw] = r
    return canvas


VARIANTS = [("stretch(A)", variant_stretch),
            ("centercrop(B)", variant_centercrop),
            ("letterbox(C)", variant_letterbox)]


def main() -> int:
    ms.load_model(str(CKPT))
    print(f"model={ms.get_model_version()}  IMG_SIZE={S}  TTA={ms.USE_TTA}\n")

    results = {}
    for name, truth in CASES:
        p = DL / name
        if not p.exists():
            print(f"{name}: MISSING")
            continue
        pil = Image.open(p).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        row = {}
        for vname, fn in VARIANTS:
            sq = cv2.cvtColor(fn(bgr), cv2.COLOR_BGR2RGB)
            res = ms.predict_image(to_bytes(sq), with_cam=False)
            row[vname] = (res["classification"]["class"],
                          round(res["classification"]["confidence"], 3))
        results[name] = {"truth": truth, "size": f"{w}x{h}", "variants": row}

    # report
    hdr = f"{'file':<32} {'src':<9} {'truth':<20} " + " ".join(f"{v:<26}" for v, _ in VARIANTS)
    print(hdr)
    print("-" * len(hdr))
    score = {v: [0, 0] for v, _ in VARIANTS}   # [correct, in_scope]
    for name, r in results.items():
        cells = []
        for v, _ in VARIANTS:
            cls, conf = r["variants"][v]
            mark = ""
            if r["truth"]:
                score[v][1] += 1
                if cls == r["truth"]:
                    score[v][0] += 1
                    mark = "OK"
                else:
                    mark = "X"
            cells.append(f"{cls:<22}{conf:>4} {mark:<2}")
        print(f"{name:<32} {r['size']:<9} {str(r['truth']):<20} " + " ".join(cells))

    print()
    for v, _ in VARIANTS:
        c, n = score[v]
        print(f"  {v:<14} in-scope correct: {c}/{n}")

    out = HERE / "_diag_aspect_result.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
