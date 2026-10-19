#!/usr/bin/env python3
"""Run the user's own downloaded test images through the LIVE v5 HTTP API.

Goes through the real serving interface (http://localhost:8000/v1/predict),
not a direct model import, so it exercises exactly what the website does.
Honours the API's 10-requests/minute rate limit by retrying on HTTP 429.

Usage: python _check_user_images.py [dir]
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:8000"
PREDICT = f"{API}/v1/predict"

# The 11 classes the v5 checkpoint was trained on. Anything outside this set
# is unanswerable by design -- the model must pick a nearest neighbour.
V5_CLASSES = {
    "healthy", "bacterial_spot", "early_blight", "late_blight", "leaf_miner",
    "leaf_mold", "mosaic_virus", "septoria_leaf_spot", "spider_mites",
    "target_spot", "yellow_leaf_curl_virus",
}

NAMES = [
    "bacterialspot.jpg", "bacterialspot1.jpg",
    "lateblight.jpg", "lateblight1.jpg", "lateblight2.jpg",
    "magnesiumdeficiency.jpg", "magnesiumdeficiency1.jpg",
    "nitrogendeficiency.jpg", "nitrogendeficiency1.jpg",
    "leaf1.jpg", "spotdisease.jpg", "test1leaf.jpg", "test2leaf.jpg",
    "tomato-early-blight.webp", "tomato-late-blight.webp",
    "tomato-septoria-leaf-spot.webp",
]


def post_image(path: Path, tries: int = 8):
    """POST one image; on 429 wait out the rate-limit window and retry."""
    data = path.read_bytes()
    boundary = "----cropguardcheck"
    ctype = "image/webp" if path.suffix.lower() == ".webp" else "image/jpeg"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        data, b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="with_cam"\r\n\r\n',
        b"false\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    for attempt in range(tries):
        req = urllib.request.Request(
            PREDICT, data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"      429 rate-limited, waiting 30s (attempt {attempt+1})...")
                time.sleep(30)
                continue
            return {"_error": f"HTTP {e.code}: {e.read()[:200]!r}"}
        except Exception as e:  # noqa: BLE001
            return {"_error": f"{type(e).__name__}: {e}"}
    return {"_error": "still 429 after retries"}


def truth_of(name: str) -> str:
    """Derive the obvious ground truth from the filename."""
    n = name.lower().replace("tomato-", "").replace(".webp", "").replace(".jpg", "")
    for key, cls in [
        ("bacterialspot", "bacterial_spot"), ("lateblight", "late_blight"),
        ("magnesiumdeficiency", "magnesium_deficiency"),
        ("nitrogendeficiency", "nitrogen_deficiency"),
        ("septoria", "septoria_leaf_spot"), ("early-blight", "early_blight"),
        ("earlyblight", "early_blight"),
    ]:
        if key in n:
            return cls
    return "?"


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads"))
    print(f"Testing against live API {API}  (dir={d})\n")
    hdr = f"{'file':<34} {'truth':<22} {'predicted':<24} {'conf':>5}  {'sev':<8} {'det':>3}  note"
    print(hdr)
    print("-" * len(hdr))

    rows, correct, n_in_scope = [], 0, 0
    for name in NAMES:
        p = d / name
        if not p.exists():
            print(f"{name:<34} MISSING")
            continue
        res = post_image(p)
        if "_error" in res:
            print(f"{name:<34} ERROR {res['_error']}")
            rows.append({"file": name, "error": res["_error"]})
            continue

        truth = truth_of(name)
        pred = res["classification"]["class"]
        conf = res["classification"]["confidence"]
        sev = res["severity"]["stage"]
        ndet = len(res["detections"])

        note = ""
        if truth != "?" and truth not in V5_CLASSES:
            note = f"TRUTH NOT A v5 CLASS -> nearest match forced"
        elif truth != "?":
            n_in_scope += 1
            if pred == truth:
                correct += 1
            else:
                note = "WRONG"

        print(f"{name:<34} {truth:<22} {pred:<24} {conf:>5.3f}  {sev:<8} {ndet:>3}  {note}")
        rows.append({"file": name, "truth": truth, "pred": pred, "conf": conf,
                     "sev": sev, "ndet": ndet, "note": note})

    print()
    print(f"In-scope correct: {correct}/{n_in_scope}")
    out = Path(__file__).with_name("_user_images_result.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
