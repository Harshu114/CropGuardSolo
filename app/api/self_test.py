import json
import sys
import urllib.request
import numpy as np
import cv2

BOUNDARY = "----st1234"

def predict(path):
    with open(path, "rb") as f:
        img = f.read()
    body = b""
    for name, val in (("image", ("f.jpg", "f.jpg", "image/jpeg", img)),
                      ("conf", "0.25"), ("with_cam", "false")):
        body += b"--" + BOUNDARY.encode() + b"\r\n"
        if isinstance(val, str):
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode()
        else:
            body += (f'Content-Disposition: form-data; name="{name}"; filename="{val[1]}"\r\n'
                     f"Content-Type: {val[2]}\r\n\r\n").encode() + val[3] + b"\r\n"
    body += b"--" + BOUNDARY.encode() + b"--\r\n"
    req = urllib.request.Request("http://localhost:8000/v1/predict", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def lesion_overlap(photo_path, mask_url):
    """Fraction of mask pixels that sit on discolored (non-green) pixels."""
    img = cv2.imread(photo_path)
    h, w = img.shape[:2]
    mask = cv2.imread("/tmp/st_mask.png", 0)
    if mask is None:
        return None
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # healthy green leaf: hue 35-85, decent saturation
    green = ((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 60)).astype(np.uint8)
    m = mask > 127
    tot = int(m.sum())
    if tot < 50:
        return None  # mask essentially empty
    on_green = int((m & (green > 0)).sum())
    return 1.0 - on_green / tot  # fraction of mask on NON-green = lesion-like

photos = [
    ("spotdisease.jpg", "bacterial_spot (user-labeled)"),
    ("lateblight1.jpg", "late_blight (user-labeled)"),
    ("tomato-late-blight.webp", "late_blight (filename)"),
    ("tomato-early-blight.webp", "early_blight (filename)"),
    ("tomato-septoria-leaf-spot.webp", "septoria (filename)"),
    ("leaf1.jpg", "unknown"),
]
# Photo dir: env override, else a couple of common locations. Missing photos
# are SKIPPED with a clear note instead of printing six identical API ERROR
# lines (the previous hardcoded /home/<user>/Downloads path made this script
# useless on any other machine).
import os as _os
_D = _os.environ.get("CROPGUARD_TEST_PHOTOS")
_cands = ([_D] if _D else []) + [
    _os.path.expanduser("~/Downloads"),
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_user_images"),
]
D = next((c + _os.sep for c in _cands if c and _os.path.isdir(c)), None)
_missing = [] if D else [fn for fn, _ in photos]
print(f"photo dir: {D or 'NOT FOUND (set CROPGUARD_TEST_PHOTOS)'}")
print(f"{'photo':36s} {'truth':28s} {'predicted (conf)':24s} mask-on-lesion")
print("-" * 100)
for fn, truth in photos:
    p = D + fn
    if not _os.path.exists(p):
        print(f"{fn:36s} {truth:28s} SKIPPED (photo not found)")
        continue
    try:
        r = predict(p)
    except Exception as e:
        print(f"{fn:36s} {truth:28s} API ERROR: {str(e)[:40]}")
        continue
    cls = r["classification"]
    top3 = ", ".join(f"{t['class']} {int(t['confidence']*100)}%" for t in cls["top3"][:2])
    # fetch mask
    urllib.request.urlretrieve("http://localhost:8000" + r["segmentation"]["mask_png_url"], "/tmp/st_mask.png")
    ov = lesion_overlap(p, r["segmentation"]["mask_png_url"])
    ov_s = f"{ov*100:.0f}%" if ov is not None else ("tiny/empty" if r["segmentation"]["leaf_fraction"] < 0.005 else "n/a")
    print(f"{fn:36s} {truth:28s} {cls['class'] + ' (' + str(int(cls['confidence']*100)) + '%)':24s} {ov_s}")
    print(f"{'':65s} top3: {top3} | leaf_region={r['segmentation']['leaf_fraction']:.3f} | boxes={len(r['detections'])}")
