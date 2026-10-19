import json
import os
import sys
import urllib.request

IMG = sys.argv[1] if len(sys.argv) > 1 else "/home/logiclocus/Downloads/tomato-late-blight.webp"
BOUNDARY = "----cbverify1234"

with open(IMG, "rb") as f:
    img = f.read()

body = b""
for name, val in (("image", ("filename", "photo.webp", "image/webp", img)),
                  ("conf", "0.25"),
                  ("with_cam", "true")):
    body += b"--" + BOUNDARY.encode() + b"\r\n"
    if isinstance(val, str):
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode()
    else:
        _, fname, ctype, data = val
        body += (f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
body += b"--" + BOUNDARY.encode() + b"--\r\n"

req = urllib.request.Request("http://localhost:8000/v1/predict", data=body, method="POST",
                             headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
try:
    r = json.load(urllib.request.urlopen(req, timeout=120))
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:300])
    sys.exit(1)

print("model:", r["model_version"])
print("class:", r["classification"]["class"], f"({r['classification']['confidence']:.2f})")
print("top3:", [(t["class"], round(t["confidence"], 2)) for t in r["classification"]["top3"]])
print("severity:", r["severity"]["stage"], "| reliable:", r["severity"].get("reliable"),
      "| leaf region:", round(r["segmentation"]["leaf_fraction"], 3))
print("detections:", len(r["detections"]))
for d in r["detections"][:4]:
    print("  ", d["class"], round(d["score"], 2), [round(v) for v in d["box"]])

cam_url = "http://localhost:8000" + r["cam_png_url"]
cam_path = "/tmp/cam_check.png"
urllib.request.urlretrieve(cam_url, cam_path)
sz = os.path.getsize(cam_path)
print("cam file bytes:", sz, "-> REAL CAM" if sz > 1000 else "-> STILL DUMMY")

mask_url = "http://localhost:8000" + r["segmentation"]["mask_png_url"]
mask_path = "/tmp/mask_check.png"
urllib.request.urlretrieve(mask_url, mask_path)
msz = os.path.getsize(mask_path)
print("mask file bytes:", msz, "-> REAL MASK" if msz > 800 else "-> EMPTY/WEAK")
