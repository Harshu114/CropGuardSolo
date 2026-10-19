import glob, os, json, urllib.request, uuid
import numpy as np, cv2

# (a) what the TRAINING masks actually contain
ms = sorted(glob.glob("/home/logiclocus/HarshalPrivate/Projects/collegeProjects/All tries of cropguard/ClaudeCropGuard/datasetclaude/masks/train/*.png"))[:40]
uniq = {}
frac126 = []
for p in ms:
    m = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if m is None: continue
    for v, c in zip(*np.unique(m, return_counts=True)):
        uniq[int(v)] = uniq.get(int(v), 0) + int(c)
    frac126.append(float((m == 128).mean()))
print("TRAIN MASK unique-value histogram (40 files):", uniq)
print("mean fraction of pixels == 128 :", round(float(np.mean(frac126)), 6) if frac126 else None)
print("mean fraction of pixels == 255 :", None)

# (b) what the SERVED mask looks like for a real photo
img = os.path.expanduser("~/Downloads/tomato-late-blight.webp")
if not os.path.exists(img):
    img = os.path.expanduser("~/Downloads/lateblight1.jpg")
boundary = uuid.uuid4().hex
body = b""
for name, val in (("conf", "0.15"), ("with_cam", "false")):
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n").encode()
fn = os.path.basename(img)
body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{fn}\"\r\n"
         f"Content-Type: image/jpeg\r\n\r\n").encode() + open(img, "rb").read() + b"\r\n"
body += f"--{boundary}--\r\n".encode()
req = urllib.request.Request("http://localhost:8000/v1/predict", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
r = json.loads(urllib.request.urlopen(req, timeout=60).read())
print("\nserved:", fn, "->", r["classification"]["class"], r["classification"]["confidence"],
      "| severity", r["severity"], "| leaf_fraction", r["segmentation"]["leaf_fraction"],
      "| detections", len(r["detections"]))
os.makedirs("_diag_mask_out", exist_ok=True)
m = urllib.request.urlopen(f"http://localhost:8000/v1/masks/{r['image_id']}.png", timeout=30).read()
open(f"_diag_mask_out/{fn}.mask.png", "wb").write(m)
seg = cv2.imread(f"_diag_mask_out/{fn}.mask.png", cv2.IMREAD_GRAYSCALE)
n, _, stats, _ = cv2.connectedComponentsWithStats((seg > 127).astype(np.uint8), 8)
areas = sorted([int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)], reverse=True)
print("SERVED mask: foreground fraction", round(float((seg > 127).mean()), 4),
      "| components >=12px:", len([a for a in areas if a >= 12]), "| top areas:", areas[:6])
print("first detection box_norm:", r["detections"][0]["box_norm"] if r["detections"] else None)
