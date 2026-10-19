import os, glob
import cv2
import numpy as np
from PIL import Image

ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
train_images = sorted(glob.glob(f"{ROOT}/images/train/*"))

below_1 = []
between_1_5 = []

for p in train_images:
    stem = os.path.splitext(os.path.basename(p))[0]
    lp = f"{ROOT}/masks_leaf/train/{stem}.png"
    if not os.path.exists(lp): continue
    mask = cv2.imread(lp, 0)
    frac = (mask > 127).sum() / mask.size
    if frac < 0.01:
        below_1.append((stem, frac, p, lp))
    elif frac < 0.05:
        between_1_5.append((stem, frac, p, lp))

print(f"Group < 1%: {len(below_1)} images")
print(f"Group 1-5%: {len(between_1_5)} images")

# Generate visual overlay contact sheet for all 39 images
# Each item will display: Original image + Leaf mask (green overlay) + Lesion mask (red overlay)
sheet_imgs = []
metadata = []

all_39 = [(stem, frac, p, lp, "<1%") for stem, frac, p, lp in below_1] + \
         [(stem, frac, p, lp, "1-5%") for stem, frac, p, lp in between_1_5]

for stem, frac, img_p, leaf_p, grp in all_39:
    img = cv2.imread(img_p)
    H, W = img.shape[:2]
    leaf = cv2.imread(leaf_p, 0)
    les_p = f"{ROOT}/masks_lesion/train/{stem}.png"
    les = cv2.imread(les_p, 0) if os.path.exists(les_p) else np.zeros((H, W), np.uint8)
    
    leaf_res = cv2.resize(leaf, (W, H), interpolation=cv2.INTER_NEAREST)
    les_res = cv2.resize(les, (W, H), interpolation=cv2.INTER_NEAREST)
    
    overlay = img.copy()
    overlay[leaf_res > 127] = [0, 255, 0] # green leaf
    overlay[les_res > 127] = [0, 0, 255]  # red lesion
    vis = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    
    # Check boundary ratio or heuristic:
    # A leaf mask failure in 1-5% is typically:
    # 1. GrabCut clipped away 95% of an obvious large leaf (severe false background)
    # 2. Or GrabCut captured a small detached piece of background
    # Let's measure properties:
    vis_thumb = cv2.resize(vis, (160, 160))
    # Put text
    cv2.putText(vis_thumb, f"{frac*100:.1f}%", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    sheet_imgs.append(vis_thumb)
    metadata.append((stem, frac, grp))

# Make a contact sheet: 8 columns x 5 rows
n_cols = 8
n_rows = 5
contact_sheet = np.zeros((n_rows * 160, n_cols * 160, 3), dtype=np.uint8)
for idx, thumb in enumerate(sheet_imgs):
    r = idx // n_cols
    c = idx % n_cols
    contact_sheet[r*160:(r+1)*160, c*160:(c+1)*160] = thumb

cv2.imwrite("exclusion_39_contact_sheet.png", contact_sheet)
print("Saved exclusion_39_contact_sheet.png")

