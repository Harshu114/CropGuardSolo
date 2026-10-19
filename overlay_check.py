import os
import glob
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
DATA_ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
OUTPUT_DIR = "diagnostic_overlays"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Split to check
split = "train"
img_dir = os.path.join(DATA_ROOT, "images", split)
les_dir = os.path.join(DATA_ROOT, "masks_lesion", split)
lf_dir = os.path.join(DATA_ROOT, "masks_leaf", split)

# Get random 50 images
all_imgs = glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")) + glob.glob(os.path.join(img_dir, "*.webp"))
if len(all_imgs) == 0:
    print("No images found!")
    exit(1)

sample_imgs = random.sample(all_imgs, min(50, len(all_imgs)))

print(f"Checking {len(sample_imgs)} images from {split} split...")

correct_count = 0

for i, img_path in enumerate(sample_imgs):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    les_path = os.path.join(les_dir, stem + ".png")
    lf_path = os.path.join(lf_dir, stem + ".png")

    if not os.path.exists(les_path) or not os.path.exists(lf_path):
        print(f"[{i}] Missing masks for {stem}")
        continue

    # Load
    img = cv2.imread(img_path)
    if img is None:
        print(f"[{i}] Could not read image {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lesion = cv2.imread(les_path, 0)
    leaf = cv2.imread(lf_path, 0)

    if lesion is None or leaf is None:
        print(f"[{i}] Could not read masks for {stem}")
        continue

    # Resize to match img if necessary
    h, w = img.shape[:2]
    lesion = cv2.resize(lesion, (w, h), interpolation=cv2.INTER_NEAREST)
    leaf = cv2.resize(leaf, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binary masks
    lesion_bin = (lesion > 127).astype(np.uint8)
    leaf_bin = (leaf > 127).astype(np.uint8)

    # Logic check: Lesion must be a subset of Leaf
    # Intersection should equal lesion
    intersection = cv2.bitwise_and(lesion_bin, leaf_bin)
    is_subset = np.array_equal(intersection, lesion_bin)

    # Compute severity
    lesion_area = np.sum(lesion_bin)
    leaf_area = np.sum(leaf_bin)
    severity = lesion_area / max(leaf_area, 1.0)

    # Visual Overlay
    # Create a colored overlay
    overlay = img.copy()
    # Red for lesion
    overlay[lesion_bin == 1] = [255, 0, 0]
    # Green for leaf (where not lesion)
    overlay[(leaf_bin == 1) & (lesion_bin == 0)] = [0, 255, 0]

    # Blend
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Add text
    pil_img = Image.fromarray(blended)
    draw = ImageDraw.Draw(pil_img)
    text = f"Sev: {severity:.3f} | Subset: {is_subset}"
    draw.text((10, 10), text, fill=(255, 255, 255))

    # Save
    pil_img.save(os.path.join(OUTPUT_DIR, f"{stem}_overlay.png"))

    if is_subset and leaf_area > 0:
        correct_count += 1

print(f"Done. Correct subset fraction: {correct_count/len(sample_imgs):.2%}")
print(f"Overlays saved to {OUTPUT_DIR}/")
