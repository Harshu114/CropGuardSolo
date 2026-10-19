import os
import glob
import numpy as np
import cv2
import random
from PIL import Image, ImageDraw

DATA_ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"

def get_counts():
    counts = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATA_ROOT, "images", split)
        counts[split] = len(glob.glob(os.path.join(img_dir, "*.*")))
    return counts

def compute_rect_score(mask_dir, split):
    masks = glob.glob(os.path.join(mask_dir, split, "*.png"))
    if not masks: return 0
    scores = []
    for m_path in masks:
        mask = cv2.imread(m_path, 0)
        if mask is None: continue
        mask_bin = (mask > 127).astype(np.uint8)
        if np.sum(mask_bin) == 0:
            scores.append(0)
            continue

        # Find bounding box of the mask
        coords = np.column_stack(np.where(mask_bin > 0))
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        # Create a rectangle mask from that bbox
        rect = np.zeros_like(mask_bin)
        rect[y0:y1+1, x0:x1+1] = 1

        # IoU between mask and its own bbox
        intersection = np.sum(np.logical_and(mask_bin, rect))
        union = np.sum(np.logical_or(mask_bin, rect))
        scores.append(intersection / union)
    return np.mean(scores) if scores else 0

def analyze_leaf_areas():
    leaf_dir = os.path.join(DATA_ROOT, "masks_leaf", "train")
    masks = glob.glob(os.path.join(leaf_dir, "*.png"))
    low, high, total = 0, 0, 0
    for m_path in masks:
        mask = cv2.imread(m_path, 0)
        if mask is None: continue
        total += 1
        frac = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
        if frac < 0.05: low += 1
        if frac > 0.95: high += 1
    return low, high, total

def make_contact_sheet(n=30):
    img_dir = os.path.join(DATA_ROOT, "images", "train")
    les_dir = os.path.join(DATA_ROOT, "masks_lesion", "train")
    lf_dir = os.path.join(DATA_ROOT, "masks_leaf", "train")

    all_imgs = glob.glob(os.path.join(img_dir, "*.*"))
    sample = random.sample(all_imgs, min(n, len(all_imgs)))

    # Contact sheet: 5 rows, 6 cols
    cols, rows = 6, 5
    thumb_size = (200, 200)
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), (255, 255, 255))

    for i, img_path in enumerate(sample):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB").resize(thumb_size)

        les_path = os.path.join(les_dir, stem + ".png")
        lf_path = os.path.join(lf_dir, stem + ".png")

        # Overlay
        overlay = np.array(img).copy()
        if os.path.exists(les_path) and os.path.exists(lf_path):
            les = cv2.imread(les_path, 0)
            lf = cv2.imread(lf_path, 0)
            les = cv2.resize(les, thumb_size[::-1], interpolation=cv2.INTER_NEAREST)
            lf = cv2.resize(lf, thumb_size[::-1], interpolation=cv2.INTER_NEAREST)

            les_bin = (les > 127)
            lf_bin = (lf > 127)

            overlay[les_bin] = [255, 0, 0] # Red
            overlay[lf_bin & ~les_bin] = [0, 255, 0] # Green

        res = Image.fromarray(overlay)
        sheet.paste(res, ((i % cols) * thumb_size[0], (i // cols) * thumb_size[1]))

    sheet.save("contact_sheet.png")

if __name__ == "__main__":
    print("--- Corpus Counts ---")
    print(get_counts())

    print("\n--- Rectangularity (Lesion Masks) ---")
    # Check train split as proxy
    print(f"Train rect-score: {compute_rect_score(os.path.join(DATA_ROOT, 'masks_lesion'), 'train'):.4f}")

    print("\n--- Leaf Area Analysis (Train) ---")
    low, high, tot = analyze_leaf_areas()
    print(f"Total: {tot} | <5%: {low} | >95%: {high}")

    print("\nGenerating contact_sheet.png...")
    make_contact_sheet()
    print("Done.")
