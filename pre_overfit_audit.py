import os
import glob
import numpy as np
import cv2
from PIL import Image

DATA_ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"

def check_fallback():
    fallback_count = 0
    total_images = 0

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(DATA_ROOT, "images", split)
        les_dir = os.path.join(DATA_ROOT, "masks_lesion", split)

        imgs = glob.glob(os.path.join(img_dir, "*.*"))
        for img_p in imgs:
            total_images += 1
            stem = os.path.splitext(os.path.basename(img_p))[0]
            les_p = os.path.join(les_dir, stem + ".png")
            if not os.path.exists(les_p):
                fallback_count += 1

    print(f"Total Images: {total_images}")
    print(f"Fallback hits: {fallback_count}")
    return fallback_count

def analyze_1_5_percent():
    leaf_dir = os.path.join(DATA_ROOT, "masks_leaf", "train")
    masks = glob.glob(os.path.join(leaf_dir, "*.png"))

    targets = []
    for m_path in masks:
        mask = cv2.imread(m_path, 0)
        if mask is None: continue
        frac = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
        if 0.01 <= frac < 0.05:
            targets.append(m_path)

    print(f"Images in [1%, 5%): {len(targets)}")

    failures = 0
    for m_path in targets:
        mask = cv2.imread(m_path, 0)
        # Heuristic: if it's just a small blob in the middle of a black screen
        # while the image clearly has a leaf, it's a failure.
        # I'll save them for a quick visual check.
        stem = os.path.splitext(os.path.basename(m_path))[0]
        img_p = os.path.join(DATA_ROOT, "images", "train", stem + ".jpg")
        if not os.path.exists(img_p): img_p = os.path.join(DATA_ROOT, "images", "train", stem + ".png")

        if os.path.exists(img_p):
            img = cv2.imread(img_p)
            mask_res = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = img.copy()
            overlay[mask_res > 127] = [0, 255, 0]
            res = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            cv2.imwrite(f"check_1_5_{stem}.png", res)

    return targets

def compute_baseline():
    train_les_dir = os.path.join(DATA_ROOT, "masks_lesion", "train")
    train_lf_dir = os.path.join(DATA_ROOT, "masks_leaf", "train")

    train_sevs = []
    stems = [os.path.splitext(f)[0] for f in os.listdir(train_les_dir) if f.endswith(".png")]
    for stem in stems:
        lp = os.path.join(train_lf_dir, stem + ".png")
        if not os.path.exists(lp): continue
        les = cv2.imread(os.path.join(train_les_dir, stem + ".png"), 0)
        lf = cv2.imread(lp, 0)
        lf_a = np.sum(lf > 127)
        if lf_a > 0:
            train_sevs.append(min(1.0, np.sum(les > 127) / lf_a))

    train_median = np.median(train_sevs)
    print(f"Train Median Severity: {train_median:.4f}")

    for split in ["val", "test"]:
        les_dir = os.path.join(DATA_ROOT, "masks_lesion", split)
        lf_dir = os.path.join(DATA_ROOT, "masks_leaf", split)

        val_sevs = []
        stems_s = [os.path.splitext(f)[0] for f in os.listdir(les_dir) if f.endswith(".png")]
        for stem in stems_s:
            lp = os.path.join(lf_dir, stem + ".png")
            if not os.path.exists(lp): continue
            les = cv2.imread(os.path.join(les_dir, stem + ".png"), 0)
            lf = cv2.imread(lp, 0)
            lf_a = np.sum(lf > 127)
            if lf_a > 0:
                val_sevs.append(min(1.0, np.sum(les > 127) / lf_a))

        if val_sevs:
            mae = np.mean(np.abs(np.array(val_sevs) - train_median))
            print(f"Baseline MAE on {split}: {mae:.4f}")

if __name__ == "__main__":
    print("--- Fallback Check ---")
    check_fallback()

    print("\n--- 1-5% Area Analysis ---")
    analyze_1_5_percent()

    print("\n--- Severity Baseline ---")
    compute_baseline()
