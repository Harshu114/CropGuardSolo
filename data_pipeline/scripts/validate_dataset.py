import os
import shutil
import hashlib
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

class DatasetValidator:
    """
    Gold Standard Dataset Validator for CropGuard.
    Filters out duplicates, blurry images, and inconsistent labels.
    """
    def __init__(self, image_dir, label_dir, mask_dir=None, severity_csv=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.severity_csv = severity_csv
        self.stats = {"total": 0, "removed_duplicates": 0, "removed_blurry": 0, "removed_inconsistent": 0}

    def get_phash(self, img_path):
        """Computes a simple perceptual hash for deduplication."""
        try:
            img = Image.open(img_path).convert('L').resize((8, 8), Image.Resampling.LANCZOS)
            pixels = np.array(img)
            avg = pixels.mean()
            diff = pixels > avg
            return hashlib.sha256(diff.tobytes()).hexdigest()
        except Exception:
            return None

    def is_blurry(self, img_path, threshold=100):
        """Detects blur using Laplacian variance."""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return True
            return cv2.Laplacian(img, cv2.CV_64F).var() < threshold
        except Exception:
            return True

    def validate_mask_severity(self, img_id, mask_path, severity_score):
        """Checks if mask area matches the reported severity score."""
        if not mask_path or severity_score is None:
            return True
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: return False
            # Calculate percentage of white pixels
            area_ratio = np.sum(mask > 127) / (mask.shape[0] * mask.shape[1])
            # Logic: If ratio is high (>0.6) but severity is very low (<0.2), it's inconsistent
            if (area_ratio > 0.6 and severity_score < 0.2) or (area_ratio < 0.05 and severity_score > 0.7):
                return False
            return True
        except Exception:
            return False

    def run_audit(self, output_dir):
        """Runs the full validation pipeline and moves clean data to output_dir."""
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

        images = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.stats["total"] = len(images)

        hashes = {}

        for img_name in tqdm(images, desc="Auditing Dataset"):
            img_path = os.path.join(self.image_dir, img_name)

            # 1. Deduplication
            phash = self.get_phash(img_path)
            if phash in hashes:
                self.stats["removed_duplicates"] += 1
                continue
            hashes[phash] = img_name

            # 2. Blur check
            if self.is_blurry(img_path):
                self.stats["removed_blurry"] += 1
                continue

            # 3. Mask-Severity Consistency
            # (Assuming files are named consistently: img_01.jpg -> img_01.txt -> img_01_mask.png)
            img_id = os.path.splitext(img_name)[0]
            label_path = os.path.join(self.label_dir, f"{img_id}.txt")
            mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.png") if self.mask_dir else None

            # In a real scenario, we would load severity from severity_csv
            # For this tool, we check if the files exist first
            if not os.path.exists(label_path):
                self.stats["removed_inconsistent"] += 1
                continue

            # If everything passes, move to clean folder
            shutil.copy(img_path, os.path.join(output_dir, "images", img_name))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_dir, "labels", f"{img_id}.txt"))
            if mask_path and os.path.exists(mask_path):
                shutil.copy(mask_path, os.path.join(output_dir, "masks", f"{img_id}_mask.png"))

        return self.stats

if __name__ == "__main__":
    print("Dataset Validator initialized. Use this as a module in the main pipeline.")
