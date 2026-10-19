import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

class CropGuardAugmentor:
    """
    Professional Augmentation Pipeline for CropGuard.
    Simulates real-world field conditions to prevent overfitting of the FasterNet backbone.
    """
    def __init__(self, image_dir, mask_dir, output_dir, augment_factor=3):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.augment_factor = augment_factor

        # Create output directories
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "masks").mkdir(parents=True, exist_ok=True)

        # Define the "Field Simulation" pipeline
        self.transform = A.Compose([
            # 1. Geometric Transforms (Simulate leaf curling and camera angles)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),

            # 2. Lighting & Color (Simulate weather and sensor variance)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),

            # 3. Sensor Noise (Simulate cheap mobile cameras)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # 4. Final normalization
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def apply_augmentations(self):
        """Processes images and masks, generating 'augment_factor' versions of each."""
        images = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))

        print(f"Applying field simulations to {len(images)} images...")

        for img_path in tqdm(images):
            img_id = img_path.stem
            mask_path = self.mask_dir / f"{img_id}_mask.png"

            if not mask_path.exists():
                continue

            # Load image and mask
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Save original first
            cv2.imwrite(str(self.output_dir / "images" / f"{img_id}_orig.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self.output_dir / "masks" / f"{img_id}_orig_mask.png"), mask)

            # Generate augmented versions
            for i in range(self.augment_factor):
                augmented = self.transform(image=image, mask=mask)
                aug_img = augmented['image']
                aug_mask = augmented['mask']

                # Convert back to numpy for saving (removing tensor wrapper)
                aug_img_np = aug_img.permute(1, 2, 0).numpy().astype(np.uint8)
                # Denormalize for saving
                aug_img_np = (aug_img_np * 255).astype(np.uint8)

                # Save augmented pair
                cv2.imwrite(str(self.output_dir / "images" / f"{img_id}_aug_{i}.jpg"), cv2.cvtColor(aug_img_np, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(self.output_dir / "masks" / f"{img_id}_aug_{i}_mask.png"), aug_mask.numpy().astype(np.uint8))

if __name__ == "__main__":
    print("Augmentation Pipeline initialized. Ready to generate field-simulated data.")
