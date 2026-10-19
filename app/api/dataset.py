"""CropGuard dataset: tolerant YOLO-label loader + multi-task targets.

Works with BOTH old (6-field) and standard (5-field) label lines, and treats
missing .txt files as background images (no boxes) -- so the dataset works
as-is on Kaggle without re-uploading.

Targets generated per image:
  - det: YOLO bboxes (class, cx, cy, w, h) normalized
  - cls: majority bbox class (background = -1, excluded from cls loss)
  - seg: pseudo-mask (bbox rectangles filled = lesion), float32 {0,1}
  - sev: continuous severity in [0,1] = lesion area ratio (from pseudo-mask)
         and ordinal stage: 0 healthy / 1 mild(<10%) / 2 moderate(<30%) / 3 severe
"""
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_labels(label_path):
    """Returns list of (cls:int, cx, cy, w, h). Tolerates 6-field legacy lines.
    Missing file -> empty list (background image)."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    try:
        with open(label_path) as f:
            for line in f:
                p = line.split()
                if len(p) < 5:
                    continue
                try:
                    c, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
                except ValueError:
                    continue
                boxes.append((c, cx, cy, w, h))
    except OSError:
        return []
    return boxes


def make_pseudo_mask(boxes, size):
    """Fill normalized bbox rectangles -> (size,size) float32 mask {0,1}."""
    m = np.zeros((size, size), dtype=np.float32)
    for _, cx, cy, w, h in boxes:
        x1 = max(int((cx - w / 2) * size), 0)
        y1 = max(int((cy - h / 2) * size), 0)
        x2 = min(int((cx + w / 2) * size), size)
        y2 = min(int((cy + h / 2) * size), size)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1.0
    return m


def severity_from_ratio(ratio, has_lesion):
    """Paper protocol: 0% healthy, <10% mild, <30% moderate, >=30% severe."""
    if not has_lesion:
        return 0.0, 0
    cont = float(np.clip(ratio, 0.0, 1.0))
    if cont < 0.10:
        return cont, 1
    if cont < 0.30:
        return cont, 2
    return cont, 3


class CropGuardDataset(Dataset):
    def __init__(self, root, split="train", size=320, augment=True,
                 max_images=None):
        # supports both layouts: root/images/{split} (Kaggle upload) and
        # root/{split}/images (data.yaml style)
        img_a = os.path.join(root, "images", split)
        img_b = os.path.join(root, split, "images")
        if os.path.isdir(img_a):
            img_dir, lbl_dir = img_a, os.path.join(root, "labels", split)
        elif os.path.isdir(img_b):
            img_dir, lbl_dir = img_b, os.path.join(root, split, "labels")
        else:
            raise RuntimeError(f"No image dir for split '{split}' under {root}")
        self.root = img_dir
        self.size = size
        self.augment = augment and split == "train"
        # v2 layout: real masks at root/masks/{split}/stem.png (255=leaf, 128=lesion)
        mdir = os.path.join(root, "masks", split)
        self.masks_dir = mdir if os.path.isdir(mdir) else None
        self.samples = []
        names = sorted(os.listdir(img_dir)) if os.path.isdir(img_dir) else []
        if max_images:
            names = names[:max_images]
        for n in names:
            stem, ext = os.path.splitext(n)
            if ext.lower() not in IMG_EXTS:
                continue
            self.samples.append((
                os.path.join(img_dir, n),
                os.path.join(lbl_dir, stem + ".txt"),
            ))
        if not self.samples:
            raise RuntimeError(f"No images found under {img_dir}")

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------- augment ---
    def _augment(self, img, boxes):
        """Geometric+photometric augs on PIL image and normalized boxes.
        Pseudo-mask is re-rendered from the final boxes, so it always stays
        consistent with the geometry -- no separate mask transform needed."""
        # horizontal flip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]
        # HSV jitter
        if random.random() < 0.5:
            import torchvision.transforms.functional as TF
            img = TF.adjust_hue(img, random.uniform(-0.02, 0.02))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        # scale jitter (pad or center-crop) -- keep boxes valid
        s = random.uniform(0.9, 1.1)
        if abs(s - 1.0) > 1e-3:
            W, H = img.size
            nw, nh = int(W * s), int(H * s)
            img = img.resize((nw, nh), Image.BILINEAR)
            if s < 1.0:  # pad back to original size
                canvas = Image.new("RGB", (W, H), (114, 114, 114))
                canvas.paste(img, ((W - nw) // 2, (H - nh) // 2))
                img = canvas
                boxes = [(c, 0.5 + (cx - 0.5) * s, 0.5 + (cy - 0.5) * s,
                          w * s, h * s) for c, cx, cy, w, h in boxes]
            else:  # center crop back to original size
                x1, y1 = (nw - W) // 2, (nh - H) // 2
                img = img.crop((x1, y1, x1 + W, y1 + H))
                boxes = [(c, 0.5 + (cx - 0.5) / s, 0.5 + (cy - 0.5) / s,
                          w / s, h / s) for c, cx, cy, w, h in boxes]
        return img, boxes

    @staticmethod
    def _clip_boxes(boxes):
        out = []
        for c, cx, cy, w, h in boxes:
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            if w <= 0.001 or h <= 0.001 or w > 1.2 or h > 1.2:
                continue
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            out.append((int(c), cx, cy, min(w, 1.0), min(h, 1.0)))
        return out

    # ---------------------------------------------------------------- get ---
    def _transform_mask(self, m, flip, s):
        """Apply the same geometry as _augment to a mask (NEAREST)."""
        if flip:
            m = m[:, ::-1]
        if abs(s - 1.0) > 1e-3:
            H, W = m.shape[:2]
            nh, nw = int(H * s), int(W * s)
            m = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
            if s < 1.0:
                canvas = np.zeros((H, W), m.dtype)
                y, x = (H - nh) // 2, (W - nw) // 2
                canvas[y:y + nh, x:x + nw] = m
                m = canvas
            else:
                y, x = (nh - H) // 2, (nw - W) // 2
                m = m[y:y + H, x:x + W]
        return m

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = load_yolo_labels(lbl_path)
        boxes = self._clip_boxes(boxes)

        # v2 real mask (native resolution) if present
        raw_mask = None
        if self.masks_dir is not None:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            mp = os.path.join(self.masks_dir, stem + ".png")
            if os.path.exists(mp):
                raw_mask = cv2.imread(mp, 0)

        flip, s = False, 1.0
        if self.augment:
            flip = random.random() < 0.5
            s = random.uniform(0.9, 1.1)
        if self.augment:
            img, boxes = self._augment(img, boxes)
        if raw_mask is not None and self.augment:
            raw_mask = self._transform_mask(raw_mask, flip, s)

        img = img.resize((self.size, self.size), Image.BILINEAR)

        # targets
        if boxes:
            cls = int(np.bincount([b[0] for b in boxes],
                                  minlength=11).argmax())
        else:
            cls = -1  # background: excluded from cls loss

        box_mask = make_pseudo_mask(boxes, self.size)
        if raw_mask is not None:
            # NOTE: the masks in this corpus are LEAF masks -- values are binary
            # {0, 255} and the value 128 (lesion) NEVER appears (verified over the
            # whole corpus). So a lesion/leaf ratio cannot be read off the mask
            # directly; doing so silently pins severity to a constant 0/"mild"
            # for every diseased image and the severity head learns nothing.
            leaf_px = float((raw_mask == 255).sum())
            lesion_px = float((raw_mask == 128).sum())
            m = cv2.resize(raw_mask, (self.size, self.size),
                           interpolation=cv2.INTER_NEAREST)
            if lesion_px > 0:
                # true lesion mask -> ratio straight from the mask
                ratio = lesion_px / max(leaf_px, 1.0)
                mask = (m == 128).astype(np.float32)
            else:
                # leaf-only mask -> lesion area comes from the boxes, normalised
                # by the LEAF area so severity means "how much of the leaf is
                # affected". Normalising by the whole image would instead track
                # leaf size, not disease severity.
                leaf_frac = leaf_px / float(raw_mask.size)
                ratio = float(box_mask.mean()) / max(leaf_frac, 1e-6)
                mask = box_mask
            ratio = float(np.clip(ratio, 0.0, 1.0))
            has = leaf_px > 0 and cls >= 0
            sev_cont, sev_ord = severity_from_ratio(ratio, has)
            if mask.sum() < 4:      # no visible lesion -> bbox rectangles
                mask = box_mask
        else:
            # legacy fallback: pseudo-mask from boxes
            mask = box_mask
            ratio = float(mask.mean())
            sev_cont, sev_ord = severity_from_ratio(ratio, len(boxes) > 0)

        det = torch.zeros((max(len(boxes), 1), 5), dtype=torch.float32)
        for i, (c, cx, cy, w, h) in enumerate(boxes):
            det[i] = torch.tensor([c, cx, cy, w, h], dtype=torch.float32)

        import torchvision.transforms.functional as TF
        x = TF.to_tensor(img)

        return {
            "image": x,
            "det_boxes": det,
            "n_boxes": len(boxes),
            "cls": cls,
            "mask": torch.from_numpy(mask),
            "sev_cont": torch.tensor(sev_cont, dtype=torch.float32),
            "sev_ord": torch.tensor(sev_ord, dtype=torch.long),
        }


def collate(batch):
    """Detection-style collate: variable box counts kept as list; rest stacked."""
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "det_boxes": [b["det_boxes"] for b in batch],
        "n_boxes": torch.tensor([b["n_boxes"] for b in batch]),
        "cls": torch.tensor([b["cls"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "sev_cont": torch.stack([b["sev_cont"] for b in batch]),
        "sev_ord": torch.stack([b["sev_ord"] for b in batch]),
    }
    return out
