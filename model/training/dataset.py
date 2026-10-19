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
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)
        self.root = img_dir
        self.size = size
        self.augment = augment and split == "train"

        # Load exclusion list for leaf/severity targets.
        # Try cwd-relative first (train.py sets cwd to model/training/),
        # then repo-root-relative, then same-dir-as-this-file.
        self.excluded_leaf = set()
        _here = os.path.dirname(os.path.abspath(__file__))
        _repo = os.path.dirname(os.path.dirname(_here))
        for exclusion_path in [
            "leaf_exclusion_list.txt",
            os.path.join(_repo, "leaf_exclusion_list.txt"),
            os.path.join(_here, "leaf_exclusion_list.txt"),
        ]:
            if os.path.exists(exclusion_path):
                with open(exclusion_path, "r") as f:
                    self.excluded_leaf = {line.strip() for line in f if line.strip()}
                break

        # v2 layout: real masks at root/masks/{split}/stem.png (255=leaf, 128=lesion)
        mdir = os.path.join(root, "masks", split)
        self.masks_dir = mdir if os.path.isdir(mdir) else None

        # v7 layout: SEPARATE lesion + leaf mask dirs (from derive_dataset.py)
        les_dir = os.path.join(root, "masks_lesion", split)
        lf_dir = os.path.join(root, "masks_leaf", split)
        self.lesion_masks_dir = les_dir if os.path.isdir(les_dir) else None
        self.leaf_masks_dir = lf_dir if os.path.isdir(lf_dir) else None
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
    def _augment(self, img, boxes, lesion_mask=None, leaf_mask=None,
                 combined_mask=None, flip=None, scale=None):
        """Apply one shared geometric transform to image, boxes, and masks.

        Image resizing is bilinear.  Every native-resolution mask is transformed
        with the identical flip/scale/crop or pad using nearest-neighbour.
        """
        if flip is None:
            flip = random.random() < 0.5
        if scale is None:
            scale = random.uniform(0.9, 1.1)
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]
        # HSV jitter
        if random.random() < 0.5:
            import torchvision.transforms.functional as TF
            img = TF.adjust_hue(img, random.uniform(-0.02, 0.02))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
        # scale jitter (pad or center-crop) -- keep boxes valid
        s = scale
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
        # ONE geometry path for every mask: the SAME flip and scale the image
        # just received (image bilinear, masks nearest). Previously the flip
        # was hardcoded off here and applied inline above, which is easy to
        # break -- now the flip flag is threaded through explicitly and
        # asserted against the image in the unit tests.
        def transform(m):
            return self._transform_mask(m, flip=flip, s=s) if m is not None else None
        return img, boxes, transform(lesion_mask), transform(leaf_mask), transform(combined_mask)

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
    def _paste_background(self, img_pil, leaf_mask_fullres, boxes):
        """Leaf-mask background replacement (expert Tier-1 #4).

        Cuts the leaf out with the leaf mask and pastes it on a random
        synthetic field background -- the most reliable lab-to-field fix.
        Returns (new PIL image). Geometry of boxes is UNCHANGED (now only
        color/context change).
        """
        w, h = img_pil.size
        lm = np.asarray(leaf_mask_fullres.resize((w, h), Image.NEAREST))
        cut = np.asarray(img_pil).copy()
        cut[lm == 0] = 0
        # random field background: soil gradient + grass noise
        bg = np.zeros((h, w, 3), np.uint8)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        soil = (np.array([96, 74, 52]) * (0.7 + 0.3 * t)).astype(np.uint8)
        bg[:] = soil
        noise = np.random.default_rng(0).integers(-12, 12, (h, w, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out = bg.copy()
        out[lm > 0] = cut[lm > 0]
        return Image.fromarray(out)

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

        # v7 SEPARATE lesion + leaf masks take priority (native resolution)
        raw_lesion, raw_leaf = None, None
        stem = os.path.splitext(os.path.basename(img_path))[0]
        if self.lesion_masks_dir is not None:
            mp = os.path.join(self.lesion_masks_dir, stem + ".png")
            if os.path.exists(mp):
                raw_lesion = cv2.imread(mp, 0)
        if self.leaf_masks_dir is not None:
            lp = os.path.join(self.leaf_masks_dir, stem + ".png")
            if os.path.exists(lp):
                raw_leaf = cv2.imread(lp, 0)

        # v2 combined mask (255=leaf, 128=lesion) as fallback
        raw_mask = None
        if self.masks_dir is not None:
            mp = os.path.join(self.masks_dir, stem + ".png")
            if os.path.exists(mp):
                raw_mask = cv2.imread(mp, 0)

        flip, s = False, 1.0
        if self.augment:
            flip = random.random() < 0.5
            s = random.uniform(0.9, 1.1)
        if self.augment:
            img, boxes, raw_lesion, raw_leaf, raw_mask = self._augment(
                img, boxes, raw_lesion, raw_leaf, raw_mask, flip=flip, scale=s)

        img = img.resize((self.size, self.size), Image.BILINEAR)

        # v7 leaf-mask paste augmentation (train split only, 30% of images)
        if self.augment and raw_leaf is not None and random.random() < 0.3:
            img = self._paste_background(img, Image.fromarray(raw_leaf), boxes)

        # resize masks to model resolution
        if raw_lesion is not None:
            raw_lesion = cv2.resize(raw_lesion, (self.size, self.size),
                                    interpolation=cv2.INTER_NEAREST)
        if raw_leaf is not None:
            raw_leaf = cv2.resize(raw_leaf, (self.size, self.size),
                                  interpolation=cv2.INTER_NEAREST)

        # targets
        if boxes:
            cls = int(np.bincount([b[0] for b in boxes],
                                  minlength=11).argmax())
        else:
            cls = -1  # background: excluded from cls loss

        # v7: leaf channel for the 2-channel seg head.
        # Priority: dedicated leaf-mask dir > combined mask's leaf (255) region
        # > all-ones fallback (severity degrades to lesion/img -- still honest).
        if raw_leaf is not None:
            leaf_mask = (raw_leaf > 127).astype(np.float32)
            # EXCLUSION: Mark failures from the exclusion list as -1.0
            if stem in self.excluded_leaf:
                leaf_mask[:] = -1.0
        elif raw_mask is not None:

            leaf_mask = (cv2.resize(raw_mask, (self.size, self.size),
                                    interpolation=cv2.INTER_NEAREST) == 255
                         ).astype(np.float32)
        else:
            leaf_mask = np.ones((self.size, self.size), dtype=np.float32)
        # and when a dedicated lesion mask exists it beats the combined-mask
        # heuristics:
        if raw_lesion is not None:
            mask = (raw_lesion > 127).astype(np.float32)
            lesion_px = float((raw_lesion > 127).sum())
            leaf_px = float(leaf_mask.sum())
            ratio = lesion_px / max(leaf_px, 1.0)
            ratio = float(np.clip(ratio, 0.0, 1.0))
            has = leaf_px > 0 and ((len(boxes) > 0) or (lesion_px > 0)) and cls >= 0
            sev_cont, sev_ord = severity_from_ratio(ratio, has)
        else:
            mask, sev_cont, sev_ord = None, 0.0, 0        # legacy fills below


        box_mask = make_pseudo_mask(boxes, self.size)
        if raw_lesion is None and raw_mask is not None:
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
        elif raw_lesion is None and mask is None:
            # legacy fallback: pseudo-mask from boxes
            mask = box_mask
            ratio = float(mask.mean())
            sev_cont, sev_ord = severity_from_ratio(ratio, len(boxes) > 0)

        # EXCLUSION sentinel: if this image's leaf mask is flagged as bad,
        # emit sev_cont=-1.0 / sev_ord=-1 so sev_loss_fn skips it entirely.
        # The leaf channel already carries -1.0 pixels (set above), which
        # _dice_bce treats as "no supervision".  cls and lesion-seg targets
        # remain intact so those heads still train on these images.
        if stem in self.excluded_leaf:
            sev_cont = -1.0
            sev_ord = -1

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
            "leaf": torch.from_numpy(leaf_mask),
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
        "leaf": torch.stack([b["leaf"] for b in batch]),
        "sev_cont": torch.stack([b["sev_cont"] for b in batch]),
        "sev_ord": torch.stack([b["sev_ord"] for b in batch]),
    }
    return out
