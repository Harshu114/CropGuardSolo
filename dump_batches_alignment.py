import os
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from model.training.dataset import CropGuardDataset, collate

ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"

# Training DataLoader with augment=False to isolate dataset geometry transforms
ds = CropGuardDataset(ROOT, split="train", size=320, augment=False)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)

os.makedirs("alignment_dumps", exist_ok=True)

batch_count = 0
for b_idx, batch in enumerate(loader):
    if b_idx >= 8: break # 8 batches
    imgs = batch["image"] # (B, 3, 320, 320) in [0, 1]
    masks = batch["mask"] # (B, 320, 320)
    leafs = batch["leaf"] # (B, 320, 320)
    
    B = imgs.shape[0]
    for i in range(B):
        # Image tensor -> uint8 BGR
        im_np = (imgs[i].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        im_bgr = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        
        m_les = (masks[i].numpy() > 0.5)
        m_leaf = (leafs[i].numpy() > 0.5)
        
        overlay = im_bgr.copy()
        overlay[m_leaf] = [0, 255, 0] # green leaf
        overlay[m_les] = [0, 0, 255]  # red lesion
        
        vis = cv2.addWeighted(overlay, 0.45, im_bgr, 0.55, 0)
        cv2.imwrite(f"alignment_dumps/b{b_idx}_img{i}.png", vis)
        batch_count += 1

print(f"Dumped {batch_count} images across 8 batches to alignment_dumps/")

# Create a montage contact sheet: 8 columns x 4 rows
vis_files = sorted(glob := [f"alignment_dumps/b{b}_img{i}.png" for b in range(8) for i in range(4)])
grid = []
for r in range(4):
    row_imgs = [cv2.resize(cv2.imread(vis_files[r*8 + c]), (120, 120)) for c in range(8)]
    grid.append(np.hstack(row_imgs))
montage = np.vstack(grid)
cv2.imwrite("alignment_montage.png", montage)
print("Saved alignment_montage.png")
