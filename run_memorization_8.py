import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from model.architecture.v7_model import CropGuardMultiTaskV7
from model.training.dataset import CropGuardDataset, collate
from model.training.losses import seg_loss_v7

ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select 8 images: 6 diseased (diverse classes) + 2 healthy images
ds = CropGuardDataset(ROOT, split="train", size=320, augment=False)
indices_8 = [0, 1, 2, 3, 4, 5, 713, 714] # includes diseased and healthy
ds_8 = Subset(ds, indices_8)
loader = DataLoader(ds_8, batch_size=8, shuffle=False, collate_fn=collate)

# Model: lr 1e-3 for heads, 1e-4 for backbone
model = CropGuardMultiTaskV7(nc=8, pretrained=True).to(device)
for d in [m for m in model.modules() if isinstance(m, nn.Dropout)]:
    d.p = 0.0

backbone_params = list(model.backbone.parameters())
head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params, "lr": 1e-3}
], weight_decay=1e-4)

# Confirm focal-Tversky/BCE handles empty lesion masks without NaNs
batch_test = next(iter(loader))
b_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch_test.items()}
out_test = model(b_gpu["image"], active_tasks={"seg"})
loss_check = seg_loss_v7(out_test["seg_logits"], b_gpu["mask"], b_gpu["leaf"])
print(f"NaN check on batch with healthy (empty lesion): loss={loss_check.item():.4f}, isnan={torch.isnan(loss_check).item()}")
assert not torch.isnan(loss_check), "NaN in seg loss!"

print("\nStarting 8-image memorization test (500 optimizer steps)...")
model.train()
for step in range(1, 501):
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["image"], active_tasks={"seg"})
        loss = seg_loss_v7(out["seg_logits"], batch["mask"], batch["leaf"])
        loss.backward()
        optimizer.step()

    if step % 100 == 0 or step == 1:
        model.eval()
        with torch.no_grad():
            out_eval = model(batch["image"], active_tasks={"seg"})
            prob_les = torch.sigmoid(out_eval["seg_logits"][:, 0]).cpu()
            gt_les = batch["mask"].cpu()
            
            # Find best threshold among [0.3, 0.4, 0.5, 0.6, 0.7]
            best_iou, best_dice, best_thr = 0.0, 0.0, 0.5
            for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
                pred = (prob_les > thr).float()
                ious, dices = [], []
                for i in range(8):
                    p = pred[i]; g = gt_les[i]
                    inter = (p * g).sum().item()
                    union = p.sum().item() + g.sum().item() - inter
                    iou = inter / union if union > 0 else (1.0 if p.sum() == 0 and g.sum() == 0 else 0.0)
                    dice = (2 * inter) / (p.sum().item() + g.sum().item()) if (p.sum().item() + g.sum().item()) > 0 else (1.0 if p.sum() == 0 and g.sum() == 0 else 0.0)
                    ious.append(iou)
                    dices.append(dice)
                m_iou = np.mean(ious)
                if m_iou > best_iou:
                    best_iou = m_iou
                    best_dice = np.mean(dices)
                    best_thr = thr
            print(f"Step {step:3d}: seg_loss = {loss.item():.4f} | Best Thr = {best_thr:.1f} | Mean IoU = {best_iou:.4f} | Mean Dice = {best_dice:.4f}")
        model.train()

print(f"\nFinal Memorization Result at Step 500: Best IoU = {best_iou:.4f} (Pass bar: >= 0.50)")
if best_iou >= 0.50:
    print("MEMORIZATION GATE PASSED!")
else:
    print("MEMORIZATION GATE FAILED!")
