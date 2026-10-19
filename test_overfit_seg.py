import os, sys, random, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if os.path.join(_ROOT, "model", "training") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "model", "training"))

from model.architecture.v7_model import CropGuardMultiTaskV7
from model.training.dataset import CropGuardDataset, collate

ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_full = CropGuardDataset(ROOT, split="train", augment=False)
with open("overfit_50_indices.txt") as f:
    indices = [int(line.strip()) for line in f if line.strip()]

ds_50 = Subset(ds_full, indices)
loader = DataLoader(ds_50, batch_size=16, shuffle=True, collate_fn=collate)
eval_loader = DataLoader(ds_50, batch_size=16, shuffle=False, collate_fn=collate)

model = CropGuardMultiTaskV7(nc=11, pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def seg_loss_custom(seg_logits, lesion_mask, leaf_mask):
    # Lesion channel: Dice + BCE with pos_weight or focal
    logits_les = seg_logits[:, 0:1]
    target_les = lesion_mask[:, None] if lesion_mask.dim() == 3 else lesion_mask

    # Dice for lesion (scale up dice term so it heavily penalizes missing lesions)
    prob_les = logits_les.sigmoid()
    inter_les = (prob_les * target_les).sum(dim=(2, 3))
    union_les = prob_les.sum(dim=(2, 3)) + target_les.sum(dim=(2, 3))
    dice_les = 1.0 - ((2.0 * inter_les + 1e-4) / (union_les + 1e-4)).mean()

    # Focal or weighted BCE for lesion
    pos_weight = torch.tensor([50.0], device=logits_les.device)
    bce_les = F.binary_cross_entropy_with_logits(logits_les, target_les, pos_weight=pos_weight)

    loss_les = bce_les + 3.0 * dice_les

    # Leaf channel
    logits_leaf = seg_logits[:, 1:2]
    target_leaf = leaf_mask[:, None] if leaf_mask.dim() == 3 else leaf_mask
    valid_leaf = (target_leaf != -1.0).float()

    bce_leaf = (F.binary_cross_entropy_with_logits(logits_leaf, target_leaf.clamp(0, 1), reduction="none") * valid_leaf).sum() / (valid_leaf.sum() + 1e-6)
    prob_leaf = logits_leaf.sigmoid()
    inter_lf = (prob_leaf * target_leaf.clamp(0, 1) * valid_leaf).sum(dim=(2, 3))
    union_lf = (prob_leaf * valid_leaf).sum(dim=(2, 3)) + (target_leaf.clamp(0, 1) * valid_leaf).sum(dim=(2, 3))
    dice_leaf = 1.0 - ((2.0 * inter_lf + 1.0) / (union_lf + 1.0)).mean()

    loss_leaf = bce_leaf + dice_leaf
    return loss_les + loss_leaf

print("Training with custom seg loss...")
for ep in range(1, 51):
    model.train()
    ep_cls, ep_seg = 0.0, 0.0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        cls_loss = F.cross_entropy(out["cls_logits"], batch["cls"])
        seg_loss = seg_loss_custom(out["seg_logits"], batch["mask"], batch["leaf"])
        loss = cls_loss + seg_loss
        loss.backward()
        optimizer.step()
        ep_cls += cls_loss.item()
        ep_seg += seg_loss.item()

    if ep % 10 == 0 or ep == 1:
        print(f"Ep {ep:<3}: Cls={ep_cls/len(loader):.4f}, Seg={ep_seg/len(loader):.4f}")

# Eval
model.eval()
ious, dices, non_empty = [], [], 0
pred_sevs, gt_sevs = [], []
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        seg_probs = torch.sigmoid(out["seg_logits"])
        pred_lesion = (seg_probs[:, 0] > 0.5).float()
        gt_lesion = batch["mask"].float()
        for i in range(pred_lesion.shape[0]):
            pl = pred_lesion[i]
            gl = gt_lesion[i]
            if pl.sum() > 0: non_empty += 1
            inter = (pl * gl).sum().item()
            union = pl.sum().item() + gl.sum().item() - inter
            iou = (inter + 1e-6) / (union + 1e-6) if union > 0 else 1.0
            dice = (2 * inter + 1e-6) / (pl.sum().item() + gl.sum().item() + 1e-6) if (pl.sum().item() + gl.sum().item()) > 0 else 1.0
            ious.append(iou)
            dices.append(dice)
            if batch["sev_cont"][i] >= 0.0:
                pred_sevs.append(out["sev_cont"][i].item())
                gt_sevs.append(batch["sev_cont"][i].item())

print(f"Non-empty fraction: {non_empty}/50 ({non_empty/50:.2f})")
print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Mean Dice: {np.mean(dices):.4f}")
print(f"Severity MAE: {np.mean(np.abs(np.array(pred_sevs) - np.array(gt_sevs))):.4f}")
