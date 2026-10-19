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

# 1. Dataset without augmentations
ds_full = CropGuardDataset(ROOT, split="train", augment=False)
with open("overfit_50_indices.txt") as f:
    indices = [int(line.strip()) for line in f if line.strip()]

ds_50 = Subset(ds_full, indices)
loader = DataLoader(ds_50, batch_size=10, shuffle=True, collate_fn=collate)
eval_loader = DataLoader(ds_50, batch_size=10, shuffle=False, collate_fn=collate)

# 2. Model: dropout off, pretrained=True
model = CropGuardMultiTaskV7(nc=11, pretrained=True).to(device)
dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
for d in dropouts:
    d.p = 0.0

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def seg_loss_custom(seg_logits, lesion_mask, leaf_mask):
    # Lesion channel: pos_weight BCE + Dice
    logits_les = seg_logits[:, 0:1]
    target_les = lesion_mask[:, None] if lesion_mask.dim() == 3 else lesion_mask

    pos_weight = torch.tensor([50.0], device=logits_les.device)
    bce_les = F.binary_cross_entropy_with_logits(logits_les, target_les, pos_weight=pos_weight)

    prob_les = logits_les.sigmoid()
    inter_les = (prob_les * target_les).sum(dim=(2, 3))
    union_les = prob_les.sum(dim=(2, 3)) + target_les.sum(dim=(2, 3))
    dice_les = 1.0 - ((2.0 * inter_les + 1e-4) / (union_les + 1e-4)).mean()

    loss_les = bce_les + dice_les

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

print("Starting 50-image overfit test...")
print(f"{'Epoch':<6} {'Loss_cls':<10} {'Loss_seg':<10} {'Sev_MAE':<10} {'Loss_tot':<10}")

epochs = 100
for ep in range(1, epochs + 1):
    model.train()
    ep_losses = {"cls": 0.0, "seg": 0.0, "total": 0.0}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        cls_loss = F.cross_entropy(out["cls_logits"], batch["cls"])
        seg_loss = seg_loss_custom(out["seg_logits"], batch["mask"], batch["leaf"])
        total_loss = cls_loss + seg_loss
        total_loss.backward()
        optimizer.step()

        ep_losses["cls"] += cls_loss.item()
        ep_losses["seg"] += seg_loss.item()
        ep_losses["total"] += total_loss.item()
        n_batches += 1

    for k in ep_losses:
        ep_losses[k] /= n_batches

    if ep % 10 == 0 or ep == 1:
        with torch.no_grad():
            psevs, gsevs = [], []
            for b in eval_loader:
                b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
                o = model(b["image"], with_analysis=True, active_tasks={"cls", "seg"})
                valid = b["sev_cont"] >= 0.0
                if valid.any():
                    psevs.extend(o["sev_cont"][valid].cpu().numpy().tolist())
                    gsevs.extend(b["sev_cont"][valid].cpu().numpy().tolist())
            sev_mae_ep = np.mean(np.abs(np.array(psevs) - np.array(gsevs)))
        print(f"{ep:<6} {ep_losses['cls']:<10.4f} {ep_losses['seg']:<10.4f} {sev_mae_ep:<10.4f} {ep_losses['total']:<10.4f}")

# Final Evaluation
print("\n--- Evaluating on the same 50 images ---")
model.eval()

all_cls_preds = []
all_cls_gts = []
all_pred_sevs = []
all_gt_sevs = []
lesion_ious = []
lesion_dices = []
non_empty_lesions = 0
total_images = 0

with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        # Cls
        pred_cls = out["cls_logits"].argmax(dim=-1).cpu().numpy()
        gt_cls = batch["cls"].cpu().numpy()
        valid_cls = gt_cls >= 0
        all_cls_preds.extend(pred_cls[valid_cls])
        all_cls_gts.extend(gt_cls[valid_cls])

        # Seg
        seg_probs = torch.sigmoid(out["seg_logits"])
        pred_lesion = (seg_probs[:, 0] > 0.5).float()
        gt_lesion = batch["mask"].float()

        B = pred_lesion.shape[0]
        for i in range(B):
            total_images += 1
            pl = pred_lesion[i]
            gl = gt_lesion[i]

            if pl.sum() > 0:
                non_empty_lesions += 1

            inter = (pl * gl).sum().item()
            union = pl.sum().item() + gl.sum().item() - inter
            iou = (inter + 1e-6) / (union + 1e-6) if union > 0 else 1.0
            dice = (2 * inter + 1e-6) / (pl.sum().item() + gl.sum().item() + 1e-6) if (pl.sum().item() + gl.sum().item()) > 0 else 1.0

            lesion_ious.append(iou)
            lesion_dices.append(dice)

            if batch["sev_cont"][i] >= 0.0:
                all_pred_sevs.append(out["sev_cont"][i].item())
                all_gt_sevs.append(batch["sev_cont"][i].item())

cls_acc = np.mean(np.array(all_cls_preds) == np.array(all_cls_gts))
non_empty_frac = non_empty_lesions / total_images
mean_iou = np.mean(lesion_ious)
mean_dice = np.mean(lesion_dices)
sev_mae = np.mean(np.abs(np.array(all_pred_sevs) - np.array(all_gt_sevs)))
spearman_corr, _ = spearmanr(all_pred_sevs, all_gt_sevs)

print(f"Classification accuracy: {cls_acc:.4f} (Pass bar: >= 0.95)")
print(f"Fraction non-empty lesion predictions: {non_empty_frac:.4f} ({non_empty_lesions}/{total_images})")
print(f"Lesion IoU: {mean_iou:.4f} (Pass bar: >= ~0.8)")
print(f"Lesion Dice: {mean_dice:.4f}")
print(f"Severity MAE: {sev_mae:.4f} (Constant baseline MAE: ~0.038-0.040)")
print(f"Spearman vs computed ratio: {spearman_corr:.4f}")

if cls_acc >= 0.95 and mean_iou >= 0.75 and non_empty_frac > 0.8:
    print("\nOVERFIT GATE PASSED!")
else:
    print("\nOVERFIT GATE FAILED!")
