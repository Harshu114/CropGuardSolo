import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset

from model.architecture.v7_model import CropGuardMultiTaskV7
from model.training.dataset import CropGuardDataset, collate

ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ds_full = CropGuardDataset(ROOT, split="train", augment=False)
with open("overfit_50_indices.txt") as f:
    indices = [int(line.strip()) for line in f if line.strip()]

ds_50 = Subset(ds_full, indices)
eval_loader = DataLoader(ds_50, batch_size=10, shuffle=False, collate_fn=collate)

# Train 50 epochs with pos_weight=50 setup
model = CropGuardMultiTaskV7(nc=11, pretrained=True).to(device)
for d in [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]:
    d.p = 0.0

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def seg_loss_pw(seg_logits, lesion_mask, leaf_mask):
    logits_les = seg_logits[:, 0:1]
    target_les = lesion_mask[:, None] if lesion_mask.dim() == 3 else lesion_mask
    pos_weight = torch.tensor([50.0], device=logits_les.device)
    bce_les = F.binary_cross_entropy_with_logits(logits_les, target_les, pos_weight=pos_weight)
    prob_les = logits_les.sigmoid()
    inter_les = (prob_les * target_les).sum(dim=(2, 3))
    union_les = prob_les.sum(dim=(2, 3)) + target_les.sum(dim=(2, 3))
    dice_les = 1.0 - ((2.0 * inter_les + 1e-4) / (union_les + 1e-4)).mean()

    logits_leaf = seg_logits[:, 1:2]
    target_leaf = leaf_mask[:, None] if leaf_mask.dim() == 3 else leaf_mask
    valid_leaf = (target_leaf != -1.0).float()
    bce_leaf = (F.binary_cross_entropy_with_logits(logits_leaf, target_leaf.clamp(0, 1), reduction="none") * valid_leaf).sum() / (valid_leaf.sum() + 1e-6)
    prob_leaf = logits_leaf.sigmoid()
    inter_lf = (prob_leaf * target_leaf.clamp(0, 1) * valid_leaf).sum(dim=(2, 3))
    union_lf = (prob_leaf * valid_leaf).sum(dim=(2, 3)) + (target_leaf.clamp(0, 1) * valid_leaf).sum(dim=(2, 3))
    dice_leaf = 1.0 - ((2.0 * inter_lf + 1.0) / (union_lf + 1.0)).mean()
    return bce_les + dice_les + bce_leaf + dice_leaf

train_loader = DataLoader(ds_50, batch_size=10, shuffle=True, collate_fn=collate)
for ep in range(50):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        cls_loss = F.cross_entropy(out["cls_logits"], batch["cls"])
        seg_loss = seg_loss_pw(out["seg_logits"], batch["mask"], batch["leaf"])
        (cls_loss + seg_loss).backward()
        optimizer.step()

# Detailed evaluation across thresholds
model.eval()
probs_all = []
gt_all = []
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        prob = torch.sigmoid(out["seg_logits"][:, 0]).cpu()
        probs_all.append(prob)
        gt_all.append(batch["mask"].cpu())

probs = torch.cat(probs_all, dim=0) # (50, 320, 320)
gts = torch.cat(gt_all, dim=0)     # (50, 320, 320)

true_frac = (gts > 0.5).float().mean().item()
print(f"True lesion pixel fraction: {true_frac:.5f} ({true_frac*100:.3f}%)")

for thr in [0.3, 0.5, 0.7]:
    preds = (probs > thr).float()
    pred_frac = preds.mean().item()
    non_empty = int((preds.sum(dim=(1, 2)) > 0).sum().item())
    
    ious, dices = [], []
    for i in range(50):
        p = preds[i]
        g = gts[i]
        inter = (p * g).sum().item()
        union = p.sum().item() + g.sum().item() - inter
        iou = inter / union if union > 0 else (1.0 if p.sum() == 0 and g.sum() == 0 else 0.0)
        dice = (2 * inter) / (p.sum().item() + g.sum().item()) if (p.sum().item() + g.sum().item()) > 0 else (1.0 if p.sum() == 0 and g.sum() == 0 else 0.0)
        ious.append(iou)
        dices.append(dice)
    
    print(f"\nThreshold {thr:.1f}:")
    print(f"  Predicted pixel fraction: {pred_frac:.5f} ({pred_frac*100:.3f}%) [Ratio vs true: {pred_frac/true_frac:.1f}x]")
    print(f"  Non-empty predictions: {non_empty}/50 ({non_empty/50*100:.1f}%)")
    print(f"  Mean IoU: {np.mean(ious):.4f}")
    print(f"  Mean Dice: {np.mean(dices):.4f}")

