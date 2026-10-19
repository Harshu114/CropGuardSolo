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
from model.training.losses import multitask_loss

ROOT = "/home/logiclocus/tmp/cgpack/cropguard_v7_split"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load 50 stratified samples
ds_full = CropGuardDataset(ROOT, split="train", augment=False)
with open("overfit_50_indices.txt") as f:
    indices = [int(line.strip()) for line in f if line.strip()]

ds_50 = Subset(ds_full, indices)
loader = DataLoader(ds_50, batch_size=10, shuffle=True, collate_fn=collate)
eval_loader = DataLoader(ds_50, batch_size=10, shuffle=False, collate_fn=collate)

# Model: pretrained backbone + upgraded high-resolution BilinearDecoder
model = CropGuardMultiTaskV7(nc=11, pretrained=True).to(device)
dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
for d in dropouts:
    d.p = 0.0

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("Starting 50-image overfit test (Upgraded High-Res Decoder + Focal-Tversky)...")
print(f"{'Epoch':<6} {'Loss_cls':<10} {'Loss_seg':<10} {'Sev_MAE':<10} {'Loss_tot':<10}")

epochs = 60
for ep in range(1, epochs + 1):
    model.train()
    ep_losses = {"cls": 0.0, "seg": 0.0, "total": 0.0}
    n_batches = 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        total_loss, parts = multitask_loss(
            out, batch, 320, nc=11, reg_max=8,
            train_sev=False, active_tasks={"cls", "seg"}, two_channel_seg=True,
            label_smoothing=0.0
        )
        total_loss.backward()
        optimizer.step()

        ep_losses["cls"] += parts.get("cls", 0.0).item()
        ep_losses["seg"] += parts.get("seg", 0.0).item()
        ep_losses["total"] += total_loss.item()
        n_batches += 1

    for k in ep_losses:
        ep_losses[k] /= n_batches

    if ep % 5 == 0 or ep == 1:
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

# Final detailed threshold evaluation
print("\n--- Final Evaluation on the same 50 images ---")
model.eval()

probs_all, gts_all, cls_preds, cls_gts, sev_preds, sev_gts = [], [], [], [], [], []
with torch.no_grad():
    for batch in eval_loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch["image"], with_analysis=True, active_tasks={"cls", "seg"})
        
        prob = torch.sigmoid(out["seg_logits"][:, 0]).cpu()
        probs_all.append(prob)
        gts_all.append(batch["mask"].cpu())
        
        pred_cls = out["cls_logits"].argmax(dim=-1).cpu().numpy()
        gt_cls = batch["cls"].cpu().numpy()
        valid_cls = gt_cls >= 0
        cls_preds.extend(pred_cls[valid_cls])
        cls_gts.extend(gt_cls[valid_cls])
        
        valid_sev = batch["sev_cont"] >= 0.0
        if valid_sev.any():
            sev_preds.extend(out["sev_cont"][valid_sev].cpu().numpy().tolist())
            sev_gts.extend(batch["sev_cont"][valid_sev].cpu().numpy().tolist())

probs = torch.cat(probs_all, dim=0)
gts = torch.cat(gts_all, dim=0)

cls_acc = np.mean(np.array(cls_preds) == np.array(cls_gts))
print(f"Classification accuracy: {cls_acc:.4f} (Bar: >= 0.95)")

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
    print(f"  Predicted lesion pixel fraction: {pred_frac:.5f} ({pred_frac*100:.3f}%)")
    print(f"  Non-empty predictions: {non_empty}/50 ({non_empty/50*100:.1f}%)")
    print(f"  Mean IoU: {np.mean(ious):.4f} (Pass bar: >= 0.50)")
    print(f"  Mean Dice: {np.mean(dices):.4f}")

sev_mae = np.mean(np.abs(np.array(sev_preds) - np.array(sev_gts)))
spearman_corr, _ = spearmanr(sev_preds, sev_gts)
print(f"\nSeverity MAE: {sev_mae:.4f} (Constant baseline MAE: ~0.038-0.040)")
print(f"Severity Spearman: {spearman_corr:.4f}")

