"""Items 2-4: Healthy source + shortcut check, DataLoader alignment, 8-image memorization test.
Run: PYTHONPATH=.:model/training python3 items_2_audit.py
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'model', 'training'))
sys.path.insert(0, _ROOT)

from dataset import CropGuardDataset, collate
from model.architecture.v7_model import CropGuardMultiTaskV7
from model.training.losses import multitask_loss

DATA = '/home/logiclocus/tmp/cgpack/cropguard_v7_split'

# =============================================================================
# ITEM 2: Healthy source + shortcut check
# =============================================================================
print("\n" + "="*70)
print("ITEM 2: Healthy source + shortcut check")
print("="*70)

train_ds = CropGuardDataset(DATA, 'train', 320, augment=False)
val_ds   = CropGuardDataset(DATA, 'val',   320, augment=False)

# Separate healthy (class 0) vs diseased
from dataset import load_yolo_labels

def get_class(lbl_path):
    boxes = load_yolo_labels(lbl_path)
    return int(max((b[0] for b in boxes), default=-1))

def stats(ds, name):
    h_widths, h_heights, h_means, h_sizes = [], [], [], []
    d_widths, d_heights, d_means, d_sizes = [], [], [], []
    for i in range(len(ds)):
        img_path, lbl_path = ds.samples[i]
        c = get_class(lbl_path)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        w, h = img.size
        size_bytes = os.path.getsize(img_path)
        mean_rgb = np.array(img).mean(axis=(0,1))
        if c == 0:
            h_widths.append(w); h_heights.append(h)
            h_means.append(mean_rgb); h_sizes.append(size_bytes)
        else:
            d_widths.append(w); d_heights.append(h)
            d_means.append(mean_rgb); d_sizes.append(size_bytes)
    return h_widths, h_heights, h_means, h_sizes, d_widths, d_heights, d_means, d_sizes

hw, hh, hm, hs, dw, dh, dm, ds_ = stats(train_ds, 'train')
print(f"\nTRAIN healthy ({len(hw)} images): mean_w={np.mean(hw):.1f} mean_h={np.mean(hh):.1f} mean_size={np.mean(hs)/1024:.1f}KB mean_rgb=[{np.mean(hm,axis=0)}]")
print(f"TRAIN diseased ({len(dw)} images): mean_w={np.mean(dw):.1f} mean_h={np.mean(dh):.1f} mean_size={np.mean(ds_)/1024:.1f}KB mean_rgb=[{np.mean(dm,axis=0)}]")

# Print source dirs for healthy images
healthy_dirs = set()
for i in range(len(train_ds)):
    img_path, lbl_path = train_ds.samples[i]
    c = get_class(lbl_path)
    if c == 0:
        healthy_dirs.add(os.path.dirname(img_path))
print(f"\nHealthy image source directories ({len(healthy_dirs)} distinct):")
for d in sorted(healthy_dirs):
    n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])
    print(f"  {d}  ({n} files)")

# Logistic regression shortcut check on val
print("\n--- Logistic regression shortcut on val ---")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def extract_features(ds):
    X, y = [], []
    for i in range(len(ds)):
        img_path, lbl_path = ds.samples[i]
        c = get_class(lbl_path)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            continue
        w, h = img.size
        size_bytes = os.path.getsize(img_path)
        mean_rgb = np.array(img).mean(axis=(0,1))
        feat = [w, h, size_bytes, mean_rgb[0], mean_rgb[1], mean_rgb[2]]
        X.append(feat)
        y.append(1 if c == 0 else 0)
    return np.array(X, dtype=np.float32), np.array(y)

X_train, y_train = extract_features(train_ds)
X_val,   y_val   = extract_features(val_ds)
print(f"Train features: {X_train.shape}, val features: {X_val.shape}")
print(f"Train healthy fraction: {y_train.mean():.3f}, val healthy fraction: {y_val.mean():.3f}")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(X_train_s, y_train)
val_preds = clf.predict(X_val_s)
val_acc = accuracy_score(y_val, val_preds)
print(f"Logistic regression val accuracy (size+colour features): {val_acc:.4f} ({val_acc*100:.2f}%)")
if val_acc > 0.85:
    print("  WARNING: > 85% separable — healthy images have different style from diseased.")
else:
    print("  OK: < 85% — healthy images are stylistically similar to diseased.")

# =============================================================================
# ITEM 3: DataLoader alignment overlays (CPU — GPU is training)
# =============================================================================
print("\n" + "="*70)
print("ITEM 3: DataLoader alignment overlays (8 batches, CPU)")
print("="*70)

val_ds_aug = CropGuardDataset(DATA, 'val', 320, augment=False)
loader = DataLoader(val_ds_aug, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)

os.makedirs('alignment_check', exist_ok=True)
MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

non_empty_lesion = 0
misalign_flags = []
for batch_idx, batch in enumerate(loader):
    if batch_idx >= 8:
        break
    imgs  = batch['image']       # (B,3,H,W)
    masks = batch['mask']        # (B,H,W) lesion
    leaves = batch['leaf']       # (B,H,W) leaf

    B = imgs.shape[0]
    fig, axes = plt.subplots(B, 3, figsize=(9, 3*B))
    if B == 1:
        axes = axes[None]

    for b in range(B):
        img_t = imgs[b].permute(1,2,0) * STD + MEAN
        img_np = img_t.clamp(0,1).numpy()

        les = masks[b].numpy()
        lef = leaves[b].numpy()

        # check lesion pixels inside leaf
        les_px = (les > 0.5).astype(float)
        lef_px = (lef > 0.5).astype(float)
        if les_px.sum() > 0 and lef_px.sum() > 0:
            frac_inside = (les_px * lef_px).sum() / les_px.sum()
            misalign_flags.append(frac_inside)
            non_empty_lesion += 1
        elif les_px.sum() == 0:
            misalign_flags.append(float('nan'))  # healthy: no lesion expected

        axes[b, 0].imshow(img_np); axes[b, 0].set_title(f'Img b{batch_idx}_{b}')
        overlay_les = img_np.copy()
        overlay_les[les > 0.5] = [1.0, 0.0, 0.0]
        axes[b, 1].imshow(overlay_les); axes[b, 1].set_title('Lesion overlay')
        overlay_lef = img_np.copy()
        overlay_lef[lef > 0.5, 1] = np.clip(overlay_lef[lef > 0.5, 1] + 0.4, 0, 1)
        axes[b, 2].imshow(overlay_lef); axes[b, 2].set_title('Leaf overlay')

    plt.tight_layout()
    plt.savefig(f'alignment_check/batch_{batch_idx:02d}.png', dpi=72)
    plt.close()
    print(f"  Batch {batch_idx}: saved alignment_check/batch_{batch_idx:02d}.png")

valid_flags = [f for f in misalign_flags if not math.isnan(f)]
print(f"\nAlignment check — lesion pixels inside leaf:")
print(f"  Samples with non-empty lesion: {non_empty_lesion}")
print(f"  Mean frac of lesion px inside leaf: {np.mean(valid_flags):.4f}")
print(f"  Min: {np.min(valid_flags):.4f}  Max: {np.max(valid_flags):.4f}")
if np.min(valid_flags) >= 0.99:
    print("  PASS: all lesion pixels lie inside leaf mask (no geometric misalignment)")
else:
    bad = [(i, f) for i,f in enumerate(valid_flags) if f < 0.99]
    print(f"  WARNING: {len(bad)} samples with lesion-outside-leaf fractions")

# =============================================================================
# ITEM 4: 8-image memorization test (500 steps)
# =============================================================================
print("\n" + "="*70)
print("ITEM 4: 8-image memorization test (500 steps, lr heads=1e-3 backbone=1e-4)")
print("="*70)

device = torch.device('cpu')  # GPU may be busy finishing training
model = CropGuardMultiTaskV7(nc=8, pretrained=False).to(device)
model.train()

# Sample 8 diverse images from val (avoid healthy-only to have non-zero lesion masks)
indices = []
class_seen = set()
for i in range(len(val_ds)):
    _, lbl_path = val_ds.samples[i]
    c = get_class(lbl_path)
    if c not in class_seen:
        indices.append(i)
        class_seen.add(c)
    if len(indices) >= 8:
        break
# If fewer than 8 distinct classes, pad from beginning
if len(indices) < 8:
    for i in range(len(val_ds)):
        if i not in indices:
            indices.append(i)
        if len(indices) >= 8:
            break

subset = torch.utils.data.Subset(val_ds, indices)
mem_loader = DataLoader(subset, batch_size=8, shuffle=False, collate_fn=collate)
batch = next(iter(mem_loader))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
imgs  = batch['image']
masks = batch['mask']

print(f"  8 images selected, classes: {[get_class(val_ds.samples[i][1]) for i in indices]}")
les_fracs = [(masks[b] > 0.5).float().mean().item() for b in range(len(indices))]
les_fracs_str = [f'{v:.4f}' for v in les_fracs]
print(f"  Lesion pixel fractions: {les_fracs_str}")

# Separate param groups: backbone 1e-4, heads 1e-3
backbone_params = list(model.backbone.parameters()) + list(model.neck.parameters())
head_params = [p for p in model.parameters()
               if not any(p is q for q in backbone_params)]
opt = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-4},
    {'params': head_params,     'lr': 1e-3},
], weight_decay=1e-4)

print(f"\n{'Step':>6} {'seg_loss':>10} {'cls_loss':>10} {'IoU@best':>10} {'Dice@best':>10} {'non_empty':>10}")
for step in range(1, 501):
    opt.zero_grad(set_to_none=True)
    out = model(imgs, active_tasks={'cls', 'seg'})
    total, parts = multitask_loss(out, batch, 320, nc=8, active_tasks={'cls', 'seg'},
                                  two_channel_seg=True, train_sev=False, label_smoothing=0.05)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()

    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            out_e = model(imgs, active_tasks={'cls', 'seg'})
        model.train()
        probs = out_e['seg_logits'][:, 0:1].sigmoid()
        les_gt = masks[:, None].float()
        best_iou = 0.0; best_dice = 0.0; best_thr = 0.5; best_ne = 0
        for thr in [0.3, 0.5, 0.7]:
            pred_bin = (probs > thr).float()
            ne = int((pred_bin.sum(dim=(1,2,3)) > 0).sum().item())
            inter = (pred_bin * les_gt).sum().item()
            union = ((pred_bin + les_gt) > 0).float().sum().item()
            iou = inter / max(union, 1.0)
            dice = 2*inter / max(pred_bin.sum().item() + les_gt.sum().item(), 1.0)
            if iou > best_iou:
                best_iou = iou; best_dice = dice; best_thr = thr; best_ne = ne
        seg_l = parts.get('seg', torch.tensor(0.0)).item()
        cls_l = parts.get('cls', torch.tensor(0.0)).item()
        print(f"  {step:>4}  {seg_l:>10.4f}  {cls_l:>10.4f}  {best_iou:>10.4f}  {best_dice:>10.4f}  {best_ne:>8}/{len(indices)}")

print("\nItems 2-4 complete.")
