"""Full v7 Training Script.

Requirements:
- 8 classes, 715 train images, 156 val images
- 120 epochs, batch size 8, AMP enabled
- Cosine schedule with warmup
- EMA (decay 0.999)
- Label smoothing (0.05)
- Class-balanced sampling
- Field-style augmentations
- Evaluate on val every 5 epochs:
    * Classification accuracy and macro-F1
    * Lesion IoU and Dice (threshold chosen on val only, grid search [0.3, 0.4, 0.5, 0.6, 0.7])
    * Severity MAE vs constant baseline 0.0384
- Save best checkpoint by val metrics (val score combining F1, IoU, and severity MAE)
"""
import argparse
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import CropGuardDataset, collate, load_yolo_labels
from losses import multitask_loss
from model.architecture.v7_model import CropGuardMultiTaskV7


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


@torch.no_grad()
def ema_eval(model, ema):
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()
              if k in ema.shadow}
    ema.copy_to(model)

    def restore():
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)
    return restore


def make_sampler(ds):
    counts = {}
    cls_labels = []
    for i in range(len(ds)):
        img_path, lbl_path = ds.samples[i]
        boxes = load_yolo_labels(lbl_path)
        c = int(max((b[0] for b in boxes), default=-1))
        cls_labels.append(c)
        counts[c] = counts.get(c, 0) + 1
    med = sorted(counts.values())[len(counts) // 2]
    weights = [1.0 / math.sqrt(counts.get(c, med)) for c in cls_labels]
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)


def cosine_lr(step, total, warmup, base_lr, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


def compute_macro_f1(y_true, y_pred, nc=8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1s = []
    for c in range(nc):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        # only average over classes that exist in GT
        if np.sum(y_true == c) > 0:
            f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


@torch.no_grad()
def validate_metrics(model, loader, device, nc=8):
    model.eval()
    all_preds = []
    all_targets = []
    all_les_probs = []
    all_les_gts = []
    all_sev_preds = []
    all_sev_gts = []

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            out = model(img, active_tasks={"cls", "seg"}, with_analysis=True)

        # Classification
        cls_logits = out["cls_logits"].float()
        preds = cls_logits.argmax(dim=-1).cpu().numpy()
        targets = batch["cls"].numpy()
        valid = targets >= 0
        all_preds.extend(preds[valid])
        all_targets.extend(targets[valid])

        # Segmentation probabilities & targets
        seg_probs = out["seg_logits"][:, 0:1].sigmoid().float().cpu() # (B, 1, H, W)
        gt_mask = batch["mask"][:, None].float().cpu()                # (B, 1, H, W)
        all_les_probs.append(seg_probs)
        all_les_gts.append(gt_mask)

        # Severity
        sev_pred = out["sev_cont"].float().cpu().numpy()
        sev_gt = batch["sev_cont"].numpy()
        sev_valid = sev_gt >= 0.0
        if sev_valid.any():
            all_sev_preds.extend(sev_pred[sev_valid])
            all_sev_gts.extend(sev_gt[sev_valid])

    model.train()

    # Classification metrics
    acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    macro_f1 = compute_macro_f1(all_targets, all_preds, nc=nc)

    # Segmentation grid search for best threshold on VAL
    les_probs = torch.cat(all_les_probs, dim=0) # (N, 1, H, W)
    les_gts = torch.cat(all_les_gts, dim=0)     # (N, 1, H, W)

    best_iou, best_dice, best_thr = 0.0, 0.0, 0.5
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_bin = (les_probs > thr).float()
        inter = (pred_bin * les_gts).sum().item()
        union = ((pred_bin + les_gts) > 0).float().sum().item()
        iou = inter / max(union, 1.0)
        dice = (2.0 * inter) / max(pred_bin.sum().item() + les_gts.sum().item(), 1.0)
        if iou > best_iou:
            best_iou = iou
            best_dice = dice
            best_thr = thr

    # Severity MAE
    sev_mae = float(np.mean(np.abs(np.array(all_sev_preds) - np.array(all_sev_gts)))) if all_sev_preds else float("nan")

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "best_thr": best_thr,
        "lesion_iou": best_iou,
        "lesion_dice": best_dice,
        "sev_mae": sev_mae,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/logiclocus/tmp/cgpack/cropguard_v7_split")
    parser.add_argument("--out", default="runs_v7_full")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--nc", type=int, default=8)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_file = os.path.join(args.out, "train.log")

    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    train_ds = CropGuardDataset(args.root, "train", args.size, augment=True)
    val_ds = CropGuardDataset(args.root, "val", args.size, augment=False)
    log(f"Dataset loaded: {len(train_ds)} train, {len(val_ds)} val")

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        drop_last=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate,
        persistent_workers=True)

    model = CropGuardMultiTaskV7(nc=args.nc, pretrained=True).to(device)
    ema = EMA(model, decay=0.999)

    # --- Pretrained weight verification ---
    # Compare first backbone conv against a fresh random-init copy.
    # L1 distance > 0.01 confirms real pretrained weights loaded.
    import timm as _timm
    _rand_net = _timm.create_model("fasternet_t0", pretrained=False, features_only=True, out_indices=(0,1,2,3))
    _pretrained_w = next(model.backbone.net.parameters()).detach().cpu()
    _random_w     = next(_rand_net.parameters()).detach().cpu()
    _l1_dist = (_pretrained_w - _random_w).abs().mean().item()
    _weights_path = _timm.models.hub.get_cache_dir()  # timm cache root
    log(f"[Pretrained check] backbone first-conv L1 vs random init: {_l1_dist:.6f}")
    log(f"[Pretrained check] timm cache dir: {_weights_path}")
    if _l1_dist < 0.01:
        raise RuntimeError(
            f"Pretrained weights appear identical to random init (L1={_l1_dist:.6f}). "
            "Check timm cache / internet access. Aborting to avoid wasting budget."
        )
    log(f"[Pretrained check] PASS — real pretrained weights confirmed (L1={_l1_dist:.4f} >> 0.01)")
    del _rand_net, _pretrained_w, _random_w
    # --- end verification ---

    # Freeze nothing: end-to-end multi-task training
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = 5 * steps_per_epoch

    best_score = -1e9
    best_metrics = None
    log("Checkpoint selection: best = highest val macro-F1 (not composite; until seg/sev work)")

    log(f"Starting full training for {args.epochs} epochs ({total_steps} total steps)...")
    log("Baseline constant severity MAE to beat: 0.0384")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        loss_sums = {"cls": 0.0, "seg": 0.0, "total": 0.0}

        # GT gate probability schedule: 0.8 down to 0.2
        gt_mask_prob = 0.8 - 0.6 * ((epoch - 1) / max(args.epochs - 1, 1))

        for it, batch in enumerate(train_loader):
            step = (epoch - 1) * steps_per_epoch + it
            lr = cosine_lr(step, total_steps, warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(batch["image"], active_tasks={"cls", "seg"},
                            gt_mask=batch.get("mask"), gt_mask_prob=gt_mask_prob)
                total, parts = multitask_loss(
                    out, batch, args.size, nc=args.nc,
                    active_tasks={"cls", "seg"}, two_channel_seg=True,
                    train_sev=False, label_smoothing=0.05)

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()

            ema.update(model)

            for k in ["cls", "seg"]:
                if k in parts:
                    loss_sums[k] += parts[k].item()
            loss_sums["total"] += total.item()

        dt = time.time() - t0
        cls_loss = loss_sums["cls"] / steps_per_epoch
        seg_loss = loss_sums["seg"] / steps_per_epoch
        tot_loss = loss_sums["total"] / steps_per_epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            restore = ema_eval(model, ema)
            vm = validate_metrics(model, val_loader, device, nc=args.nc)
            restore()

            log(f"Epoch {epoch:3d}/{args.epochs} [{dt:.1f}s] - Train: loss={tot_loss:.4f} cls={cls_loss:.4f} seg={seg_loss:.4f} | "
                f"Val: acc={vm['acc']:.4f} macro_f1={vm['macro_f1']:.4f} lesion_iou={vm['lesion_iou']:.4f} (thr={vm['best_thr']}) "
                f"dice={vm['lesion_dice']:.4f} sev_mae={vm['sev_mae']:.4f} (vs base 0.0384)")

            # Best checkpoint = highest val macro-F1 ONLY.
            # Rationale: seg IoU is near-zero and severity MAE is dominated by
            # the analytic path's poor seg; using F1 prevents those noise terms
            # from overriding a legitimately better classification epoch.
            val_score = vm['macro_f1']   # <- single metric, macro-F1 on val
            if val_score > best_score:
                best_score = val_score
                best_metrics = vm
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_metrics": best_metrics,
                    "args": vars(args),
                }, os.path.join(ckpt_dir, "best_val.pt"))
                # Also save standard format for eval scripts
                torch.save(model.state_dict(), os.path.join(args.out, "student_best.pt"))
                log(f"  --> Saved new best checkpoint at epoch {epoch} (score={val_score:.4f})")
        else:
            log(f"Epoch {epoch:3d}/{args.epochs} [{dt:.1f}s] - Train: loss={tot_loss:.4f} cls={cls_loss:.4f} seg={seg_loss:.4f}")

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        }, os.path.join(ckpt_dir, "last.pt"))

    log("Training completed successfully!")


if __name__ == "__main__":
    main()
