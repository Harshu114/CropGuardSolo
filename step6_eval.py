"""Step 6 evaluation on best-by-val checkpoint.
Reports val metrics first, then test metrics. Thresholds tuned on val only.

Run: PYTHONPATH=.:model/training python3 step6_eval.py 2>&1 | tee step6_eval.log
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'model', 'training'))
sys.path.insert(0, _ROOT)

from dataset import CropGuardDataset, collate, load_yolo_labels
from model.architecture.v7_model import CropGuardMultiTaskV7

CLASS_NAMES = ["healthy","bacterial_spot","early_blight","late_blight",
               "leaf_mold","mosaic_virus","septoria_leaf_spot","yellow_leaf_curl_virus"]
NC = 8
DATA   = '/home/logiclocus/tmp/cgpack/cropguard_v7_split'
CKPT   = 'runs_v7_120/checkpoints/best_val.pt'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def macro_f1(y_true, y_pred, nc=NC):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    f1s = []
    for c in range(nc):
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        denom = 2*tp+fp+fn
        if np.sum(y_true==c) > 0:
            f1s.append((2*tp/denom) if denom>0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0

def per_class_f1(y_true, y_pred, nc=NC):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    out = {}
    for c in range(nc):
        tp = np.sum((y_true==c)&(y_pred==c))
        fp = np.sum((y_true!=c)&(y_pred==c))
        fn = np.sum((y_true==c)&(y_pred!=c))
        denom = 2*tp+fp+fn
        n = int(np.sum(y_true==c))
        out[CLASS_NAMES[c]] = {'f1': float((2*tp/denom) if denom>0 else 0.0), 'n': n}
    return out

def ece(confs, corrects, n_bins=10):
    confs, corrects = np.array(confs), np.array(corrects)
    bins = np.linspace(0, 1, n_bins+1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        if mask.sum() > 0:
            ece_val += mask.sum()/len(confs) * abs(corrects[mask].mean() - confs[mask].mean())
    return float(ece_val)

def confusion_matrix(y_true, y_pred, nc=NC):
    cm = np.zeros((nc, nc), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

@torch.no_grad()
def evaluate(model, loader, split_name, thr=None):
    """Evaluate on a split. thr=None -> tune threshold on this split (only for val)."""
    model.eval()
    all_preds, all_targets = [], []
    all_confs, all_corrects = [], []
    all_seg_probs, all_seg_gts = [], []
    all_sev_pred, all_sev_gt = [], []
    TEMP = float(model.abstain.temperature.item())

    for batch in loader:
        batch = {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast('cuda', enabled=(DEVICE.type=='cuda')):
            out = model(batch['image'], active_tasks={'cls','seg'}, with_analysis=True)

        # Classification
        logits = out['cls_logits'].float()
        preds  = logits.argmax(-1).cpu().numpy()
        targets= batch['cls'].cpu().numpy()
        valid  = targets >= 0
        all_preds.extend(preds[valid])
        all_targets.extend(targets[valid])
        scaled_probs = (logits[valid] / TEMP).softmax(-1)
        confs  = scaled_probs.max(-1).values.cpu().numpy()
        all_confs.extend(confs)
        all_corrects.extend((preds[valid] == targets[valid]).astype(float))

        # Segmentation (lesion channel = 0)
        seg_probs = out['seg_logits'][:, 0:1].sigmoid().float().cpu()
        seg_gt    = batch['mask'][:, None].float().cpu()
        all_seg_probs.append(seg_probs)
        all_seg_gts.append(seg_gt)

        # Severity
        sev_pred = out['sev_cont'].float().cpu().numpy()
        sev_gt   = batch['sev_cont'].cpu().numpy()
        sv = sev_gt >= 0.0
        all_sev_pred.extend(sev_pred[sv])
        all_sev_gt.extend(sev_gt[sv])

    # Classification metrics
    acc      = float(np.mean(np.array(all_preds)==np.array(all_targets)))
    mf1      = macro_f1(all_targets, all_preds)
    eceval   = ece(all_confs, all_corrects)
    pcf1     = per_class_f1(all_targets, all_preds)
    cm       = confusion_matrix(all_targets, all_preds)

    # Segmentation: tune threshold on val only
    seg_probs_cat = torch.cat(all_seg_probs)
    seg_gts_cat   = torch.cat(all_seg_gts)
    if thr is None:
        best_iou, best_dice, best_thr = 0.0, 0.0, 0.5
        for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pred_bin = (seg_probs_cat > t).float()
            inter = (pred_bin * seg_gts_cat).sum().item()
            union = ((pred_bin + seg_gts_cat) > 0).float().sum().item()
            iou   = inter / max(union, 1.0)
            dice  = 2*inter / max(pred_bin.sum().item()+seg_gts_cat.sum().item(), 1.0)
            if iou > best_iou:
                best_iou, best_dice, best_thr = iou, dice, t
    else:
        best_thr = thr
        pred_bin  = (seg_probs_cat > best_thr).float()
        inter     = (pred_bin * seg_gts_cat).sum().item()
        union     = ((pred_bin + seg_gts_cat) > 0).float().sum().item()
        best_iou  = inter / max(union, 1.0)
        best_dice = 2*inter / max(pred_bin.sum().item()+seg_gts_cat.sum().item(), 1.0)

    # Severity
    sev_arr_pred = np.array(all_sev_pred)
    sev_arr_gt   = np.array(all_sev_gt)
    sev_mae      = float(np.mean(np.abs(sev_arr_pred - sev_arr_gt))) if len(sev_arr_gt)>0 else float('nan')
    sev_spear    = float(spearmanr(sev_arr_gt, sev_arr_pred).correlation) if len(sev_arr_gt)>1 else float('nan')
    const_mae    = float(np.mean(np.abs(np.zeros_like(sev_arr_gt) + np.median(sev_arr_gt) - sev_arr_gt))) if len(sev_arr_gt)>0 else float('nan')

    print(f"\n{'='*60}")
    print(f"SPLIT: {split_name} (threshold tuned on VAL = {best_thr})")
    print(f"{'='*60}")
    print(f"  N images evaluated:    {len(all_targets)}")
    print(f"  Classification top-1:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro-F1:              {mf1:.4f}")
    print(f"  ECE:                   {eceval:.4f}")
    print(f"  Lesion IoU @ {best_thr}:  {best_iou:.4f}")
    print(f"  Lesion Dice @ {best_thr}: {best_dice:.4f}")
    print(f"  Severity MAE:          {sev_mae:.4f}  (constant-baseline MAE: 0.0384 val / 0.0409 test)")
    print(f"  Severity Spearman:     {sev_spear:.4f}")
    print(f"\n  Per-class F1:")
    for name, v in pcf1.items():
        bar = '#'*int(v['f1']*20)
        print(f"    {name:30s}  n={v['n']:4d}  F1={v['f1']:.3f}  {bar}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " ".join(f"{i:4d}" for i in range(NC))
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{v:4d}" for v in row)
        print(f"  {i}  {row_str}")

    return {
        'split': split_name, 'n': len(all_targets),
        'acc': acc, 'macro_f1': mf1, 'ece': eceval,
        'lesion_iou': best_iou, 'lesion_dice': best_dice,
        'best_thr': best_thr,
        'sev_mae': sev_mae, 'sev_spear': sev_spear,
        'const_mae_baseline': 0.0384,
        'per_class_f1': pcf1,
    }, best_thr


def main():
    print(f"Loading checkpoint: {CKPT}")
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    sd = ck.get('model', ck)
    model = CropGuardMultiTaskV7(nc=NC, pretrained=False).to(DEVICE)
    model.load_state_dict(sd, strict=True)
    model.eval()
    TEMP = float(model.abstain.temperature.item())
    print(f"Loaded. Temperature = {TEMP:.4f}. Best at epoch = {ck.get('epoch','?')}")

    # ---- VAL first (tune threshold here) ----
    val_ds = CropGuardDataset(DATA, 'val', 320, augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False,
                            collate_fn=collate, num_workers=2, pin_memory=True)
    val_metrics, val_thr = evaluate(model, val_loader, 'VAL', thr=None)

    # ---- TEST (apply val threshold — do not re-tune) ----
    test_ds = CropGuardDataset(DATA, 'test', 320, augment=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             collate_fn=collate, num_workers=2, pin_memory=True)
    test_metrics, _ = evaluate(model, test_loader, 'TEST (PlantSeg)', thr=val_thr)

    # ---- PlantDoc classification-only ----
    print(f"\n{'='*60}")
    print("PlantDoc TEST (79 tomato images, classification only)")
    print("  [Run eval_plantdoc.py separately for detailed PlantDoc metrics]")
    print(f"{'='*60}")

    # Save results
    out = {'val': val_metrics, 'test': test_metrics}
    with open('step6_metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nMetrics saved to step6_metrics.json")

if __name__ == '__main__':
    main()
