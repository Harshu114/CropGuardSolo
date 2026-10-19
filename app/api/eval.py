"""Evaluation: detection mAP@0.5 / mAP@0.5:0.95, classification Top-1 / macro-F1,
segmentation IoU / Dice, severity MAE / RMSE / R2 / Quadratic Weighted Kappa.

Standalone (no torchmetrics dependency) -- implements mAP via PR curves.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CropGuardDataset, collate
from model import CropGuardMultiTask
from losses import dfl_decode, box_xywh_to_xyxy

CLASS_NAMES = ["healthy", "bacterial_spot", "early_blight", "late_blight",
               "leaf_miner", "leaf_mold", "mosaic_virus",
               "septoria_leaf_spot", "spider_mites",
               "target_spot", "yellow_leaf_curl_virus"]


@torch.no_grad()
def nms(pred_xyxy, scores, iou_thr=0.55):
    """Simple NMS. pred_xyxy (N,4) pixels; scores (N,). Returns keep idx."""
    if pred_xyxy.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = pred_xyxy.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = torch.max(x1[i], x1[rest]); iy1 = torch.max(y1[i], y1[rest])
        ix2 = torch.min(x2[i], x2[rest]); iy2 = torch.min(y2[i], y2[rest])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)


@torch.no_grad()
def batch_decode(out, img_size, nc, conf_thr=0.05, max_det=100, reg_max=8):
    """Model outputs -> per-image detection lists [(boxes_xyxy_px, scores, labels)]."""
    B = out["det_cls"].shape[0]
    anchors = out["anchors"]
    boxes_xywh = dfl_decode(out["det_box"], anchors, img_size, reg_max)
    scores_all = out["det_cls"].sigmoid()
    res = []
    for b in range(B):
        sc, lb = scores_all[b].max(dim=-1)                       # A
        conf = sc  # objectness folded into cls
        m = conf > conf_thr
        if m.sum() == 0:
            res.append((torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.long)))
            continue
        bb = box_xywh_to_xyxy(boxes_xywh[b][m]) * img_size       # pixels
        ss, ll = conf[m], lb[m]
        keep = nms(bb, ss)
        keep = keep[:max_det]
        res.append((bb[keep], ss[keep], ll[keep]))
    return res


def ap_per_class(tp_by_conf, n_gt, n_conf=100):
    """tp_by_conf: list of (conf, iou_match_flag, cls). Returns AP@0.5 & AP@0.5:0.95."""
    if n_gt == 0 or not tp_by_conf:
        return np.nan, np.nan
    conf = np.array([t[0] for t in tp_by_conf])
    flag50 = np.array([t[1] for t in tp_by_conf], dtype=bool)
    flag5095 = np.array([t[2] for t in tp_by_conf], dtype=bool)
    order = np.argsort(-conf)
    conf, flag50, flag5095 = conf[order], flag50[order], flag5095[order]
    tp50 = np.cumsum(flag50)
    fp50 = np.cumsum(~flag50)
    tp95 = np.cumsum(flag5095)
    fp95 = np.cumsum(~flag5095)
    recall50 = tp50 / max(n_gt, 1)
    prec50 = tp50 / np.maximum(tp50 + fp50, 1)
    recall95 = tp95 / max(n_gt, 1)
    prec95 = tp95 / np.maximum(tp95 + fp95, 1)

    def ap(rec, prec):
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return ap(recall50, prec50), ap(recall95, prec95)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/cropguard-corpus-claude")
    ap.add_argument("--ckpt", default="/kaggle/working/runs/student_best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default="/kaggle/working/metrics.json")
    ap.add_argument("--nc", type=int, default=11,
                    help="number of disease classes; MUST match training")
    ap.add_argument("--reg-max", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropGuardMultiTask(nc=args.nc, pretrained=False,
                               reg_max=args.reg_max).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    ds = CropGuardDataset(args.root, args.split, args.size, augment=False,
                          max_images=args.max_images)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.workers,
                                         collate_fn=collate, pin_memory=True)

    # Accumulators for detection metrics across all IoU thresholds
    # COCO-style mAP@0.5:0.95 requires AP at 10 thresholds
    IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    # tp_by_iou[thr] = list of (conf, flag50, flag_iou)
    tp_by_iou = {t: [] for t in IOU_THRESHOLDS}
    n_gt = 0
    cls_correct = cls_total = 0
    cls_true, cls_pred = [], []
    inter_sum = union_sum = dice_inter_sum = dice_sum = 0.0
    sev_abs, sev_sq, sev_t_all, sev_p_all = [], [], [], []
    ord_true, ord_pred = [], []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(batch["image"], with_severity=True)
        img_size = batch["image"].shape[-1]
        dets = batch_decode(out, img_size, args.nc, conf_thr=args.conf)

        # ---------------- detection matching ----------------
        for b, (pb, ps, pl) in enumerate(dets):
            nb = int(batch["n_boxes"][b])
            gt = batch["det_boxes"][b][:nb].to(device)          # (n,5) cls,cx,cy,w,h
            gcls = gt[:, 0].long()
            gt_xyxy = box_xywh_to_xyxy(gt[:, 1:5]) * img_size
            matched = torch.zeros(nb, dtype=torch.bool, device=device)
            ious = torch.zeros(0, device=device)
            if pb.numel() and nb:
                from torchvision.ops import box_iou
                ious = box_iou(pb, gt_xyxy)                     # (P,G)
            for i in range(pb.shape[0]):
                if not nb:
                    for t in IOU_THRESHOLDS:
                        tp_by_iou[t].append((float(ps[i]), False, False))
                    continue
                j = int(ious[i].argmax())
                iou = float(ious[i, j])
                ok = bool(not bool(matched[j]) and int(gcls[j]) == int(pl[i]))
                if ok:
                    matched[j] = True
                for t in IOU_THRESHOLDS:
                    flag50 = bool(ok and iou >= 0.50)
                    flag_iou = bool(ok and iou >= t)
                    tp_by_iou[t].append((float(ps[i]), flag50, flag_iou))
            n_gt += nb

        # ---------------- classification ----------------
        cls = batch["cls"]
        valid = cls >= 0
        if valid.any():
            pred = out["cls_logits"][valid].argmax(-1)
            gold = cls[valid]
            cls_correct += int((pred == gold).sum())
            cls_total += int(valid.sum())
            cls_true += gold.tolist()
            cls_pred += pred.tolist()

        # ---------------- segmentation ----------------
        prob = out["seg_logits"].sigmoid()          # (B,1,H,W)
        pred_m = (prob > 0.5).float()
        gt_m = batch["mask"][:, None]               # (B,1,H,W)
        inter_sum += float((pred_m * gt_m).sum())
        union_sum += float(((pred_m + gt_m) > 0).float().sum())
        d_inter = (pred_m * gt_m).sum(dim=(1, 2, 3))
        d_sum = pred_m.sum(dim=(1, 2, 3)) + gt_m.sum(dim=(1, 2, 3))
        dice_inter_sum += float(d_inter.sum())
        dice_sum += float(d_sum.sum())

        # ---------------- severity ----------------
        sev_cont, sev_ord = out["sev_cont"].float(), out["sev_ord"].argmax(-1)
        sev_t, ord_t = batch["sev_cont"], batch["sev_ord"]
        sev_abs += (sev_cont - sev_t).abs().tolist()
        sev_sq += ((sev_cont - sev_t) ** 2).tolist()
        sev_t_all += sev_t.tolist()
        sev_p_all += sev_cont.tolist()
        ord_true += ord_t.tolist()
        ord_pred += sev_ord.tolist()

    # ------------- aggregate -------------
    # Compute AP at each IoU threshold for COCO-style mAP@0.5:0.95
    aps = {}
    for t in IOU_THRESHOLDS:
        ap50, ap = ap_per_class(tp_by_iou[t], n_gt)
        aps[t] = ap
    mAP50 = aps[0.50]
    mAP50_95 = sum(aps.values()) / len(aps)  # COCO-style mean AP across 10 thresholds
    acc = cls_correct / max(cls_total, 1)
    macro_f1 = macro_f1_score(cls_true, cls_pred, k=args.nc)
    iou = inter_sum / max(union_sum, 1)
    dice = 2 * dice_inter_sum / max(dice_sum, 1)
    mae = float(np.mean(sev_abs)) if sev_abs else float("nan")
    rmse = float(np.sqrt(np.mean(sev_sq))) if sev_sq else float("nan")
    r2 = r2_score(np.array(sev_t_all), np.array(sev_p_all))
    qwk = quadratic_weighted_kappa(np.array(ord_true), np.array(ord_pred), k=4)

    metrics = {
        "split": args.split, "n_images": len(ds),
        "mAP@0.5": mAP50, "mAP@0.5:0.95": mAP50_95,
        "cls_top1": acc, "cls_macro_f1": macro_f1,
        "seg_mIoU": iou, "seg_dice": dice,
        "sev_mae": mae, "sev_rmse": rmse, "sev_r2": r2, "sev_qwk": qwk,
        "paper_targets": {"mAP@0.5": 0.892, "mAP@0.5:0.95": 0.624,
                          "cls_top1": 0.987, "seg_mIoU": 0.863,
                          "seg_dice": 0.912, "sev_mae": 0.042},
    }
    print(json.dumps(metrics, indent=2))
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)


def ap_per_class_global(tp_list, n_gt):
    return ap_per_class(tp_list, n_gt)


def macro_f1_score(y_true, y_pred, k):
    from collections import Counter
    if not y_true:
        return float("nan")
    f1s = []
    for c in range(k):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f1s))


def r2_score(y, yhat):
    if len(y) < 2:
        return float("nan")
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / max(ss_tot, 1e-9)


def quadratic_weighted_kappa(y_true, y_pred, k):
    if len(y_true) == 0:
        return float("nan")
    O = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        O[min(int(t), k - 1), min(int(p), k - 1)] += 1
    O = O / O.sum()
    w = np.array([[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)])
    t_hist = O.sum(1)
    p_hist = O.sum(0)
    E = np.outer(t_hist, p_hist)
    num = (w * O).sum()
    den = (w * E).sum()
    return float(1 - num / max(den, 1e-9))


if __name__ == "__main__":
    main()
