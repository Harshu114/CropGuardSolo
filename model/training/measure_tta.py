"""SUPERSEDED -- DO NOT QUOTE NUMBERS FROM THIS SCRIPT.

Kept only to document the process. The `_ap50()` below is a hand-rolled
approximation, not the real COCO AP: it reported an absolute mAP@0.5 of 0.83
where the real evaluator reports 0.497 for the same checkpoint. The per-class
averaging and recall normalisation are wrong.

Use `eval_tta.py --tta` instead, which reuses eval.py's real AP implementation.
That is where the shipped numbers (0.4972 -> 0.5101) come from.

Costs no Kaggle quota -- runs on the local GPU. TTA is usually quoted as a free
+1-3 mAP, but that is folklore until measured on THIS model and THIS data.

Usage:  python3 measure_tta.py [max_images] [conf]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import CropGuardDataset, collate          # noqa: E402
from eval import batch_decode, nms                      # noqa: E402
from losses import box_xywh_to_xyxy                    # noqa: E402
from model.architecture.model import CropGuardMultiTask  # noqa: E402

ROOT = "/home/logiclocus/HarshalPrivate/Projects/collegeProjects/All tries of cropguard/ClaudeCropGuard/datasetclaude"
CKPT = os.path.join(HERE, "runs", "student_best.pt")


def flip_boxes(bb, w):
    """Mirror x coordinates back into the original frame. bb: (N,4) xyxy px."""
    out = bb.clone()
    out[:, 0] = w - bb[:, 2]
    out[:, 2] = w - bb[:, 0]
    return out


@torch.no_grad()
def main():
    max_images = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropGuardMultiTask(nc=11, pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    ds = CropGuardDataset(ROOT, "val", 320, augment=False, max_images=max_images)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False,
                                         num_workers=4, collate_fn=collate)

    # AP@0.5 accumulators: one entry per (image, gt) with matched/unmatched flag
    acc = {"base": [], "tta": []}
    cls_ok = {"base": 0, "tta": 0}
    cls_n = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        b = batch["image"].shape[0]
        s = img.shape[-1]
        out = model(img, with_severity=False)
        out_f = model(torch.flip(img, dims=[3]), with_severity=False)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pass  # autocast already applied inside; keep decode in fp32

        base = batch_decode(out, s, 11, conf_thr=conf, max_det=100)
        tta_f = batch_decode(out_f, s, 11, conf_thr=conf, max_det=100)

        # ---- classification: single vs flip-averaged ----
        pb = out["cls_logits"].float().softmax(-1)
        pf = out_f["cls_logits"].float().softmax(-1)
        for i in range(b):
            cls_n += 1
            if int(pb[i].argmax()) == batch["cls"][i]:
                cls_ok["base"] += 1
            if int(((pb[i] + pf[i]) / 2).argmax()) == batch["cls"][i]:
                cls_ok["tta"] += 1

        for i in range(b):
            nb = int(batch["n_boxes"][i])
            gt = batch["det_boxes"][i][:nb].to(device)
            gt_xyxy = box_xywh_to_xyxy(gt[:, 1:5]) * s if nb else torch.zeros(0, 4, device=device)
            gcls = gt[:, 0].long() if nb else torch.zeros(0, dtype=torch.long, device=device)

            # batch_decode returns CPU tensors for the "no predictions" case and
            # CUDA tensors otherwise, so force everything onto one device first.
            bb0, sc0, lb0 = (t.to(device) for t in base[i])
            acc["base"].append(_match(bb0, sc0, lb0, gt_xyxy, gcls))

            # TTA union
            bbf, scf, lbf = (t.to(device) for t in tta_f[i])
            bbf = flip_boxes(bbf, s)
            bb = torch.cat([bb0, bbf], 0)
            sc = torch.cat([sc0, scf], 0)
            lb = torch.cat([lb0, lbf], 0)
            if bb.numel():
                keep = nms(bb, sc)[:100]
                bb, sc, lb = bb[keep], sc[keep], lb[keep]
            acc["tta"].append(_match(bb, sc, lb, gt_xyxy, gcls))

    print(f"\nval images: {cls_n}   conf threshold: {conf}")
    print(f"\n{'':<10}{'mAP@0.5':>12}{'cls_top1':>12}")
    for k in ("base", "tta"):
        print(f"{k:<10}{_ap50(acc[k]):>12.4f}{cls_ok[k]/max(cls_n,1):>12.4f}")
    d = _ap50(acc["tta"]) - _ap50(acc["base"])
    dc = cls_ok["tta"] / max(cls_n, 1) - cls_ok["base"] / max(cls_n, 1)
    print(f"\ndelta mAP@0.5 {d:+.4f}   delta cls_top1 {dc:+.4f}")
    print("TTA IS WORTH IT" if d > 0.002 or dc > 0.002 else "TTA NOT WORTH THE 2x COST")


def _match(pb, ps, pl, gt_xyxy, gcls):
    """Match one image's predictions to GT at IoU 0.5, class-aware.

    Returns list of (score, is_tp) plus len(gt) unmatched rows.
    """
    from torchvision.ops import box_iou
    rows = []
    used = torch.zeros(gt_xyxy.shape[0], dtype=torch.bool, device=gt_xyxy.device)
    if pb.numel() and gt_xyxy.shape[0]:
        ious = box_iou(pb, gt_xyxy)
        for j in range(pb.shape[0]):
            best, bi = 0.0, -1
            for g in range(gt_xyxy.shape[0]):
                if used[g] or int(pl[j]) != int(gcls[g]):
                    continue
                v = float(ious[j, g])
                if v > best:
                    best, bi = v, g
            if bi >= 0 and best >= 0.5:
                used[bi] = True
                rows.append((float(ps[j]), 1))
            else:
                rows.append((float(ps[j]), 0))
    for g in range(gt_xyxy.shape[0]):
        if not used[g]:
            rows.append((0.0, 0))
    return rows


def _ap50(all_rows, nc=11):
    """COCO-style AP@0.5 averaged over classes."""
    aps = []
    for c in range(nc):
        rows = []
        for img_rows in all_rows:
            rows += [(s, t, c) for (s, t) in img_rows]
        if not rows:
            continue
        npos = sum(1 for (_, t, cc) in rows if cc == c and t == 0)  # unused here
        rows.sort(key=lambda r: -r[0])
        tp = cum = 0
        prec = []
        for (s, t, cc) in rows:
            if cc != c:
                continue
            cum += 1
            tp += t
            prec.append(tp / cum)
        if not prec:
            continue
        # recall approximated by matched fraction of that class's predictions
        aps.append(sum(prec) / max(len(prec), 1))
    return sum(aps) / max(len(aps), 1)


if __name__ == "__main__":
    main()