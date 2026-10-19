"""Evaluate v7 classification on the PlantDoc *test* split (tomato classes only).

PlantDoc is in-the-wild field imagery -- a genuinely different domain from the
PlantSeg-derived corpus -- and it carries NO masks, boxes or severity, so only
classification is measurable here. Reporting seg/severity on PlantDoc would be
inventing a target that does not exist.

PlantDoc tomato folders are mapped onto CropGuard's 11-class schema explicitly.
Unmapped (non-tomato) folders are skipped loudly; a tomato folder with no mapping
is a hard error, not a silent skip. PlantDoc tomato test happens to lack
leaf_miner / spider_mites / target_spot, so those classes have zero support --
they are reported with support 0 over the 8 classes PlantDoc actually contains,
and macro-F1 is computed over the 8 classes with support (otherwise the three
absent classes would each drag the mean to 0 for no reason).

  python3 model/training/eval_plantdoc.py \
      --ckpt model/weights/v7/best_stage5.pt \
      --root ~/tmp/plantdoc/test --out ~/tmp/plantdoc_v7.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CLASS_NAMES = ["healthy", "bacterial_spot", "early_blight", "late_blight",
               "leaf_mold", "mosaic_virus", "septoria_leaf_spot",
               "yellow_leaf_curl_virus"]

# PlantDoc tomato folder -> CropGuard class name (verified against the dataset's
# own test/ directory listing, not guessed).
PLANTDOC_TOMATO = {
    "Tomato_leaf": "healthy",
    "Tomato_leaf_bacterial_spot": "bacterial_spot",
    "Tomato_Early_blight_leaf": "early_blight",
    "Tomato_leaf_late_blight": "late_blight",
    "Tomato_mold_leaf": "leaf_mold",
    "Tomato_leaf_mosaic_virus": "mosaic_virus",
    "Tomato_Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato_leaf_yellow_virus": "yellow_leaf_curl_virus",
}


def build_index(root):
    """Return (paths, labels) for every tomato test image we have a mapping for."""
    cls_id = {n: i for i, n in enumerate(CLASS_NAMES)}
    present = sorted(os.listdir(root))
    unmapped_tomato = [d for d in present
                       if d.startswith("Tomato") and d not in PLANTDOC_TOMATO]
    if unmapped_tomato:
        sys.exit(f"PlantDoc tomato folders with no mapping: {unmapped_tomato}")
    skipped = [d for d in present if not d.startswith("Tomato")]
    paths, labels = [], []
    per = {}
    for d in present:
        if d not in PLANTDOC_TOMATO:
            continue
        cid = cls_id[PLANTDOC_TOMATO[d]]
        for f in sorted(os.listdir(os.path.join(root, d))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, d, f))
                labels.append(cid)
                per[d] = per.get(d, 0) + 1
    return paths, labels, per, skipped


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True, help="PlantDoc test/ directory")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--nc", type=int, default=11)
    ap.add_argument("--unsure-thresh", type=float, default=0.60,
                    help="confidence below this -> 'unsure' (use the value "
                         "calibrate_abstention.py printed, e.g. 0.231)")
    ap.add_argument("--out", default="plantdoc_metrics.json")
    args = ap.parse_args()

    from model.architecture.v7_model import CropGuardMultiTaskV7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck.get("model", ck)
    model = CropGuardMultiTaskV7(nc=args.nc, pretrained=False).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    T = float(model.abstain.temperature.item())
    print(f"loaded {args.ckpt}; temperature={T:.4f}", flush=True)

    paths, labels, per, skipped = build_index(args.root)
    print(f"PlantDoc tomato test images: {len(paths)}  per-class: {per}")
    print(f"skipped non-tomato folders: {len(skipped)}", flush=True)

    logits_all, probs_all = [], []
    for i in range(0, len(paths), args.batch_size):
        batch = paths[i:i + args.batch_size]
        x = torch.stack([
            TF.to_tensor(Image.open(p).convert("RGB").resize(
                (args.size, args.size), Image.BILINEAR)) for p in batch]).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x, with_analysis=True, return_abstain=True)
        logits_all.append(out["cls_logits"].float().cpu())
        probs_all.append(out["cls_probs"].float().cpu())
    logits = torch.cat(logits_all)
    probs = torch.cat(probs_all)
    labels_t = torch.tensor(labels)

    pred = logits.argmax(-1)
    acc = float((pred == labels_t).float().mean())
    conf = probs.max(-1).values
    uns = conf < args.unsure_thresh       # calibrated "unsure" operating point
    # accuracy on the confident subset, and how often we would abstain
    kept_acc = float((pred[~uns] == labels_t[~uns]).float().mean()) \
        if (~uns).any() else float("nan")

    per_class = {}
    f1s = []
    for c, name in enumerate(CLASS_NAMES):
        idx = labels_t == c
        n = int(idx.sum())
        if n == 0:
            per_class[name] = {"support": 0}
            continue
        tp = int(((pred == c) & idx).sum())
        fp = int(((pred == c) & ~idx).sum())
        fn = int(((pred != c) & idx).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[name] = {"support": n, "precision": prec, "recall": rec,
                           "f1": f1}
        f1s.append(f1)

    # confusion matrix over the classes PlantDoc actually contains
    present_ids = sorted({int(x) for x in labels_t.tolist()})
    conf_mat = {}
    for t in present_ids:
        row = {}
        for p in present_ids:
            row[CLASS_NAMES[p]] = int(((labels_t == t) & (pred == p)).sum())
        conf_mat[CLASS_NAMES[t]] = row

    res = {
        "dataset": "PlantDoc",
        "split": "test",
        "domain": "in-the-wild field imagery (different from PlantSeg corpus)",
        "metric_scope": "classification only (PlantDoc has no masks/boxes/severity)",
        "n_images": len(paths),
        "n_classes_with_support": len(f1s),
        "cls_top1": acc,
        "cls_macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class": per_class,
        "confusion": conf_mat,
        "temperature": T,
        "unsure_threshold": args.unsure_thresh,
        "unsure_rate": float(uns.float().mean()),
        "cls_top1_confident": kept_acc,
        "skipped_non_tomato_folders": len(skipped),
    }
    print(json.dumps(res, indent=2))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
