"""LayerCAM explainability for CropGuardMultiTaskV7 (v7-family checkpoints).

Ported from the app server's multiscale LayerCAM (app/api/model_server.py,
written for the v5 CropGuardMultiTask) to the V7 API. Same math: gradient of
the predicted-class logit w.r.t. the multi-scale feature maps P3+P4+P5 that
feed the classification head, LayerCAM element-wise ReLU gradient weighting,
per-level maps upsampled to input size and summed.

Instead of only producing pretty overlays, this script computes a measured
number: the fraction of CAM energy that falls inside the GT lesion mask (and
GT leaf mask) with `lift = cam_fraction / area_fraction` -- for a random map
the expected energy fraction equals the area fraction, so lift ~1 means
"says nothing", lift >> 1 means "the gradients point at the lesion".

Usage:
  python layercam_v7.py --ckpt /path/best_v9_raw.pt --data /path/corpus_v9 \
      --split val --n 40 --out /tmp/layercam_out
  # --ckpt none = random weights, mechanics self-test only
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TRAIN = os.path.join(_ROOT, "model", "training")
for p in (_ROOT, _TRAIN):
    if p not in sys.path:
        sys.path.insert(0, p)

SIZE = 320


def build_and_load(ckpt, device):
    import torch
    from model.architecture.v7_model import CropGuardMultiTaskV7
    model = CropGuardMultiTaskV7(nc=8, pretrained=False)
    info = {"ckpt": ckpt, "missing": None, "unexpected": None}
    if ckpt and ckpt != "none":
        obj = torch.load(ckpt, map_location=device, weights_only=False)
        sd = obj.get("model", obj) if isinstance(obj, dict) else obj
        missing, unexpected = model.load_state_dict(sd, strict=False)
        info["missing"], info["unexpected"] = len(missing), len(unexpected)
        info["meta_epoch"] = obj.get("epoch") if isinstance(obj, dict) else None
        # strict=False + hard gate: a partially-loaded CAM is worse than none
        if len(missing) > 5 or len(unexpected) > 5:
            raise RuntimeError(f"checkpoint/model mismatch: missing={missing[:5]} "
                               f"unexpected={unexpected[:5]}")
    model = model.to(device).eval()
    return model, info


def layercam_v7(model, x):
    """Predicted-class LayerCAM over P3+P4+P5. Returns (cam HxW in [0,1], class idx).

    grads-of-logit x ReLU, per server math; eval() keeps BatchNorm on running
    stats (batch=1), and the graph must be built under enable_grad because
    callers sit inside no_grad.
    """
    import torch
    import torch.nn.functional as F
    feats = {}
    handle = model.neck.register_forward_hook(
        lambda _m, _i, o: feats.update(zip(("p3", "p4", "p5"), o)))
    try:
        with torch.enable_grad():
            xg = x.detach().clone().float().requires_grad_(True)
            out = model(xg, active_tasks={"cls", "seg"})
            cidx = int(out["cls_logits"].argmax(1))
            levels = [feats["p3"], feats["p4"], feats["p5"]]
            grads = torch.autograd.grad(out["cls_logits"][0, cidx], levels)
    finally:
        handle.remove()
    cam = None
    for p, g in zip(levels, grads):
        alpha = torch.relu(g)                      # LayerCAM: element-wise
        contrib = torch.relu((alpha * p).sum(1, keepdim=True))
        contrib = F.interpolate(contrib, size=(SIZE, SIZE), mode="bilinear",
                                align_corners=False)
        cam = contrib if cam is None else cam + contrib
    cam = cam / cam.max().clamp(min=1e-6)
    return cam[0, 0].detach().cpu().numpy(), cidx


def energy_stats(cam, mask):
    """(energy fraction inside mask, area fraction) or None if mask empty."""
    m = mask > 0.5
    if m.sum() == 0 or cam.sum() <= 0:
        return None
    return float(cam[m].sum() / cam.sum()), float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="'none' for a random-weight self-test")
    ap.add_argument("--data", required=True, help="corpus root (images/labels/masks_*)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="layercam_out")
    ap.add_argument("--overlays", type=int, default=5)
    args = ap.parse_args()

    import torch
    from dataset import CropGuardDataset
    device = torch.device("cpu")
    model, info = build_and_load(args.ckpt, device)
    print(f"[cam] model: {info}", flush=True)

    ds = CropGuardDataset(args.data, args.split, SIZE, augment=False)
    n = min(args.n, len(ds.samples))
    os.makedirs(args.out, exist_ok=True)
    rows, les_e, les_a, leaf_e, leaf_a = [], [], [], [], []
    for i in range(n):
        sample = ds[i]
        x = sample["image"][None].float()
        cam, cidx = layercam_v7(model, x)
        stem = os.path.splitext(os.path.basename(ds.samples[i][0]))[0]
        row = {"stem": stem, "pred_cls": cidx, "cam_max": float(cam.max())}
        st = energy_stats(cam, sample["mask"].numpy())
        if st:
            row["lesion_energy_frac"], row["lesion_area_frac"] = st
            les_e.append(st[0])
            les_a.append(st[1])
        stl = energy_stats(cam, sample["leaf"].numpy())
        if stl:
            row["leaf_energy_frac"], row["leaf_area_frac"] = stl
            leaf_e.append(stl[0])
            leaf_a.append(stl[1])
        rows.append(row)
        if i < args.overlays:
            import torch.nn.functional as Fn
            from PIL import Image
            img = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            heat = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
            hm = (cam * 255).astype(np.uint8)
            heat[..., 0] = hm
            heat[..., 1] = hm // 3
            blend = (0.55 * img + 0.45 * heat).astype(np.uint8)
            Image.fromarray(blend).save(os.path.join(args.out, f"{stem}_cam.png"))
        print(f"[cam] {i + 1}/{n} {stem} cls={cidx} "
              f"lesE={row.get('lesion_energy_frac', float('nan')):.3f} "
              f"lesA={row.get('lesion_area_frac', float('nan')):.3f}", flush=True)

    summary = {
        "ckpt": args.ckpt, "split": args.split, "n": n,
        "n_with_lesion": len(les_e), "n_with_leaf": len(leaf_e),
        "lesion_cam_energy_frac_mean": float(np.mean(les_e)) if les_e else None,
        "lesion_area_frac_mean": float(np.mean(les_a)) if les_a else None,
        "lesion_lift": (float(np.mean(les_e) / np.mean(les_a))
                        if les_e and np.mean(les_a) > 0 else None),
        "leaf_cam_energy_frac_mean": float(np.mean(leaf_e)) if leaf_e else None,
        "leaf_area_frac_mean": float(np.mean(leaf_a)) if leaf_a else None,
        "leaf_lift": (float(np.mean(leaf_e) / np.mean(leaf_a))
                      if leaf_e and np.mean(leaf_a) > 0 else None),
        "rows": rows,
    }
    with open(os.path.join(args.out, "camstats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"[cam] wrote {args.out}/camstats.json")


if __name__ == "__main__":
    main()
