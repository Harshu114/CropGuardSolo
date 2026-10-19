#!/usr/bin/env python3
"""DIAGNOSTIC: per-class accuracy at native vs degraded image size.

Answers "does the resolution collapse hit the specific classes the user's
images failed on (bacterial_spot / early_blight), or is it uniform?"
Run as a FILE (DataLoader workers cannot fork from a stdin script).

Usage: python _diag_perclass_degraded.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import model_server as ms
from dataset import CropGuardDataset, collate
import _diag_degrade as dg

NAMES = ["healthy", "bacterial_spot", "early_blight", "late_blight",
         "leaf_miner", "leaf_mold", "mosaic_virus", "septoria_leaf_spot",
         "spider_mites", "target_spot", "yellow_leaf_curl_virus"]

torch.set_grad_enabled(False)


def per_class(root, model, device):
    ds = CropGuardDataset(str(root), split="test", size=320, augment=False)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                    collate_fn=collate)
    n = np.zeros(11, int)
    ok = np.zeros(11, int)
    for b in ld:
        x = b["image"].to(device)
        y = b["cls"]
        v = y >= 0
        if v.sum() == 0:
            continue
        lo = model(x, with_severity=False)["cls_logits"]
        lf = model(torch.flip(x, dims=[3]), with_severity=False)["cls_logits"]
        p = (0.5 * (lo + lf)).argmax(1).cpu()
        for t, pr in zip(y[v].tolist(), p[v].tolist()):
            n[t] += 1
            ok[t] += int(t == pr)
    return n, ok


def main() -> int:
    model = ms.load_model(str(dg.CKPT))
    device = ms.get_device()
    variants = [("native", None), ("s160", 160), ("s160_jpg50", 160)]
    res = {}
    for name, side in variants:
        q = 50 if name.endswith("jpg50") else 95
        res[name] = per_class(dg.build(name, side, q), model, device)
        print(f"built {name}")

    print(f"\n{'cls':<2} {'name':<24} {'n':>4} {'native':>8} {'160px':>8} {'160+q50':>9}")
    for t in range(11):
        cells = []
        for k in ("native", "s160", "s160_jpg50"):
            n, ok = res[k]
            cells.append(f"{ok[t]/n[t]:>8.3f}" if n[t] else f"{'-':>8}")
        print(f"{t:<2} {NAMES[t]:<24} {res['native'][0][t]:>4} " + " ".join(cells))

    print()
    for k in ("native", "s160", "s160_jpg50"):
        n, ok = res[k]
        print(f"  {k:<12} overall {ok.sum()}/{n.sum()} = {ok.sum()/n.sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
