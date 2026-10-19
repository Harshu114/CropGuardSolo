"""Rank every v5 checkpoint on the VAL split to find the true best one.

Free (no Kaggle quota): runs entirely on the local GPU. v5 only ever evaluated
`student_best.pt` (== best_stage5), selected by lowest training loss. Loss is a
poor proxy for mAP, so the stage that minimises total loss is not necessarily
the stage with the best detection/classification. Selection must be done on
VAL, never on test, or the test numbers become a selection artefact.

Usage:  python3 rank_ckpts.py [max_images]
"""
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/logiclocus/HarshalPrivate/Projects/collegeProjects/All tries of cropguard/ClaudeCropGuard/datasetclaude"
# Output goes beside this file, NOT /tmp. This machine wipes /tmp periodically
# and silently destroyed a completed ranking run before the results were read.
TMP = os.path.join(HERE, "_rank")
CKPT_DIR = os.path.join(HERE, "runs", "checkpoints")

CKPTS = ["best_stage1.pt", "best_stage2.pt", "best_stage3.pt",
         "best_stage4.pt", "best_stage5.pt", "last.pt"]


def strip(ckpt_path: str) -> str:
    """Stage checkpoints carry optimizer state; eval.py wants a bare state_dict."""
    os.makedirs(TMP, exist_ok=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    out = os.path.join(TMP, os.path.basename(ckpt_path))
    torch.save(sd, out)
    return out


def main():
    max_images = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rows = []
    for name in CKPTS:
        path = os.path.join(CKPT_DIR, name)
        if not os.path.exists(path):
            print(f"skip {name} (missing)")
            continue
        bare = strip(path)
        out_json = os.path.join(TMP, name.replace(".pt", ".json"))
        cmd = [sys.executable, os.path.join(HERE, "eval.py"),
               "--root", ROOT, "--ckpt", bare, "--split", "val",
               "--size", "320", "--batch-size", "16", "--workers", "4",
               "--conf", "0.05", "--nc", "11", "--out", out_json]
        if max_images:
            cmd += ["--max-images", str(max_images)]
        print(f"--- evaluating {name} ---", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out_json):
            print(f"  FAILED: {r.stderr[-600:]}")
            continue
        m = json.load(open(out_json))
        rows.append({
            "ckpt": name,
            "mAP@0.5": m["mAP@0.5"],
            "mAP@0.5:0.95": m["mAP@0.5:0.95"],
            "cls_top1": m["cls_top1"],
            "cls_macro_f1": m["cls_macro_f1"],
            "seg_mIoU": m["seg_mIoU"],
            "seg_dice": m["seg_dice"],
        })
        print(f"  mAP@0.5 {m['mAP@0.5']:.4f}  mAP@.5:.95 {m['mAP@0.5:0.95']:.4f}  "
              f"cls_top1 {m['cls_top1']:.4f}", flush=True)

    if not rows:
        sys.exit("no checkpoints evaluated")

    print("\n" + "=" * 92)
    print(f"{'checkpoint':<18}{'mAP@0.5':>11}{'mAP@.5:.95':>13}"
          f"{'cls_top1':>11}{'cls_mF1':>10}{'seg_mIoU':>11}{'seg_dice':>11}")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: -x["mAP@0.5"]):
        print(f"{r['ckpt']:<18}{r['mAP@0.5']:>11.4f}{r['mAP@0.5:0.95']:>13.4f}"
              f"{r['cls_top1']:>11.4f}{r['cls_macro_f1']:>10.4f}"
              f"{r['seg_mIoU']:>11.4f}{r['seg_dice']:>11.4f}")
    print("=" * 92)

    json.dump(rows, open(os.path.join(TMP, "ranking.json"), "w"), indent=2)
    print(f"\nval images used: {max_images or 'all'}")


if __name__ == "__main__":
    main()