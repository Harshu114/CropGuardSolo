"""Calibrate v7 abstention (temperature scaling + energy threshold).

Run after v7 training, on the val split:

    python3 model/training/calibrate_abstention.py \
        --ckpt runs/v7/student_best.pt --root $CROPGUARD_DATA --split val

Writes the fitted temperature INTO the checkpoint (in place, `--dry-run` to
preview) and prints the recommended prob/energy thresholds at the operating
point you choose (default: target 5% abstain rate).
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.architecture.v7_model import CropGuardMultiTaskV7  # noqa: E402


@torch.no_grad()
def collect(model, loader, device):
    logits, labels = [], []
    for batch in loader:
        out = model(batch["image"].to(device), active_tasks={"cls"})
        logits.append(out["cls_logits"].float().cpu())
        labels.append(batch["cls"])
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits, labels):
    """NLL-based temperature scaling on the val set (Guo et al. 2017)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return logT.exp().item()


def choose_thresholds(logits, labels, temp, target_abstain=0.05):
    probs = torch.softmax(logits / temp, dim=-1)
    conf, pred = probs.max(-1)
    acc = (pred == labels).float()
    energy = -torch.logsumexp(logits / temp, dim=-1)
    # prob threshold hitting target abstain rate
    k = max(1, int(len(conf) * target_abstain))
    p_thr = conf.sort().values[k - 1].item()
    e_thr = energy.sort().values[k - 1].item()
    # coverage-weighted accuracy of those kept
    keep = conf >= p_thr
    kept_acc = acc[keep].mean().item() if keep.any() else float("nan")
    return {"prob_thresh": p_thr, "energy_thresh": e_thr,
            "abstain_rate": float((~keep).float().mean()),
            "accuracy_kept": kept_acc, "accuracy_all": acc.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", default=os.environ.get(
        "CROPGUARD_DATA", "/kaggle/input/cropguard-corpus-claude"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--target-abstain", type=float, default=0.05)
    ap.add_argument("--write", action="store_true",
                    help="write fitted temperature back into the checkpoint")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import CropGuardDataset, collate
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck.get("model", ck)
    model = CropGuardMultiTaskV7(nc=11, pretrained=False).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    ds = CropGuardDataset(args.root, args.split, args.size, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
    logits, labels = collect(model, loader, device)
    valid = labels >= 0
    logits, labels = logits[valid], labels[valid]

    temp = fit_temperature(logits, labels)
    thr = choose_thresholds(logits, labels, temp, args.target_abstain)
    print(json.dumps({"temperature": temp, **thr}, indent=2))

    n_params = len(sd)
    if args.write and not args.dry_run:
        if "abstain.temperature" not in sd:
            sys.exit("checkpoint has no abstain.temperature buffer")
        sd["abstain.temperature"] = torch.tensor(temp)
        torch.save(ck, args.ckpt)
        print(f"wrote temperature={temp:.4f} into {args.ckpt} ({n_params} tensors)")


if __name__ == "__main__":
    main()
