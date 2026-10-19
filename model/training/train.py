"""CropGuard staged training (paper Table 1 + Table 2).

Stages: 1 backbone+det | 2 backbone+cls | 3 backbone+seg | 4 full det+cls+seg
        | 5 full + severity
AdamW lr=1e-3 wd=1e-4, cosine annealing with warmup, AMP FP16,
checkpoint every epoch, resumable (survives Kaggle 9h session limit).
"""
import argparse
import json
import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# Runnable both as `python model/training/train.py` (repo root or Kaggle code
# dataset) and as a module. Running a script puts ITS directory on sys.path, not
# the repo root, so add the root here for `model.architecture...`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset import CropGuardDataset, collate          # noqa: E402  (same dir)
from losses import multitask_loss, PCGrad              # noqa: E402  (same dir)
from model.architecture.model import CropGuardMultiTask  # noqa: E402
from model.architecture.v7_model import CropGuardMultiTaskV7  # noqa: E402

# Corpus root. Resolved from CROPGUARD_DATA if set, else the Kaggle dataset this
# pipeline was trained on. Never hardcode a machine-local path here again -- the
# earlier copies pointed at a folder that only existed on one laptop, so every
# eval/diagnostic script silently found no data.
DEFAULT_DATA_ROOT = os.environ.get("CROPGUARD_DATA",
                                   "/kaggle/input/cropguard-corpus-claude")

STAGE_MODULES = {  # modules trainable per stage
    1: ["backbone", "neck", "det_head"],
    # BUGFIX: stages 2/3 used to train backbone+neck, which DESTROYED the
    # frozen detection head -- its inputs drifted out from under it. Measured
    # on the real stage-2 checkpoint: per-image best IoU vs a lesion fell from
    # 0.938 (end of stage 1) to 0.079, and mAP@0.5 went to 0.0. Head-only
    # training here keeps the stage-1 detector intact until stages 4-5
    # unfreeze everything and fine-tune all heads jointly on purpose.
    2: ["cls_head"],
    3: ["seg_head"],
    4: ["backbone", "neck", "det_head", "cls_head", "seg_head"],
    5: ["backbone", "neck", "det_head", "cls_head", "seg_head", "sev_head",
        "seg_embed"],
}
STAGE_LOSSES = {1: ["det"], 2: ["cls"], 3: ["seg"],
                4: ["det", "cls", "seg"], 5: ["det", "cls", "seg", "sev"]}
# Learning rate schedule: Stage 1 is detector pretraining, later stages
# fine-tune the already-developed representation with lower LR
STAGE_LRS = {1: 1e-3, 2: 3e-4, 3: 3e-4, 4: 3e-4, 5: 1e-4}


def set_stage(model, stage):
    trainable = set(STAGE_MODULES[stage])
    top = ["backbone", "neck", "det_head", "cls_head", "seg_head", "sev_head",
           "seg_embed"]
    for name in top:
        mod = getattr(model, name, None)
        if mod is None:
            continue
        req = name in trainable
        for p in mod.parameters():
            p.requires_grad = req
    # BatchNorm in frozen parts must stay eval to avoid stat drift
    for name in top:
        mod = getattr(model, name, None)
        if mod is None or name in trainable:
            continue
        mod.eval()
    # which heads actually RUN this stage (skip inactive ones entirely:
    # prevents frozen-head FP16 overflow + BN-stat poisoning)
    return {t for t in ("det", "cls", "seg")
            if f"{t}_head" in trainable}


class EMA:
    """Exponential moving average of weights (expert: worth it).

    Kept in fp32 and evaluated with the EMA weights at validation/checkpoint
    time; a cheap, reliable accuracy win for a one-day budget.
    """

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
    """Swap EMA weights in for evaluation, return a restore callable."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()
              if k in ema.shadow}
    ema.copy_to(model)

    def restore():
        sd = model.state_dict()
        for k, v in backup.items():
            sd[k].copy_(v)
    return restore


def build_optimizer(model, stage, lr, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def make_sampler(ds):
    """Class-balanced sampling (paper sec 2.8): weight ~ 1/sqrt(count)."""
    counts = {}
    cls_labels = []
    for i in range(len(ds)):
        img_path, lbl_path = ds.samples[i]
        from dataset import load_yolo_labels
        boxes = load_yolo_labels(lbl_path)
        c = int(max((b[0] for b in boxes), default=-1))
        cls_labels.append(c)
        counts[c] = counts.get(c, 0) + 1
    med = sorted(counts.values())[len(counts) // 2]
    weights = [1.0 / math.sqrt(counts.get(c, med)) for c in cls_labels]
    return WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)


def cosine_lr(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


@torch.no_grad()
def validate(model, loader, device, stage, img_size, nc, reg_max, amp=True,
             active_tasks=None, two_ch=False):
    model.eval()
    sums, n = {}, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(batch["image"], with_severity=(stage >= 5) and not two_ch,
                        active_tasks=active_tasks) if not two_ch else \
                model(batch["image"], with_analysis=(stage >= 5),
                      active_tasks=active_tasks)
            total, parts = multitask_loss(
                out, batch, img_size, nc=nc, reg_max=reg_max,
                train_sev=(stage >= 5) and not two_ch, active_tasks=active_tasks,
                two_channel_seg=two_ch)
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        sums["total"] = sums.get("total", 0.0) + float(total)
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def run_stage(args, model, stage, train_ds, val_ds, device, sampler=None,
              start_epoch=0, best=None, ck=None):
    two_ch = getattr(model, "arch_name", "v5") == "v7"
    ema = EMA(model, decay=args.ema_decay) if (args.ema and two_ch) else None

    def gt_prob_for(epoch):
        """GT-gate probability schedule (only meaningful for v7, stage >= 4).
        Starts at 0.8 so the classifier first learns from clean masks and ends
        at 0.2 so it mostly sees its own predictions, matching inference."""
        if not two_ch or stage < 4:
            return 0.0
        frac = epoch / max(epochs - 1, 1)
        return 0.8 - 0.6 * frac
    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None, num_workers=args.workers, pin_memory=True,
        collate_fn=collate, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True, collate_fn=collate)
    # BUGFIX: set_stage() MUST run before build_optimizer(). set_stage() is
    # what flips requires_grad for this stage; building the optimizer first
    # made it capture the PREVIOUS stage's parameter set. That was invisible
    # while stages 2/3 also trained the backbone (same set as stage 1), but
    # with head-only stages the optimizer held frozen params while the loss
    # only touched the new head, so backward produced zero gradients for the
    # optimizer and GradScaler.step() aborted with
    # "No inf checks were recorded for this optimizer."
    active_tasks = set_stage(model, stage)  # heads that RUN this stage
    opt = build_optimizer(model, stage, args.lr, args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    pcgrad = PCGrad(opt, scaler) if (args.pcgrad and stage >= 4) else None

    steps_per_epoch = len(loader)
    epochs = args.stage_epochs
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.2 * total_steps)) if epochs < 25 else 5 * steps_per_epoch

    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.out, "logs", f"stage{stage}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(msg):
        line = f"[stage {stage}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    step = start_epoch * steps_per_epoch
    nonfinite_count = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        # BUGFIX: re-assert module modes EVERY epoch. validate() ends with
        # model.train(), which flips FROZEN modules (e.g. the backbone during
        # head-only stages 2/3) back to train mode, letting their BatchNorm
        # running stats drift -- another way the frozen detector's inputs got
        # corrupted. set_stage() re-applies eval() to every non-trainable
        # module, so a frozen backbone keeps the exact stats it had coming in.
        model.train()
        set_stage(model, stage)
        # BN-safe snapshot: weights AND buffers (BN running stats) each epoch.
        # Non-finite batch -> rollback here -> poisoning self-heals.
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rolled_back = False
        ep_bad = 0        # non-finite batches seen in THIS epoch
        for batch in loader:
            for g in opt.param_groups:
                # Stage-dependent LR: Stage 1 = 1e-3, later stages = 3e-4/1e-4
                g["lr"] = cosine_lr(step, total_steps, warmup,
                                    STAGE_LRS.get(stage, args.lr))
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(batch["image"], with_analysis=(stage >= 5),
                            active_tasks=active_tasks,
                            gt_mask=batch.get("mask"),
                            gt_mask_prob=gt_prob_for(epoch)) if two_ch else \
                    model(batch["image"], with_severity=(stage >= 5),
                          active_tasks=active_tasks)
                total, parts = multitask_loss(
                    out, batch, args.size, nc=args.nc, reg_max=args.reg_max,
                    train_sev=(stage >= 5) and not two_ch, active_tasks=active_tasks,
                    two_channel_seg=two_ch)
            if not torch.isfinite(total):
                # BUGFIX: a single bad batch used to discard the ENTIRE epoch --
                # weights AND BatchNorm stats were restored to the epoch-start
                # snapshot and the step counter reset. With an un-prior'd
                # detector that fired often enough that the model kept
                # restarting and effectively learned nothing all run.
                # Now: skip only the bad batch, and roll the epoch back just if
                # the instability is widespread (>=5% of the epoch's steps).
                nonfinite_count += 1
                ep_bad += 1
                opt.zero_grad(set_to_none=True)
                # NB: do NOT call scaler.update() here. update() raises
                # "No inf checks were recorded prior to update." unless step()
                # ran first, which would crash the whole run on the very first
                # non-finite loss -- the exact case this branch exists for.
                step += 1
                if ep_bad >= max(5, steps_per_epoch // 20):
                    log(f"ep {epoch}: {ep_bad} non-finite batches "
                        f"(>={max(5, steps_per_epoch // 20)}) -> rolling back "
                        f"epoch and halving stage LR")
                    model.load_state_dict(snap)
                    scaler = torch.amp.GradScaler(
                        "cuda", enabled=args.amp and device.type == "cuda")
                    step = epoch * steps_per_epoch
                    rolled_back = True
                    STAGE_LRS[stage] = STAGE_LRS.get(stage, args.lr) * 0.5
                    nonfinite_count = 0
                    break
                log(f"ep {epoch} it {step % steps_per_epoch}: NON-FINITE LOSS "
                    f"{parts} -- skipping batch ({ep_bad} this epoch)")
                continue
            if pcgrad:
                # BUGFIX: parts holds "det_cls"/"det_box"/"det_dfl", never a
                # bare "det", so the old `if k in parts` filter silently
                # DROPPED detection from PCGrad -- the detector received no
                # gradient at all in stages 4-5 (20 of 50 epochs).
                task_list = []
                for k in STAGE_LOSSES[stage]:
                    if k == "det":
                        if "det_cls" in parts:
                            task_list.append(parts["det_cls"]
                                             + parts["det_box"] + parts["det_dfl"])
                    elif k in parts:
                        task_list.append(parts[k])
                pcgrad.step(task_list)
            else:
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 10.0)
                scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            step += 1
            if step % args.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30 if \
                    device.type == "cuda" else 0
                log(f"ep {epoch} it {step % steps_per_epoch}/{steps_per_epoch} "
                    f"lr {opt.param_groups[0]['lr']:.2e} "
                    + " ".join(f"{k} {float(v.detach()):.3f}" for k, v in parts.items())
                    + f" total {float(total.detach()):.3f} vram {mem:.2f}G")
        # end epoch: validate + checkpoint (skipped if rolled back)
        if rolled_back:
            log(f"ep {epoch} rolled back; restarting epoch with lower lr")
            continue
        restore = ema_eval(model, ema) if ema is not None else None
        vm = validate(model, val_loader, device, stage, args.size, args.nc,
                      args.reg_max, amp=args.amp, active_tasks=active_tasks,
                      two_ch=two_ch)
        if restore is not None:
            restore()
        log(f"ep {epoch} VAL " + " ".join(f"{k} {v:.3f}" for k, v in vm.items())
            + f" ({time.time() - t0:.0f}s)")
        
        # Log to training_log.csv
        csv_path = os.path.join(args.out, "training_log.csv")
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, "a") as f_csv:
            if not csv_exists:
                header = ["stage", "epoch", "time_sec"] + sorted(vm.keys())
                f_csv.write(",".join(header) + "\n")
            row = [str(stage), str(epoch + 1), f"{time.time() - t0:.1f}"] + [f"{vm[k]:.4f}" for k in sorted(vm.keys())]
            f_csv.write(",".join(row) + "\n")

        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "scaler": scaler.state_dict(), "stage": stage, "epoch": epoch + 1,
                 "best": best or {}, "args": vars(args)}
        torch.save(state, os.path.join(ckpt_dir, "last.pt"))
        cur = vm.get("total", 9e9)
        if best is None or cur < best.get(f"stage{stage}", 9e9):
            best = best or {}
            best[f"stage{stage}"] = cur
            state["best"] = best
            torch.save(state, os.path.join(ckpt_dir, f"best_stage{stage}.pt"))
            torch.save(model.state_dict(), os.path.join(args.out, "student_best.pt"))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_DATA_ROOT,
                    help="corpus root (env CROPGUARD_DATA); must contain "
                         "images/{split}, labels/{split}, masks/{split}")
    ap.add_argument("--out", default="/kaggle/working/runs")
    ap.add_argument("--stages", default="1,2,3,4,5")
    ap.add_argument("--stage-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--nc", type=int, default=11)
    ap.add_argument("--reg-max", type=int, default=8)
    ap.add_argument("--arch", default="v5", choices=["v5", "v7"],
                    help="v5 = shipped architecture; v7 = lesion-gated + "
                         "2-ch seg + analytic severity + abstention")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="disable timm pretrained weights (smoke tests only)")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--pcgrad", action="store_true",
                    help="gradient surgery; expert advice: skip on a 1-day "
                         "budget (off by default)")
    ap.add_argument("--ema", action="store_true", default=True)
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--max-images", type=int, default=None,
                    help="smoke-test subset size")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force-stage", type=int, default=None,
                    help="round-2 training: ignore saved stage and start here "
                         "(e.g. --stages 4,5 --force-stage 4 to add joint epochs)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name(0)} capability {cap}", flush=True)
        if cap[0] < 7:
            raise RuntimeError(
                "GPU compute capability < 7.0 (P100 bug on Kaggle). "
                "Switch accelerator to 'GPU T4 x2' and restart.")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB",
              flush=True)

    if args.arch == "v7":
        model = CropGuardMultiTaskV7(nc=args.nc, pretrained=not args.no_pretrained,
                                     reg_max=args.reg_max).to(device)
    else:
        model = CropGuardMultiTask(nc=args.nc, pretrained=True,
                                   reg_max=args.reg_max).to(device)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model params ({args.arch}): {n_par:.2f}M", flush=True)

    train_ds = CropGuardDataset(args.root, "train", args.size, augment=True,
                                max_images=args.max_images)
    val_ds = CropGuardDataset(args.root, "val", args.size, augment=False,
                              max_images=args.max_images)
    print(f"train {len(train_ds)} | val {len(val_ds)}", flush=True)
    sampler = make_sampler(train_ds)  # computed once, reused by all stages
    stages = [int(s) for s in args.stages.split(",")]

    start_stage, start_epoch, best = 0, 0, None
    last = os.path.join(args.out, "checkpoints", "last.pt")
    if args.resume and not os.path.exists(last):
        # cross-notebook recovery: copy last.pt from an attached input
        # (e.g. Notebook 1's saved output) into /kaggle/working/runs.
        # NEVER take smoke-test checkpoints (tiny subset, garbage weights).
        import glob as _glob, shutil
        cands = [c for c in _glob.glob("/kaggle/input/**/checkpoints/last.pt",
                                       recursive=True) if "/smoke/" not in c]
        cands.sort(key=len)  # shortest path = most top-level real run
        if not cands:
            print("[resume] no last.pt found in /kaggle/input -- did you attach "
                  "the previous notebook's output (e.g. cropguardv4)? "
                  "Starting from scratch instead.", flush=True)
        for cand in cands:
            os.makedirs(os.path.dirname(last), exist_ok=True)
            shutil.copy(cand, last)
            print(f"recovered checkpoint from {cand}", flush=True)
            break
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt = build_optimizer(model, ck["stage"], args.lr, args.wd)
        opt.load_state_dict(ck["opt"])
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
        scaler.load_state_dict(ck["scaler"])
        start_stage, start_epoch, best = ck["stage"], ck["epoch"], ck.get("best")
        print(f"resumed stage {start_stage} epoch {start_epoch}", flush=True)
        # Overwrite LR from command line args for the resumed stage
    for stage in stages:
        if stage < start_stage or (stage == start_stage and start_epoch >= args.stage_epochs):
            continue
        ep = start_epoch if stage == start_stage else 0
        best = run_stage(args, model, stage, train_ds, val_ds, device,
                         sampler=sampler, start_epoch=ep, best=best)
        start_epoch = 0
        ck_last = os.path.join(args.out, "checkpoints", "last.pt")
        if os.path.exists(ck_last):
            ck = torch.load(ck_last, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])  # carry stage-N weights forward
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
