"""CropGuard v9 kernel: v8 recipe on the repaired-mask corpus.

v9 = v8 driver except the DATA: masks come from the dataset
`ganrajlongadge/cropguard-v9-filled-masks`, which repairs two v7 defects --
  masks_lesion: 1-px polygon outlines -> filled polygon regions;
  masks_leaf:   healthy-tissue-only -> fill_holes(leaf) | lesion (whole leaf,
                fixes the severity denominator and the channel-2 target).
prepare_corpus() builds WORK/corpus_v9 (v7 images/labels symlinked + repaired
masks) and every stage -- gates included -- reads that root, verifying the
repair fingerprints first. Artifact filenames keep the v8 names (best_v8_raw.pt,
stage_summary.json, ...) so existing tooling works.

Reads code from the dataset `ganrajlongadge/cropguard-v8-code` (layout at the
mount root: training/, architecture/, __init__.py) and data + old checkpoint
from `ganrajlongadge/cropguard-v7-diagnostics-input` (data/, checkpoint/).

Every stage is wrapped in try/except, prints its traceback and continues;
results are written to /kaggle/working after every stage so a crash never
costs the whole session.

STAGE 0  sha256 gate, forced-geometry gate re-run on Kaggle, pretrained-weight
         check, label/class-count reconciliation.
STAGE 1  memorization gate: 8 images, 500 steps, augmentation ON.
         PASS (lesion IoU >= 0.5) -> train normally.
         FAIL -> print optimizer/requires_grad/grad-norm/per-component
         diagnostics, retry with geometric aug OFF (photometric only).
         If that passes, PLAN B = photometric augmentation only.
STAGE 2  main training (cls + seg), 120 epochs, EMA, val every epoch.
STAGE 3  val summary for the selected checkpoint (NO test data touched).

Why this file does not import train_full_v7: that module does
`from dataset import ...` at import time and relies on cwd, which is exactly
how the v2 run died with ModuleNotFoundError. EMA and cosine_lr are inlined
here, and this driver never imports it.
"""
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback

WORK = os.environ.get("V8_WORK", "/kaggle/working")
os.makedirs(WORK, exist_ok=True)

EXPECTED_DATASET_SHA256 = "ff0102ed509d595abe08b79b8e1902fd6756cb7404f5efd102d1a507b90696db"
CAP_MINUTES = float(os.environ.get("V8_CAP_MIN", "55"))
EPOCHS = int(os.environ.get("V8_EPOCHS", "120"))
BATCH = int(os.environ.get("V8_BATCH", "8"))
LR = float(os.environ.get("V8_LR", "5e-4"))
SEED = 1337
NC = 8
SIZE = 320

# ---- hard wall-clock safety net (added after the v9 overrun, 2026-10-12) -----
# Kaggle publishes a running kernel's log/output only when the session ENDS, so
# a silent hang (stuck DataLoader worker / stuck subprocess) is invisible until
# the run is already dead. These bounds make the run end by itself:
#   * start_watchdog() force-exits the process once GLOBAL_DEADLINE_MIN
#     elapses (a separate thread is the only way to break a C-level block);
#   * subprocess.run / DataLoader get explicit timeouts, so a stall becomes a
#     catchable exception the stage wrappers already record.
# None of these touch the training math, so v8/v9/v10 comparability is intact.
GLOBAL_DEADLINE_MIN = float(os.environ.get("V8_DEADLINE_MIN", "100"))
SUB_TIMEOUT_S = int(os.environ.get("V8_SUB_TIMEOUT", "600"))
LOADER_TIMEOUT_S = int(os.environ.get("V8_LOADER_TIMEOUT", "300"))
NUM_WORKERS = int(os.environ.get("V8_NUM_WORKERS", "2"))
_T0 = time.time()
_WATCHDOG_STOP = threading.Event()
# Set by check_pretrained(): False means the Hub download failed and the run
# continues from random init with a loud warning (per the brief).
PRETRAINED_OK = True

# Env overrides exist so the whole driver can be smoke-tested on the laptop
# against a local corpus before it is allowed to burn GPU quota.
CODE_ROOTS = ([p for p in os.environ.get("V8_CODE_ROOT", "").split(os.pathsep) if p]
              + ["/kaggle/input/cropguard-v8-code",
                 "/kaggle/input/datasets/ganrajlongadge/cropguard-v8-code"])
V9_CORPUS = os.environ.get("V9_CORPUS", os.path.join(WORK, "corpus_v9"))
INPUT_ROOTS = ([V9_CORPUS]   # built by prepare_corpus()
               + [p for p in os.environ.get("V8_INPUT_ROOT", "").split(os.pathsep) if p]
               + ["/kaggle/input/cropguard-v7-diagnostics-input",
                  "/kaggle/input/datasets/ganrajlongadge/cropguard-v7-diagnostics-input"])
FILLED_ROOTS = ([p for p in os.environ.get("V9_FILLED_ROOT", "").split(os.pathsep) if p]
                + ["/kaggle/input/cropguard-v9-filled-masks",
                   "/kaggle/input/datasets/ganrajlongadge/cropguard-v9-filled-masks"])
# Constant-severity MAE on val over the v9 corpus (predict split mean, the same
# protocol the 0.0384 outline number used; n=151 = the trainer's own val set).
# Replaces the v8 constant 0.0384 -- different targets.
V9_BASELINE_MAE = float(os.environ.get("V9_BASELINE_MAE", "0.2293"))
STAGE1_STEPS = int(os.environ.get("V8_STAGE1_STEPS", "500"))
# Laptop-smoke-test-only caps (default 0 = use every image, as Kaggle does).
TRAIN_SUBSET = int(os.environ.get("V8_TRAIN_SUBSET", "0"))
VAL_SUBSET = int(os.environ.get("V8_VAL_SUBSET", "0"))

STATE = {}


def log(msg):
    print(f"[v9] {msg}", flush=True)


def save_json(name, obj):
    p = os.path.join(WORK, name)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log(f"wrote {p}")
    return p


def start_watchdog():
    """Force-exit if GLOBAL_DEADLINE_MIN is exceeded; heartbeat otherwise.

    A stuck DataLoader worker or a hung subprocess blocks the main thread
    inside C code, which no Python-level check can interrupt -- a daemon thread
    can. It writes a marker into WORK (Kaggle still publishes output for a
    failed run) and exits non-zero, so the run is bounded instead of hanging.
    """
    def _loop():
        while not _WATCHDOG_STOP.wait(30):
            elapsed = (time.time() - _T0) / 60.0
            if elapsed > GLOBAL_DEADLINE_MIN:
                log(f"GLOBAL DEADLINE {GLOBAL_DEADLINE_MIN:.0f} min exceeded "
                    f"(elapsed {elapsed:.1f} min) -- forcing exit")
                try:
                    save_json("stage_deadline_exceeded.json",
                              {"ok": False, "elapsed_min": round(elapsed, 1),
                               "state": STATE})
                except Exception:
                    pass
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            log(f"heartbeat t={elapsed:.1f} min state={STATE}")
    th = threading.Thread(target=_loop, name="watchdog", daemon=True)
    th.start()
    log(f"watchdog armed: deadline {GLOBAL_DEADLINE_MIN:.0f} min, subprocess "
        f"timeout {SUB_TIMEOUT_S}s, loader timeout {LOADER_TIMEOUT_S}s, "
        f"workers {NUM_WORKERS}")
    return th


def _define_photo_only():
    """Define once a PICKLABLE CropGuardDataset subclass forcing flip=False,
    scale=1.0 (PLAN B = photometric augmentation only).

    A closure monkey-patched onto the instance is not picklable, and Python
    3.14's forkserver DataLoader workers pickle the dataset. Called at module
    import (succeeds in worker children, where multiprocessing has already
    restored sys.path) and again from build_train_ds().
    """
    cls = globals().get("_PhotoOnlyDataset")
    if cls is not None:
        return cls
    try:
        from dataset import CropGuardDataset
    except Exception:
        return None

    class _PhotoOnlyDataset(CropGuardDataset):
        def _augment(self, img, boxes, lesion_mask=None, leaf_mask=None,
                     combined_mask=None, flip=None, scale=None):
            # photometric (HSV) augmentation still runs inside the parent
            # implementation; geometry is forced to identity.
            return CropGuardDataset._augment(self, img, boxes, lesion_mask,
                                             leaf_mask, combined_mask,
                                             flip=False, scale=1.0)

    # without this, pickle sees "<locals>" in the qualname and refuses
    _PhotoOnlyDataset.__qualname__ = "_PhotoOnlyDataset"
    globals()["_PhotoOnlyDataset"] = _PhotoOnlyDataset
    return _PhotoOnlyDataset


_define_photo_only()


# --------------------------------------------------------------- discovery --
def setup_code():
    """Make `import dataset` and `import model.architecture.*` both work.

    The published dataset is unzipped at the mount ROOT (training/,
    architecture/, __init__.py), while the code imports `model.architecture.x`
    absolutely. A `model` symlink inside /kaggle/working/code restores the
    package path without copying anything.
    """
    root = None
    for c in CODE_ROOTS:
        if os.path.isfile(os.path.join(c, "training", "dataset.py")):
            root = c
            break
    if root is None:
        raise RuntimeError(f"v8 code dataset not found in {CODE_ROOTS}")
    cwd_code = os.path.join(WORK, "code")
    os.makedirs(cwd_code, exist_ok=True)
    link = os.path.join(cwd_code, "model")
    if not os.path.exists(link):
        os.symlink(root, link)
    sys.path.insert(0, cwd_code)
    sys.path.insert(0, os.path.join(root, "training"))
    log(f"code root: {root}  (model -> {link})")

    # leaf_exclusion_list.txt is searched by dataset.py in cwd / repo root /
    # dataset dir. Copy it to every place it looks so the exclusions apply.
    src = None
    for c in INPUT_ROOTS + [root]:
        p = os.path.join(c, "code", "leaf_exclusion_list.txt")
        if os.path.isfile(p):
            src = p
            break
    if src:
        with open(src) as f:
            n = sum(1 for line in f if line.strip())
        for dst in (os.path.join(cwd_code, "leaf_exclusion_list.txt"),
                    os.path.join(root, "training", "leaf_exclusion_list.txt")):
            try:
                with open(src) as f, open(dst, "w") as g:
                    g.write(f.read())
            except OSError as e:
                log(f"  could not copy exclusion list to {dst}: {e}")
        log(f"leaf_exclusion_list.txt: {n} stems ({src})")
    else:
        log("leaf_exclusion_list.txt: NOT FOUND (no severity/leaf exclusions)")
    os.chdir(cwd_code)
    return root


def find_data_root(skip=None):
    """Directory holding images/<split>, labels/<split>, masks_lesion/<split>."""
    def ok(d):
        return all(os.path.isdir(os.path.join(d, k, "train"))
                   for k in ("images", "labels"))
    for base in INPUT_ROOTS:
        if skip is not None and os.path.abspath(base) == os.path.abspath(skip):
            continue
        for depth, (dp, dns, _fns) in enumerate(os.walk(base)):
            if dp.count(os.sep) - base.count(os.sep) > 4:
                dns[:] = []
                continue
            if ok(dp):
                return dp
    return None


def find_filled_root():
    """Directory holding masks_lesion/<split>/*.png (the filled-mask dataset)."""
    def ok(d):
        return os.path.isdir(os.path.join(d, "masks_lesion", "train"))
    for base in FILLED_ROOTS:
        for depth, (dp, dns, _fns) in enumerate(os.walk(base)):
            if dp.count(os.sep) - base.count(os.sep) > 2:
                dns[:] = []
                continue
            if ok(dp):
                return dp
    return None


def prepare_corpus():
    """v9: build WORK/corpus_v9 = v7 corpus with BOTH mask types repaired:

      masks_lesion -> filled polygon regions (were 1-px outlines)
      masks_leaf   -> fill_holes(leaf) | lesion (was healthy-tissue-only, so
                      necrosis sat OUTSIDE the leaf mask and the severity
                      denominator was wrong; see repair_masks_v9.py)

    corpus_v9 is searched FIRST by find_data_root(), so stage0 gates, stage1,
    stage2 and stage3 all read the corrected targets. Any failure raises, which
    main() catches before burning GPU quota.
    """
    if not os.path.isdir(os.path.join(V9_CORPUS, "images", "train")):
        v7 = find_data_root(skip=V9_CORPUS)
        if v7 is None:
            raise RuntimeError(f"v7 corpus not found under {INPUT_ROOTS}")
        filled = find_filled_root()
        if filled is None:
            raise RuntimeError(f"repaired-mask dataset not found under {FILLED_ROOTS}")
        os.makedirs(V9_CORPUS, exist_ok=True)
        for d in ("images", "labels"):
            os.symlink(os.path.join(v7, d), os.path.join(V9_CORPUS, d))
        for d in ("masks_lesion", "masks_leaf"):
            os.symlink(os.path.join(filled, d), os.path.join(V9_CORPUS, d))
        log(f"v9 corpus built: {V9_CORPUS} (v7={v7}, repaired={filled})")
    else:
        v7 = find_data_root(skip=V9_CORPUS)
        log(f"v9 corpus already present: {V9_CORPUS} (v7 for fingerprints={v7})")

    import numpy as np
    from PIL import Image
    counts = {d: {s: len(os.listdir(os.path.join(V9_CORPUS, d, s)))
                  for s in ("train", "val", "test")}
              for d in ("masks_lesion", "masks_leaf")}
    log(f"mask counts train/val/test: lesion="
        f"{counts['masks_lesion']['train']}/{counts['masks_lesion']['val']}/"
        f"{counts['masks_lesion']['test']} leaf="
        f"{counts['masks_leaf']['train']}/{counts['masks_leaf']['val']}/"
        f"{counts['masks_leaf']['test']}")
    for d in ("masks_lesion", "masks_leaf"):
        if counts[d] != {"train": 715, "val": 156, "test": 150}:
            raise RuntimeError(f"{d} counts {counts[d]} != 715/156/150")
    missing = 0
    for split in ("train", "val", "test"):
        idir = os.path.join(V9_CORPUS, "images", split)
        for fn in os.listdir(idir):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                for d in ("masks_lesion", "masks_leaf"):
                    if not os.path.exists(os.path.join(V9_CORPUS, d, split,
                                                       stem + ".png")):
                        missing += 1
    if missing:
        raise RuntimeError(f"{missing} masks missing (legacy fallback would kick in)")

    ratios, inside_fracs = [], []
    for split in ("train", "val", "test"):
        fns = sorted(os.listdir(os.path.join(V9_CORPUS, "masks_lesion", split)))
        for fn in fns[:: max(len(fns) // 10, 1)]:
            les9 = np.array(Image.open(
                os.path.join(V9_CORPUS, "masks_lesion", split, fn))) > 127
            leaf9 = np.array(Image.open(
                os.path.join(V9_CORPUS, "masks_leaf", split, fn))) > 127
            if les9.sum():
                inside_fracs.append(float((les9 & leaf9).sum()) / les9.sum())
            p7 = os.path.join(v7, "masks_lesion", split, fn) if v7 else None
            if p7 and os.path.exists(p7):
                a = int((np.array(Image.open(p7)) > 127).sum())
                ratios.append(int(les9.sum()) / max(a, 1))
    med = float(np.median(ratios)) if ratios else float("nan")
    ins_min = float(np.min(inside_fracs)) if inside_fracs else float("nan")
    ins_mean = float(np.mean(inside_fracs)) if inside_fracs else float("nan")
    log(f"overlay fingerprints over {len(ratios)} sampled masks: "
        f"lesion filled/outline median={med:.1f}x ; "
        f"lesion-inside-leaf mean={ins_mean:.4f} min={ins_min:.4f}")
    if not (med >= 5.0):
        raise RuntimeError(f"lesion fill overlay did NOT take effect ({med:.1f}x)")
    if not (ins_min >= 0.99):
        raise RuntimeError(f"leaf repair missing: lesion not inside leaf "
                           f"(min {ins_min:.4f})")
    STATE["corpus_v9"] = {"v7": v7, "counts": counts,
                          "overlay_median_ratio": med,
                          "lesion_inside_leaf_min": ins_min}
    return V9_CORPUS


# ------------------------------------------------------------------- utils --
def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Inline EMA (train_full_v7's EMA class is deliberately not imported)."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.checksum = 0.0

    def update(self, model):
        import torch
        with torch.no_grad():
            d = self.decay
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            # cheap fingerprint so we can PROVE the EMA actually moved
            first = next(iter(self.shadow.items()))[1]
            self.checksum = float(first.float().sum().item())

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))

    def swap_in(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k in self.shadow}
        self.copy_to(model)

        def restore():
            sd = model.state_dict()
            for k, v in backup.items():
                sd[k].copy_(v)
        return restore


def cosine_lr(step, total, warmup, base_lr, min_lr=1e-6):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * p))


def macro_f1(y_true, y_pred, nc=NC):
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    f1s, per_class = [], {}
    for c in range(nc):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        den = 2 * tp + fp + fn
        f1 = (2 * tp / den) if den > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": int(np.sum(y_true == c))}
        if per_class[c]["support"] > 0:
            f1s.append(f1)
    return (float(sum(f1s) / len(f1s)) if f1s else 0.0), per_class


# ------------------------------------------------- label reconciliation -----
def audit_labels(root):
    """Per-split class counts from YOLO labels + the stems to exclude.

    683/151/140 of 715/156/150 label files carry a class in 0-7; the rest must
    be excluded from training AND metrics rather than silently becoming
    background.
    """
    from collections import Counter
    report = {}
    exclusions = {}
    for split in ("train", "val", "test"):
        ldir = os.path.join(root, "labels", split)
        idir = os.path.join(root, "images", split)
        counts = Counter()
        n_files = 0
        bad = []
        for fn in sorted(os.listdir(ldir)) if os.path.isdir(ldir) else []:
            if not fn.endswith(".txt"):
                continue
            n_files += 1
            stem = fn[:-4]
            classes = set()
            with open(os.path.join(ldir, fn)) as f:
                for line in f:
                    p = line.split()
                    if len(p) < 5:
                        continue
                    try:
                        classes.add(int(float(p[0])))
                    except ValueError:
                        pass
            valid = {c for c in classes if 0 <= c < NC}
            for c in valid:
                counts[c] += 1
            if not valid:
                why = "empty label file" if not classes else \
                      f"class(es) outside 0-{NC - 1}: {sorted(classes)}"
                bad.append((stem, why))
        n_img = len([f for f in os.listdir(idir)
                     if os.path.splitext(f)[1].lower() in
                     (".jpg", ".jpeg", ".png", ".bmp", ".webp")]) \
            if os.path.isdir(idir) else 0
        exclusions[split] = {s for s, _ in bad}
        report[split] = {
            "images": n_img,
            "label_files": n_files,
            "sum_class_counts": sum(counts.values()),
            "per_class": {str(k): counts.get(k, 0) for k in range(NC)},
            "excluded": len(bad),
            "excluded_examples": bad[:15],
        }
        log(f"  {split}: images={n_img} label_files={n_files} "
            f"sum_class_counts={sum(counts.values())} excluded={len(bad)}")
    save_json("stage0_label_audit.json", report)
    return report, exclusions


def check_sha256():
    import dataset
    p = dataset.__file__
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    size = os.path.getsize(p)
    log(f"dataset.py {p} size={size} sha256={h}")
    if h != EXPECTED_DATASET_SHA256:
        raise RuntimeError(
            f"dataset.py is NOT the fixed copy (expected "
            f"{EXPECTED_DATASET_SHA256}, got {h}) -- aborting")
    log("sha256 gate PASS (this is the fixed dataset.py)")


def check_geometry(root):
    """Re-run the same CPU test that passed locally, here on Kaggle."""
    test = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "test_dataset_aug.py")
    if not os.path.isfile(test):
        code_root = None
        for c in CODE_ROOTS:
            cand = os.path.join(c, "training", "test_dataset_aug.py")
            if os.path.isfile(cand):
                code_root = cand
                break
        test = code_root
    if not test or not os.path.isfile(test):
        raise RuntimeError("test_dataset_aug.py not found (cannot gate geometry)")
    try:
        r = subprocess.run([sys.executable, test, root], capture_output=True,
                           text=True, cwd=os.path.dirname(test),
                           timeout=SUB_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"forced-geometry test hung: no exit after {SUB_TIMEOUT_S}s "
            f"(this is a stage-0 hang) -- aborting instead of blocking")
    log("--- forced-geometry test output ---")
    print(r.stdout, flush=True)
    if r.stderr.strip():
        print(r.stderr, flush=True)
    log(f"--- exit code {r.returncode} ---")
    if r.returncode != 0:
        raise RuntimeError("forced-geometry test FAILED on Kaggle")
    log("geometry gate PASS")


def build_model(device, train=True):
    """Single place that builds the model, honouring the Stage-0 fallback."""
    from model.architecture.v7_model import CropGuardMultiTaskV7
    m = CropGuardMultiTaskV7(nc=NC, pretrained=PRETRAINED_OK).to(device)
    return m.train() if train else m.eval()


def check_pretrained():
    global PRETRAINED_OK
    import timm
    from model.architecture.v7_model import CropGuardMultiTaskV7
    try:
        m = CropGuardMultiTaskV7(nc=NC, pretrained=True)
        mode = "pretrained"
        PRETRAINED_OK = True
    except Exception as e:  # noqa: BLE001
        log("!" * 70)
        log(f"LOUD WARNING: pretrained fasternet_t0 download FAILED ({e}).")
        log("LOUD WARNING: continuing from RANDOM INIT as the brief requires, "
            "but numbers from this run are NOT comparable to a pretrained run.")
        log("!" * 70)
        m = CropGuardMultiTaskV7(nc=NC, pretrained=False)
        mode = "random"
        PRETRAINED_OK = False
    ref = timm.create_model("fasternet_t0", pretrained=False,
                            features_only=True, out_indices=(0, 1, 2, 3))
    a = next(m.backbone.net.parameters()).detach().float().cpu()
    b = next(ref.parameters()).detach().float().cpu()
    l1 = float((a - b).abs().mean().item())
    log(f"pretrained check: mode={mode} first-conv L1 vs random init = {l1:.6f} "
        f"({'>' if l1 > 0.01 else '<='} 0.01)")
    try:
        log(f"timm cache dir: {os.path.join(os.path.expanduser('~'), '.cache/torch/hub')} "
            f"exists={os.path.isdir(os.path.join(os.path.expanduser('~'), '.cache/torch/hub'))}")
    except Exception:
        pass
    if l1 <= 0.01:
        log("!" * 70)
        log("LOUD WARNING: backbone weights look like RANDOM INIT (L1 <= 0.01).")
        log("!" * 70)
    del ref
    return mode, l1


# ------------------------------------------------------------------ stage 2 --
def loader_for(ds, shuffle=False, sampler=None, workers=None):
    import torch
    from torch.utils.data import DataLoader
    from dataset import collate
    if workers is None:
        workers = NUM_WORKERS
    extra = {}
    if workers > 0:
        # A worker that dies (OOM) or stalls used to block the main thread
        # forever with no log line (the top suspect for the v9 overrun).
        # timeout= turns that into a catchable RuntimeError; persistent_workers
        # =False recycles workers each epoch so one bad worker cannot poison the
        # rest of the run.
        extra["timeout"] = LOADER_TIMEOUT_S
        extra["persistent_workers"] = False
    return DataLoader(ds, batch_size=BATCH,
                      shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=workers, pin_memory=True,
                      collate_fn=collate, drop_last=False, **extra)


def make_sampler(ds):
    import math as _m
    from torch.utils.data import WeightedRandomSampler
    from dataset import load_yolo_labels
    labels, counts = [], {}
    for img_path, lbl_path in ds.samples:
        boxes = load_yolo_labels(lbl_path)
        c = int(np_bincount_argmax([b[0] for b in boxes])) if boxes else -1
        labels.append(c)
        counts[c] = counts.get(c, 0) + 1
    med = sorted(counts.values())[len(counts) // 2] if counts else 1
    w = [1.0 / _m.sqrt(counts.get(c, med)) for c in labels]
    return WeightedRandomSampler(w, num_samples=len(ds), replacement=True), counts


def np_bincount_argmax(vals):
    import numpy as np
    return int(np.bincount([int(v) for v in vals], minlength=NC + 3).argmax())


def class_weights(counts_np, nc=NC):
    """Mild inverse-sqrt class weights (capped 3x) -- 'mild', per the brief."""
    import numpy as np
    w = np.ones(nc, dtype="float32")
    present = [c for c in range(nc) if counts_np[c] > 0]
    if present:
        med = float(np.median([counts_np[c] for c in present]))
        for c in range(nc):
            n = counts_np[c]
            if n > 0:
                w[c] = min(3.0, (med / n) ** 0.5)
            else:
                w[c] = 1.0
    return w


def evaluate(model, loader, device, tag=""):
    """Val metrics. Lesion IoU/Dice at the best val threshold."""
    import numpy as np
    import torch
    model.eval()
    P, T = [], []
    probs, gts = [], []
    sev_pred, sev_true, sev_trueleaf = [], [], []
    pred_area, true_area = [], []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(img, active_tasks={"cls", "seg"}, with_analysis=True)
            logits = out["cls_logits"].float()
            P.extend(logits.argmax(-1).cpu().numpy().tolist())
            T.extend(batch["cls"].numpy().tolist())
            p_les = torch.sigmoid(out["seg_logits"][:, 0:1].float()).cpu()
            p_leaf = torch.sigmoid(out["seg_logits"][:, 1:2].float()).cpu()
            g_les = batch["mask"][:, None].float().cpu()
            g_leaf = batch["leaf"][:, None].float().cpu()
            probs.append(p_les)
            gts.append(g_les)
            pred_area.append(p_les.sum(dim=(1, 2, 3)).numpy())
            true_area.append(g_les.sum(dim=(1, 2, 3)).numpy())
            # analytic severity: denominator is the PREDICTED leaf area
            pl = p_les.sum(dim=(1, 2, 3))
            lf = p_leaf.sum(dim=(1, 2, 3)).clamp(min=1e-3)
            gl = g_leaf.sum(dim=(1, 2, 3)).clamp(min=1e-3)
            gt_ratio = (g_les.sum(dim=(1, 2, 3)) / gl).clamp(0, 1)
            sev_pred.append((pl / lf).clamp(0, 1).numpy())
            sev_trueleaf.append((pl / gl).clamp(0, 1).numpy())
            sev_true.append(batch["sev_cont"].numpy())
    mask_valid = np.array(T) >= 0
    P = np.array(P)[mask_valid]
    T = np.array(T)[mask_valid]
    acc = float((P == T).mean()) if len(T) else float("nan")
    f1, per_class = macro_f1(T, P)
    probs = torch.cat(probs)
    gts = torch.cat(gts)
    best = (0.0, 0.0, 0.5)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        pb = (probs > thr).float()
        inter = float((pb * gts).sum())
        union = float(((pb + gts) > 0).sum())
        iou = inter / max(union, 1.0)
        dice = 2 * inter / max(float(pb.sum()) + float(gts.sum()), 1.0)
        if iou > best[0]:
            best = (iou, dice, thr)
    sp = np.concatenate(sev_pred)
    stl = np.concatenate(sev_trueleaf)
    st = np.concatenate(sev_true)
    v = st >= 0.0
    m = {}
    m["acc"] = acc
    m["macro_f1"] = f1
    m["per_class"] = per_class
    m["lesion_iou"] = best[0]
    m["lesion_dice"] = best[1]
    m["best_thr"] = best[2]
    m["sev_mae_predmasks"] = float(np.abs(sp[v] - st[v]).mean()) if v.any() else float("nan")
    m["sev_mae_trueleaf"] = float(np.abs(stl[v] - st[v]).mean()) if v.any() else float("nan")
    m["mean_pred_lesion_area"] = float(np.concatenate(pred_area).mean())
    m["mean_true_lesion_area"] = float(np.concatenate(true_area).mean())
    m["over_pred_ratio"] = (m["mean_pred_lesion_area"] /
                            max(m["mean_true_lesion_area"], 1e-9))
    m["n_valid_cls"] = int(len(T))
    m["n_sev"] = int(v.sum())
    m["tag"] = tag
    model.train()
    return m


def fmt(m):
    return (f"acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"iou={m['lesion_iou']:.4f} dice={m['lesion_dice']:.4f} "
            f"sevMAE(pred)={m['sev_mae_predmasks']:.4f} "
            f"sevMAE(trueleaf)={m['sev_mae_trueleaf']:.4f} "
            f"pred_area={m['mean_pred_lesion_area']:.1f} "
            f"true_area={m['mean_true_lesion_area']:.1f} "
            f"over={m['over_pred_ratio']:.2f}x")


# ------------------------------------------------------------------- stages --
def stage0(root):
    import torch
    check_sha256()
    check_geometry(root)
    mode, l1 = check_pretrained()
    report, exclusions = audit_labels(root)
    summary = {"sha256_ok": True, "geometry_ok": True, "pretrained_mode": mode,
               "pretrained_l1": l1, "labels": report}
    save_json("stage0_summary.json", summary)
    return exclusions


def build_train_ds(root, exclusions, geometric=True, n_images=None):
    """Train dataset with excluded stems removed and optional photometric-only aug."""
    from dataset import CropGuardDataset
    ds = CropGuardDataset(root, "train", SIZE, augment=True)
    kept = [s for s in ds.samples
            if os.path.splitext(os.path.basename(s[0]))[0] not in exclusions]
    dropped = len(ds.samples) - len(kept)
    ds.samples = kept
    if TRAIN_SUBSET:
        ds.samples = ds.samples[:TRAIN_SUBSET]
    log(f"train: {len(ds.samples)} images ({dropped} dropped by label audit)")
    if not geometric:
        ds.__class__ = _define_photo_only()
        log("train: GEOMETRIC AUG DISABLED (flip=False, scale=1.0)")
    if n_images:
        ds.samples = ds.samples[:n_images]
        log(f"train: subset to {len(ds.samples)} images")
    return ds


def stage1(root, exclusions, device):
    """Memorization gate: 8 images, 500 steps, aug ON."""
    import numpy as np
    import torch
    from losses import seg_loss_v7

    def run(geometric, steps=None, every=100):
        steps = steps or STAGE1_STEPS
        seed_all(SEED)
        ds = build_train_ds(root, exclusions, geometric=geometric, n_images=8)
        dl = loader_for(ds, shuffle=True, workers=0)
        model = build_model(device, train=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        it = iter(dl)
        hist = []
        for step in range(1, steps + 1):
            if (time.time() - _T0) / 60.0 > GLOBAL_DEADLINE_MIN:
                raise RuntimeError("global deadline exceeded in stage1")
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(batch["image"], active_tasks={"cls", "seg"})
                seg = seg_loss_v7(out["seg_logits"], batch["mask"], batch["leaf"])
                valid = batch["cls"] >= 0
                cls = (torch.nn.functional.cross_entropy(out["cls_logits"][valid],
                                                        batch["cls"][valid],
                                                        label_smoothing=0.05)
                       if valid.any() else out["cls_logits"].sum() * 0.0)
                loss = cls + seg
            if not torch.isfinite(loss):
                log(f"  step {step}: NON-FINITE loss, aborting this run")
                return None, hist, None
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorms = {
                "seg_head": float(sum(p.grad.norm().item() ** 2
                                      for p in model.seg_head.parameters()
                                      if p.grad is not None) ** 0.5),
                "neck": float(sum(p.grad.norm().item() ** 2
                                  for p in model.neck.parameters()
                                  if p.grad is not None) ** 0.5),
                "backbone": float(sum(p.grad.norm().item() ** 2
                                      for p in model.backbone.parameters()
                                      if p.grad is not None) ** 0.5),
            }
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            if step % max(every, 1) == 0 or step == steps:
                with torch.no_grad():
                    p = torch.sigmoid(out["seg_logits"][:, 0:1].float())
                    pb = (p > 0.5).float()
                    g = batch["mask"][:, None].float()
                    inter = float((pb * g).sum())
                    union = float(((pb + g) > 0).sum())
                    iou = inter / max(union, 1.0)
                hist.append({"step": step, "iou": iou, "cls": float(cls.item()),
                             "seg": float(seg.item()), "total": float(loss.item()),
                             "gnorms": gnorms})
                log(f"  step {step:4d}: lesion IoU={iou:.4f} cls={cls.item():.4f} "
                    f"seg={seg.item():.4f} total={loss.item():.4f} "
                    f"grad_norm(seg/neck/bb)={gnorms['seg_head']:.2e}/"
                    f"{gnorms['neck']:.2e}/{gnorms['backbone']:.2e}")
        return model, hist, ds

    log("=" * 70)
    log(f"STAGE 1: memorization gate (8 images, {STAGE1_STEPS} steps, augmentation ON)")
    log(f"Baseline constant-severity MAE to beat on val (filled corpus): "
        f"{V9_BASELINE_MAE:.4f}")
    model, hist, ds = run(geometric=True)
    save_json("stage1_memorize.json", hist)
    gate_steps = STAGE1_STEPS
    # V8_FORCE_AUG bypasses the (500-step) gate verdict. It exists ONLY for the
    # laptop smoke test, which cannot afford 500 steps; Kaggle never sets it.
    passed = bool(hist and len(hist) and gate_steps >= 500
                  and hist[-1]["iou"] >= 0.5)
    if os.environ.get("V8_FORCE_AUG"):
        log(f"V8_FORCE_AUG={os.environ['V8_FORCE_AUG']} overrides the gate verdict")
        return os.environ["V8_FORCE_AUG"], hist
    log(f"STAGE 1 aug-ON gate: {'PASS' if passed else 'FAIL'} "
        f"(IoU at step 500 = {hist[-1]['iou'] if hist else float('nan'):.4f}, need >= 0.5)")

    if passed:
        return "geo", hist

    # ---- diagnostics, as the brief requires ----
    log("-" * 70)
    log("STAGE 1 FAILED: printing diagnostics")
    if model is not None:
        opt_params = sum(p.numel() for p in
                         torch.optim.AdamW(model.parameters()).param_groups[0]["params"])
        log(f"optimizer would see {opt_params} params")
        for name, mod in (("backbone", model.backbone), ("neck", model.neck),
                          ("det_head", model.det_head), ("cls_head", model.cls_head),
                          ("seg_head", model.seg_head)):
            n_all = sum(1 for _ in mod.parameters())
            n_req = sum(1 for p in mod.parameters() if p.requires_grad)
            log(f"  requires_grad {name}: {n_req}/{n_all} tensors trainable")
        last = hist[-1] if hist else {}
        log(f"  last step components: {last}")
    log("-" * 70)
    log("STAGE 1 retry with GEOMETRIC AUGMENTATION OFF (photometric only)")
    model2, hist2, _ = run(geometric=False)
    save_json("stage1_memorize_nogeo.json", hist2)
    passed2 = bool(hist2 and len(hist2) and gate_steps >= 500
                   and hist2[-1]["iou"] >= 0.5)
    log(f"STAGE 1 aug-OFF gate: {'PASS' if passed2 else 'FAIL'} "
        f"(IoU at step 500 = {hist2[-1]['iou'] if hist2 else float('nan'):.4f})")
    if passed2:
        log("PLAN B: training will use PHOTOMETRIC augmentation only")
        return "photo", hist2
    log("STAGE 1 FAILED BOTH WAYS -> STOPPING before main training.")
    return "stop", hist2


def stage2(root, exclusions, device, aug_mode):
    import numpy as np
    import torch
    from dataset import CropGuardDataset, load_yolo_labels
    from losses import seg_loss_v7

    t_start = time.time()
    geometric = aug_mode == "geo"
    train_ds = build_train_ds(root, exclusions, geometric=geometric)
    val_ds = CropGuardDataset(root, "val", SIZE, augment=False)
    val_kept = [s for s in val_ds.samples
                if os.path.splitext(os.path.basename(s[0]))[0] not in exclusions["val"]]
    if VAL_SUBSET:
        val_kept = val_kept[:VAL_SUBSET]
    val_ds.samples = val_kept
    log(f"val: {len(val_kept)} images ({len(exclusions['val'])} excluded by audit)")

    # class weights from the (filtered) train labels
    counts = np.zeros(NC, dtype="float64")
    for img_path, lbl_path in train_ds.samples:
        for c, *_ in load_yolo_labels(lbl_path):
            if 0 <= int(c) < NC:
                counts[int(c)] += 1
    cw = class_weights(counts)
    log(f"train class counts={counts.astype(int).tolist()}")
    log(f"class weights={[round(float(x), 3) for x in cw]}")
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    sampler, counts_all = make_sampler(train_ds)
    train_loader = loader_for(train_ds, sampler=sampler, workers=NUM_WORKERS)
    val_loader = loader_for(val_ds, shuffle=False, workers=NUM_WORKERS)

    seed_all(SEED)
    model = build_model(device, train=True)
    ema = EMA(model, decay=0.999)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    opt_names = []
    ids = {id(p) for p in opt.param_groups[0]["params"]}
    for name, mod in (("backbone", model.backbone), ("neck", model.neck),
                      ("seg_head", model.seg_head), ("cls_head", model.cls_head),
                      ("det_head", model.det_head)):
        inside = sum(1 for p in mod.parameters() if id(p) in ids)
        n = sum(1 for _ in mod.parameters())
        opt_names.append(f"{name}:{inside}/{n}")
        if name in ("backbone", "neck", "seg_head") and inside != n:
            raise RuntimeError(f"{name} is NOT fully in the optimizer ({inside}/{n})")
    log(f"optimizer coverage: {' '.join(opt_names)}  (asserted backbone/neck/seg_head)")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    warmup = 5 * steps_per_epoch
    csv_path = os.path.join(WORK, "v8_train_log.csv")
    csv_f = open(csv_path, "w", newline="")
    wr = csv.writer(csv_f)
    wr.writerow(["epoch", "lr", "cls", "seg", "total", "sec",
                 "raw_acc", "raw_f1", "raw_iou", "raw_dice",
                 "raw_sevMAE_pred", "raw_sevMAE_trueleaf",
                 "raw_pred_area", "raw_true_area",
                 "ema_acc", "ema_f1", "ema_iou", "ema_dice",
                 "ema_sevMAE_pred", "ema_sevMAE_trueleaf",
                 "ema_pred_area", "ema_true_area", "ema_checksum"])
    csv_f.flush()

    best = {"score": -1.0, "iou": -1.0, "epoch": 0, "which": None, "metrics": None}
    prev_ema_checksum = None
    epochs_done = 0
    for epoch in range(1, EPOCHS + 1):
        if (time.time() - _T0) / 60.0 > GLOBAL_DEADLINE_MIN:
            log("GLOBAL DEADLINE hit inside stage2 -- stopping cleanly")
            break
        elapsed_min = (time.time() - t_start) / 60.0
        per_epoch = elapsed_min / max(epochs_done, 1) if epochs_done else None
        if per_epoch and elapsed_min + per_epoch > CAP_MINUTES:
            log(f"TIME CAP: {elapsed_min:.1f} min used, ~{per_epoch:.2f} min/epoch, "
                f"cap {CAP_MINUTES} min -> stopping cleanly at epoch {epoch - 1}")
            break
        t0 = time.time()
        model.train()
        sums = {"cls": 0.0, "seg": 0.0, "total": 0.0}
        gt_mask_prob = 0.8 - 0.6 * ((epoch - 1) / max(EPOCHS - 1, 1))
        for it, batch in enumerate(train_loader):
            step = (epoch - 1) * steps_per_epoch + it
            if it and it % 200 == 0:
                log(f"  E{epoch} step {it}/{steps_per_epoch} "
                    f"({time.time() - t0:.0f}s into epoch)")
            lr = cosine_lr(step, total_steps, warmup, LR)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(batch["image"], active_tasks={"cls", "seg"},
                            gt_mask=batch.get("mask"), gt_mask_prob=gt_mask_prob)
                valid = batch["cls"] >= 0
                cls_l = (torch.nn.functional.cross_entropy(
                    out["cls_logits"][valid], batch["cls"][valid],
                    weight=cw_t, label_smoothing=0.05)
                    if valid.any() else out["cls_logits"].sum() * 0.0)
                seg_l = seg_loss_v7(out["seg_logits"], batch["mask"], batch["leaf"])
                total = cls_l + seg_l
            if not torch.isfinite(total):
                log(f"  epoch {epoch} step {it}: non-finite loss; skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            sums["cls"] += float(cls_l.item())
            sums["seg"] += float(seg_l.item())
            sums["total"] += float(total.item())
        dt = time.time() - t0
        n = max(steps_per_epoch, 1)
        ra = evaluate(model, val_loader, device, tag="raw")
        restore = ema.swap_in(model)
        em = evaluate(model, val_loader, device, tag="ema")
        restore()
        model.train()
        ema_checksum = ema.checksum
        ema_moved = (prev_ema_checksum is None or
                     abs(ema_checksum - prev_ema_checksum) > 0)
        if prev_ema_checksum is not None and not ema_moved:
            log(f"  WARNING: EMA params did NOT change between epochs "
                f"(checksum {ema_checksum}); investigate stagnation")
        prev_ema_checksum = ema_checksum
        cur_lr = opt.param_groups[0]["lr"]
        wr.writerow([epoch, f"{cur_lr:.3e}", f"{sums['cls'] / n:.4f}",
                     f"{sums['seg'] / n:.4f}", f"{sums['total'] / n:.4f}",
                     f"{dt:.1f}",
                     f"{ra['acc']:.4f}", f"{ra['macro_f1']:.4f}",
                     f"{ra['lesion_iou']:.4f}", f"{ra['lesion_dice']:.4f}",
                     f"{ra['sev_mae_predmasks']:.4f}",
                     f"{ra['sev_mae_trueleaf']:.4f}",
                     f"{ra['mean_pred_lesion_area']:.1f}",
                     f"{ra['mean_true_lesion_area']:.1f}",
                     f"{em['acc']:.4f}", f"{em['macro_f1']:.4f}",
                     f"{em['lesion_iou']:.4f}", f"{em['lesion_dice']:.4f}",
                     f"{em['sev_mae_predmasks']:.4f}",
                     f"{em['sev_mae_trueleaf']:.4f}",
                     f"{em['mean_pred_lesion_area']:.1f}",
                     f"{em['mean_true_lesion_area']:.1f}",
                     f"{ema_checksum:.6f}"])
        csv_f.flush()
        epochs_done += 1
        log(f"E{epoch:3d}/{EPOCHS} [{dt:.0f}s lr={cur_lr:.2e}] "
            f"cls={sums['cls'] / n:.4f} seg={sums['seg'] / n:.4f} | "
            f"RAW {fmt(ra)} | EMA {fmt(em)} | ema_moved={ema_moved}")
        for which, m in (("raw", ra), ("ema", em)):
            score, tie = m["macro_f1"], m["lesion_iou"]
            if score > best["score"] or (score == best["score"] and tie > best["iou"]):
                best = {"score": score, "iou": tie, "epoch": epoch,
                        "which": which, "metrics": m}
                if which == "raw":
                    torch.save({"epoch": epoch, "model": model.state_dict(),
                                "metrics": m, "which": which},
                               os.path.join(WORK, "best_v8_raw.pt"))
                else:
                    restore3 = ema.swap_in(model)
                    torch.save({"epoch": epoch, "model": model.state_dict(),
                                "metrics": m, "which": which},
                               os.path.join(WORK, "best_v8_ema.pt"))
                    restore3()
                log(f"  --> new best ({which}) epoch {epoch} "
                    f"f1={score:.4f} iou={tie:.4f}")
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "ema_shadow": ema.shadow, "opt": opt.state_dict(),
                    "metrics": {"raw": ra, "ema": em}, "epochs_done": epochs_done},
                   os.path.join(WORK, "last_v8.pt"))

    csv_f.close()
    save_json("stage2_training.json", {"best": best, "epochs_done": epochs_done,
                                       "aug_mode": aug_mode,
                                       "minutes": (time.time() - t_start) / 60.0,
                                       "csv": csv_path,
                                       "class_weights": cw.tolist()})
    return best


def stage3(root, exclusions, device, best):
    """Val summary of the SELECTED checkpoint. Test split is never touched."""
    import torch
    from dataset import CropGuardDataset
    path = os.path.join(WORK, f"best_v8_{best['which']}.pt")
    val_ds = CropGuardDataset(root, "val", SIZE, augment=False)
    val_ds.samples = [s for s in val_ds.samples
                      if os.path.splitext(os.path.basename(s[0]))[0]
                      not in exclusions["val"]]
    if VAL_SUBSET:
        val_ds.samples = val_ds.samples[:VAL_SUBSET]
    val_loader = loader_for(val_ds, shuffle=False, workers=NUM_WORKERS)
    model = build_model(device, train=False)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    m = evaluate(model, val_loader, device, tag=f"selected:{best['which']}")
    m["checkpoint"] = path
    m["selected_epoch"] = best["epoch"]
    m["selected_by"] = "val macro-F1 (tie-break lesion IoU)"
    m["constant_severity_baseline_mae"] = V9_BASELINE_MAE
    m["constant_severity_baseline_protocol"] = "val split mean, filled-mask corpus"
    m["sev_beats_baseline"] = bool(m["sev_mae_predmasks"] < V9_BASELINE_MAE)
    save_json("stage_summary.json", m)
    log("STAGE 3 val summary:")
    log("  " + fmt(m))
    log(f"  per-class F1: " +
        ", ".join(f"{c}:{v['f1']:.3f}(n={v['support']})"
                  for c, v in sorted(m["per_class"].items())))
    return m


def main():
    t0 = time.time()
    import numpy as np  # noqa: F401  (used by helpers)
    import torch
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"python {sys.version.split()[0]} torch {torch.__version__} "
        f"cuda={torch.cuda.is_available()} device={device}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)}")
    try:
        import timm
        log(f"timm {timm.__version__}")
    except Exception as e:  # noqa: BLE001
        log(f"timm import failed: {e}")

    start_watchdog()

    root = None
    try:
        setup_code()
        prepare_corpus()
        root = find_data_root()
        log(f"data root: {root}")
        if root is None:
            raise RuntimeError(f"no corpus found under {INPUT_ROOTS}")
        STATE["root"] = root
    except Exception:
        log("SETUP FAILED:\n" + traceback.format_exc())
        save_json("stage_setup_failure.json", {"ok": False,
                                               "traceback": traceback.format_exc()})
        return

    exclusions = {"train": set(), "val": set(), "test": set()}
    try:
        exclusions = stage0(root)
        STATE["stage0"] = "ok"
    except Exception:
        log("STAGE 0 FAILED (aborting -- gates are not optional):\n"
            + traceback.format_exc())
        save_json("stage0_failure.json", {"ok": False,
                                          "traceback": traceback.format_exc()})
        return

    aug_mode = "stop"
    try:
        aug_mode, _hist = stage1(root, exclusions, device)
        STATE["stage1"] = aug_mode
    except Exception:
        log("STAGE 1 CRASHED:\n" + traceback.format_exc())
        save_json("stage1_failure.json", {"ok": False,
                                          "traceback": traceback.format_exc()})
        return

    if aug_mode == "stop":
        log("Main training NOT started (memorization gate failed). ")
        save_json("stage2_training.json", {"skipped": True, "reason": "stage1 gate failed"})
        return

    best = None
    try:
        best = stage2(root, exclusions, device, aug_mode)
        STATE["stage2"] = "ok"
    except Exception:
        log("STAGE 2 CRASHED:\n" + traceback.format_exc())
        save_json("stage2_failure.json", {"ok": False,
                                          "traceback": traceback.format_exc()})
        return

    if best is None or best["epoch"] == 0:
        log("No epoch completed; nothing to summarise.")
        return
    try:
        stage3(root, exclusions, device, best)
    except Exception:
        log("STAGE 3 CRASHED:\n" + traceback.format_exc())
        save_json("stage3_failure.json", {"ok": False,
                                          "traceback": traceback.format_exc()})
    log(f"TOTAL {((time.time() - t0) / 60.0):.1f} min; stage states: {STATE}")


if __name__ == "__main__":
    main()
