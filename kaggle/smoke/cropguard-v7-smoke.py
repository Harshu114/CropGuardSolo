"""CropGuard **v7** staged training on Kaggle GPU (script kernel).

Same corpus and protocol as the v6 run so the numbers are directly comparable:

    v6 baseline, stage-5 final VAL:
    det_cls 0.429 det_box 0.164 det_dfl 3.113 cls 0.285 seg 0.280 sev 0.065

Runs, in order:
  1. train.py --arch v7  (stages 1-5, 10 epochs each -- v6's schedule)
  2. eval_tta.py  --arch v7   -> metrics_v7.json (per-class F1, ECE included)
  3. calibrate_abstention.py --write   -> fitted temperature in the checkpoint
  4. export_onnx.py --arch v7          -> cropguard_v7.onnx + CPU latency

Steps 2-4 are isolated: if evaluation or export fails, the trained checkpoints in
/kaggle/working/runs are still saved and downloadable, which is the whole point
of a 5-hour session.

Layout note (inherited from the v6 kernel): a Kaggle dataset built from a
top-level folder is zipped/extracted at the mount root, so `model/architecture`
arrives as `architecture`. This locates train.py wherever it landed and rebuilds
a `model/` package layout matching the repo.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

IN = "/kaggle/input"


def find_input(name, must_contain=None):
    """Locate a mounted dataset regardless of Kaggle's mount layout.

    Classic: /kaggle/input/<slug>.  Newer runtimes: /kaggle/input/datasets/
    <owner>/<slug>. A hardcoded /kaggle/input/<slug> made the first smoke run
    fail with `got ['datasets']` before doing any work, so resolve instead of
    assume.
    """
    cands = [os.path.join(IN, name)] + sorted(
        glob.glob(os.path.join(IN, "**", name), recursive=True))
    for c in cands:
        if os.path.isdir(c) and (must_contain is None or
                                 os.path.exists(os.path.join(c, must_contain))):
            return c
    # last resort: any mounted dir that has the marker we need
    if must_contain:
        for c in sorted(glob.glob(os.path.join(IN, "**"), recursive=True)):
            if os.path.isdir(os.path.join(c, must_contain)) and len(c) < 90:
                return c
    return None


CODE_DS = find_input("cropguard-v7-code", "training")
DATA = find_input("cropguard-corpus-claude", "images")
print("input mount:", IN, sorted(os.listdir(IN))[:6], flush=True)
print("CODE_DS:", CODE_DS, "| DATA:", DATA, flush=True)

if CODE_DS is None:
    sys.exit(f"code dataset not found under {IN}; got {os.listdir(IN)}")
if DATA is None:
    sys.exit(f"corpus not found under {IN}; got {os.listdir(IN)}")

cands = glob.glob(os.path.join(CODE_DS, "**", "train.py"), recursive=True)
if not cands:
    sys.exit(f"train.py not found under {CODE_DS}; contents: {os.listdir(CODE_DS)}")
train_py = cands[0]
pkg_src = os.path.dirname(train_py)        # .../training
base = os.path.dirname(pkg_src)            # .../model, or the mount root if flat

# NB: do NOT name this "code" -- /kaggle/working/code shadows the stdlib `code`
# module and breaks nbconvert.
work = "/kaggle/working/cgcode"
shutil.rmtree(work, ignore_errors=True)

if os.path.basename(base) == "model":
    shutil.copytree(base, os.path.join(work, "model"))
else:
    os.makedirs(os.path.join(work, "model"), exist_ok=True)
    for sub in ("architecture", "training"):
        s = os.path.join(base, sub)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(work, "model", sub))

train_py = os.path.join(work, "model", "training", "train.py")
arch_dir = os.path.join(work, "model", "architecture")
need = ["model.py", "v7_model.py", "v7_heads.py"]
missing = [n for n in need if not os.path.exists(os.path.join(arch_dir, n))]
if missing:
    sys.exit(f"v7 code incomplete, missing {missing} in {arch_dir}")

out = "/kaggle/working/runs"
env = dict(os.environ)
env["PYTHONPATH"] = work
env["CROPGUARD_DATA"] = DATA
env["HF_HUB_DISABLE_TELEMETRY"] = "1"

stages = os.environ.get("STAGES", "1,3,4")
stage_epochs = os.environ.get("STAGE_EPOCHS", "1")   # v6 used 10

# NB: `nvidia-smi -l` is LOOP mode and never exits -- running it under
# subprocess.run() without a timeout hung the first smoke kernel for 30 min
# before any training started. `-L` (list) returns immediately; timeout is a
# belt-and-braces guard.
try:
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                         timeout=30).stdout.strip()
except Exception as e:  # noqa: BLE001
    smi = f"nvidia-smi failed: {type(e).__name__}: {e}"
print("GPU:", smi.replace("\n", " | ")[:300], flush=True)
print("DATA:", sorted(os.listdir(DATA))[:8], flush=True)

# ---------------------------------------------------------------- 1. train ---
cmd = [sys.executable, train_py, "--arch", "v7",
       "--root", DATA, "--out", out,
       "--stages", stages, "--stage-epochs", stage_epochs,
       "--batch-size", os.environ.get("BATCH_SIZE", "32"),
       "--size", "320", "--workers", "2", "--amp",
       "--ema", "--log-every", "20"]
# A smoke test must be FAST. Without a subset limit this ran full epochs over
# the whole corpus, so it spent many minutes per stage and looked hung while it
# was really just slow. 300 images is enough to exercise every head and write
# checkpoints in a couple of minutes.
cmd += ["--max-images", os.environ.get("SMOKE_IMAGES", "300")]
if os.environ.get("RESUME") == "1":
    cmd += ["--resume"]
print("TRAIN:", " ".join(cmd), flush=True)
rc = subprocess.call(cmd, env=env)
print("train.py exited with", rc, flush=True)
if rc != 0:
    sys.exit(rc)

ckpt = os.path.join(out, "student_best.pt")
if not os.path.exists(ckpt):
    sys.exit(f"no checkpoint at {ckpt}; got {os.listdir(out)}")

# ------------------------------------------------------------- 2. evaluate ---
# onnxruntime is NOT in the base Kaggle image. Without it export_onnx.py can
# still write the FP32 graph but cannot benchmark or quantize it
# (ModuleNotFoundError). Install it here so the INT8 step actually runs.
if os.environ.get("NO_ORT") != "1":
    print("installing onnxruntime + onnx for export/quantization...", flush=True)
    subprocess.call([sys.executable, "-m", "pip", "install", "-q",
                     "onnxruntime", "onnx"], env=env)
for name, args in (
    ("eval", ["--arch", "v7", "--ckpt", ckpt, "--root", DATA,
              "--split", "test", "--out", "/kaggle/working/metrics_v7.json"]),
    ("calibrate", ["--ckpt", ckpt, "--root", DATA, "--split", "val",
                   "--target-abstain", "0.05", "--write"]),
    ("export", ["--arch", "v7", "--ckpt", ckpt,
                "--out", "/kaggle/working/cropguard_v7.onnx",
                "--quant-mode", os.environ.get("QUANT_MODE", "both"),
                "--calib-root", DATA]),
):
    script = {"eval": "eval_tta.py", "calibrate": "calibrate_abstention.py",
              "export": "export_onnx.py"}[name]
    path = os.path.join(work, "model", "training", script)
    print(f"\n=== {name}: {script} ===", flush=True)
    try:
        subprocess.call([sys.executable, path] + args, env=env)
    except Exception as e:  # noqa: BLE001 - never lose trained weights to this
        print(f"{name} FAILED (checkpoints are safe): {type(e).__name__}: {e}",
              flush=True)

mj = "/kaggle/working/metrics_v7.json"
if os.path.exists(mj):
    m = json.load(open(mj))
    print("\n=== v7 vs v6 (stage-5 final VAL) ===", flush=True)
    print(f"  cls_top1   {m.get('cls_top1'):.3f}   macro_f1 {m.get('cls_macro_f1'):.3f}"
          f"   ECE {m.get('cls_ece'):.4f}")
    print(f"  mAP@0.5    {m.get('mAP@0.5'):.3f}   seg_mIoU {m.get('seg_mIoU'):.3f}")
    print(f"  sev_mae    {m.get('sev_mae'):.3f}   sev_qwk  {m.get('sev_qwk'):.3f}")
print("DONE", flush=True)
