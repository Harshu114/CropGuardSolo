"""CropGuard v7 on the **PlantSeg** single-dataset corpus (Kaggle script kernel).

This is the run the expert brief describes: one dataset, every label derived.

  download   PlantSeg (3 GB, Zenodo) + annotation repo  -- done HERE, on Kaggle,
             because the local link to Zenodo is throttled to ~5 KB/s and the
             data step needs datacenter bandwidth (expert: "download PlantSeg,
             SAM weights and the backbone weights beforehand"). The tmux-style
             split matters: this whole step is CACHED as kernel output, so a
             later training-only kernel can attach it instead of re-deriving.
  derive     prepare_plantseg.py -> plantseg_tomato, then derive_dataset.py
             -> cropguard_v7 (class from filename via VERIFIED class_map.yaml,
             boxes from lesion-mask components, leaf masks via GrabCut,
             severity = lesion/leaf on close-ups only, non-tomato negatives)
  split      dedup_split.py (perceptual hash; duplicate groups never straddle
             splits)
  train      train.py --arch v7
  eval       eval_tta.py --arch v7 (per-class F1 + ECE)
  calibrate  calibrate_abstention.py --write
  export     export_onnx.py --arch v7

Outputs saved to /kaggle/working for reuse:
  cropguard_v7_split/   the derived, deduped corpus  (publish as a dataset)
  runs/student_best.pt  the v7 checkpoint
  metrics_v7.json, cropguard_v7.onnx
"""
import os
import subprocess
import sys

IN = "/kaggle/input"
WORK = "/kaggle/working"


def find_input(name, must_contain=None):
    """Resolve a mounted dataset: /kaggle/input/<slug> (classic) or
    /kaggle/input/datasets/<owner>/<slug> (newer runtimes)."""
    import glob as _glob
    cands = [os.path.join(IN, name)] + sorted(
        _glob.glob(os.path.join(IN, "**", name), recursive=True))
    for c in cands:
        if os.path.isdir(c) and (must_contain is None or
                                 os.path.exists(os.path.join(c, must_contain))):
            return c
    return None


CODE_DS = find_input(os.environ.get("CODE_DS", "cropguard-v7-code"), "training")
print("input mount:", IN, sorted(os.listdir(IN))[:6], flush=True)
print("CODE_DS:", CODE_DS, flush=True)
if CODE_DS is None:
    sys.exit(f"code dataset not found under {IN}; got {os.listdir(IN)}")
PS_ZIP_URL = "https://zenodo.org/records/13762907/files/plantseg.zip"
PS_ANNOS_REPO = "https://github.com/tqwei05/PlantSeg"

DERIVE_LEAF = os.environ.get("DERIVE_LEAF", "grabcut")   # grabcut | sam
STAGES = os.environ.get("STAGES", "1,2,3,4,5")
STAGE_EPOCHS = os.environ.get("STAGE_EPOCHS", "10")


def run(cmd, env=None, cwd=None, check=True, label=""):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env=env, cwd=cwd)
    print(f"[{label or cmd[0]}] exit {rc}", flush=True)
    if check and rc != 0:
        sys.exit(f"{label or cmd[0]} failed with {rc}")
    return rc


# ------------------------------------------------------------- 0. code tree ---
# Same flattened-mount handling as the corpus kernel: a dataset built from a
# top-level folder arrives at the mount root, so `model/architecture` can appear
# as `architecture`.
code = "/kaggle/working/cgcode"
os.makedirs(os.path.join(code, "model"), exist_ok=True)
for sub in ("architecture", "training"):
    dst = os.path.join(code, "model", sub)
    for cand in (os.path.join(CODE_DS, sub), os.path.join(CODE_DS, "model", sub)):
        if os.path.isdir(cand):
            run(["cp", "-r", cand, dst], label=f"code:{sub}")
            break
need = [os.path.join(code, "model", "architecture", n)
        for n in ("v7_model.py", "v7_heads.py")]
missing = [p for p in need if not os.path.exists(p)]
if missing:
    sys.exit(f"v7 code missing: {missing}; CODE_DS contents: {os.listdir(CODE_DS)}")
env = dict(os.environ)
env["PYTHONPATH"] = code              # so `python <script>` finds model.*

# The data-pipeline scripts + class_map.yaml ship in the SAME code dataset but
# are not a python package, so they are run by absolute path.
DP_SRC = os.path.join(CODE_DS, "data_pipeline")
DP = os.path.join(code, "data_pipeline", "scripts")
if os.path.isdir(DP_SRC):
    # Copy the *contents* of data_pipeline (note the trailing "/."): a plain
    # `cp -r src dst` where dst already exists nests it as dst/src, which is
    # what silently broke the first plantseg run (scripts landed in
    # data_pipeline/data_pipeline/).
    run(["mkdir", "-p", os.path.join(code, "data_pipeline")])
    run(["cp", "-r", os.path.join(DP_SRC, "."),
         os.path.join(code, "data_pipeline")], label="code:data_pipeline")
CFG = os.path.join(code, "data_pipeline", "configs", "class_map.yaml")

# --------------------------------------------------------------- 1. download ---
if not os.path.exists(os.path.join(WORK, "plantseg.zip")):
    run(["curl", "-L", "--retry", "3", "-o", os.path.join(WORK, "plantseg.zip"),
         PS_ZIP_URL], check=False, label="download-plantseg")
if not os.path.exists(os.path.join(WORK, "PlantSeg")):
    run(["git", "clone", "--depth", "1", PS_ANNOS_REPO,
         os.path.join(WORK, "PlantSeg")], check=False, label="clone-annos")

# ------------------------------------------------------------ 2. class check ---
# Verify the mapping BEFORE deriving labels from 3 GB of images (expert: "build
# and verify a mapping table first"). A non-zero exit here means class_map.yaml
# is missing an entry -- fix it, do not trim the table.

# ------------------------------------------------------ 3. PlantSeg -> corpus ---
if not os.path.isdir(os.path.join(WORK, "plantseg")):
    run(["unzip", "-q", os.path.join(WORK, "plantseg.zip"), "-d",
         os.path.join(WORK, "plantseg")], label="unzip")
run([sys.executable, os.path.join(DP, "derive_dataset.py"),
     "--check-annos", os.path.join(WORK, "PlantSeg", "coco_annotations"),
     "--map", CFG], env=env, check=False, label="class-map-check")

run([sys.executable, os.path.join(DP, "prepare_plantseg.py"),
     "--root", os.path.join(WORK, "plantseg"),
     "--annos", os.path.join(WORK, "PlantSeg", "coco_annotations"),
     "--out", os.path.join(WORK, "plantseg_tomato"), "--plant", "tomato"],
    env=env, check=False, label="prepare_plantseg")

# ------------------------------------------------------------ 4. derive all ---
run([sys.executable, os.path.join(DP, "derive_dataset.py"),
     "--in", os.path.join(WORK, "plantseg_tomato"),
     "--out", os.path.join(WORK, "cropguard_v7"),
     "--map", CFG,
     "--leaf-backend", DERIVE_LEAF,
     "--negatives-in", os.path.join(WORK, "plantseg")],
    env=env, label="derive_dataset")

# ------------------------------------------------------------ 5. dedup split ---
run([sys.executable, os.path.join(DP, "dedup_split.py"),
     "--in", os.path.join(WORK, "cropguard_v7"),
     "--out", os.path.join(WORK, "cropguard_v7_split")], env=env, label="dedup_split")

DATA = os.path.join(WORK, "cropguard_v7_split")
runs = os.path.join(WORK, "runs")

# ------------------------------------------------------------------ 6. train ---
env["CROPGUARD_DATA"] = DATA
run([sys.executable, os.path.join(code, "model", "training", "train.py"),
     "--arch", "v7", "--root", DATA, "--out", runs,
     "--stages", STAGES, "--stage-epochs", STAGE_EPOCHS,
     "--batch-size", os.environ.get("BATCH_SIZE", "32"),
     "--size", "320", "--workers", "2", "--amp", "--ema",
     "--log-every", "50"], env=env, label="train")

ckpt = os.path.join(runs, "student_best.pt")
# onnxruntime is NOT in the base Kaggle image (see the other drivers).
if os.environ.get("NO_ORT") != "1":
    print("installing onnxruntime + onnx for export/quantization...", flush=True)
    subprocess.call([sys.executable, "-m", "pip", "install", "-q",
                     "onnxruntime", "onnx"], env=env)
for label, script, extra in (
    ("eval", "eval_tta.py",
     ["--arch", "v7", "--ckpt", ckpt, "--root", DATA, "--split", "test",
      "--out", os.path.join(WORK, "metrics_v7.json")]),
    ("calibrate", "calibrate_abstention.py",
     ["--ckpt", ckpt, "--root", DATA, "--split", "val",
      "--target-abstain", "0.05", "--write"]),
    ("export", "export_onnx.py",
     ["--arch", "v7", "--ckpt", ckpt, "--out",
      os.path.join(WORK, "cropguard_v7.onnx"),
      "--quant-mode", os.environ.get("QUANT_MODE", "both"),
      "--calib-root", DATA]),
):
    run([sys.executable, os.path.join(code, "model", "training", script)] + extra,
        env=env, check=False, label=label)   # never lose weights to step 7

print("\nDONE. Reusable outputs: cropguard_v7_split/, runs/, metrics_v7.json",
      flush=True)
