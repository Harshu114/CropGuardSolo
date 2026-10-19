"""CropGuard v7 -- PlantSeg corpus derivation ONLY (no training).

Splitting the PlantSeg story in two is deliberate:

  * this kernel  -- download -> derive -> dedup-split, and CACHES the corpus as
                    kernel output. It needs no GPU, so it burns **zero** of the
                    account's GPU quota (the 3 GB Zenodo download is the long
                    pole, not the compute).
  * the train kernel -- attaches the published `cropguard-v7-plantseg-corpus`
                    dataset and trains, so a 4 h GPU slot is never spent twice
                    on data that is already derived.

Derives exactly the labels the expert brief asks for (one dataset, every label
generated):
  class      <- PlantSeg filename via the VERIFIED class_map.yaml
  boxes      <- connected components of the lesion mask
  leaf mask  <- GrabCut on the image (--leaf-backend grabcut | sam)
  severity   <- lesion area / leaf area, close-ups only
  negatives  <- the 33 non-tomato hosts become `not_tomato`

Outputs (saved to /kaggle/working, then publish as a dataset):
  cropguard_v7_split/   the derived, deduped corpus
  derive_summary.json   counts per split/class (what to sanity-check first)
"""
import json
import os
import subprocess
import sys

IN = "/kaggle/input"
WORK = "/kaggle/working"

PS_ZIP_URL = "https://zenodo.org/records/13762907/files/plantseg.zip"
PS_ANNOS_REPO = "https://github.com/tqwei05/PlantSeg"


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


def run(cmd, env=None, label="", check=True):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, env=env)
    print(f"[{label or cmd[0]}] exit {rc}", flush=True)
    if check and rc != 0:
        sys.exit(f"{label or cmd[0]} failed with {rc}")
    return rc


CODE_DS = find_input(os.environ.get("CODE_DS", "cropguard-v7-code"), "training")
print("input mount:", IN, sorted(os.listdir(IN))[:6], flush=True)
print("CODE_DS:", CODE_DS, flush=True)
if CODE_DS is None:
    sys.exit(f"code dataset not found under {IN}; got {os.listdir(IN)}")

DERIVE_LEAF = os.environ.get("DERIVE_LEAF", "grabcut")   # grabcut | sam

# ------------------------------------------------------------- 0. code tree ---
code = "/kaggle/working/cgcode"
os.makedirs(os.path.join(code, "model"), exist_ok=True)
for sub in ("architecture", "training"):
    dst = os.path.join(code, "model", sub)
    for cand in (os.path.join(CODE_DS, sub), os.path.join(CODE_DS, "model", sub)):
        if os.path.isdir(cand):
            run(["cp", "-r", cand, dst], label=f"code:{sub}")
            break

env = dict(os.environ)
env["PYTHONPATH"] = code

DP_SRC = os.path.join(CODE_DS, "data_pipeline")
DP = os.path.join(code, "data_pipeline", "scripts")
if os.path.isdir(DP_SRC):
    # Copy the *contents* (trailing "/."): `cp -r src dst` with dst already
    # present would nest it as dst/src.
    run(["mkdir", "-p", os.path.join(code, "data_pipeline")])
    run(["cp", "-r", os.path.join(DP_SRC, "."),
         os.path.join(code, "data_pipeline")], label="code:data_pipeline")
CFG = os.path.join(code, "data_pipeline", "configs", "class_map.yaml")
if not os.path.exists(os.path.join(DP, "derive_dataset.py")):
    sys.exit(f"derive_dataset.py missing at {DP}; CODE_DS contents: {os.listdir(CODE_DS)}")

# --------------------------------------------------------------- 1. download ---
zip_path = os.path.join(WORK, "plantseg.zip")
if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 2_000_000_000):
    run(["curl", "-L", "--retry", "5", "--retry-delay", "5", "-o", zip_path,
         PS_ZIP_URL], check=False, label="download-plantseg")
size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
print(f"plantseg.zip = {size / 1e9:.2f} GB", flush=True)
if size < 2_000_000_000:
    sys.exit(f"plantseg.zip incomplete ({size} B) -- Zenodo is throttling; re-run")

if not os.path.isdir(os.path.join(WORK, "PlantSeg", "coco_annotations")):
    run(["git", "clone", "--depth", "1", PS_ANNOS_REPO,
         os.path.join(WORK, "PlantSeg")], check=False, label="clone-annos")

# ------------------------------------------------------------ 2. class check ---
# Verify the mapping BEFORE deriving labels from 3 GB of images: a non-zero exit
# means class_map.yaml is missing an entry -- fix the table, do not trim it.
run([sys.executable, os.path.join(DP, "derive_dataset.py"),
     "--check-annos", os.path.join(WORK, "PlantSeg", "coco_annotations"),
     "--map", CFG], env=env, check=False, label="class-map-check")

# ------------------------------------------------------ 3. PlantSeg -> corpus ---
if not os.path.isdir(os.path.join(WORK, "plantseg")):
    run(["unzip", "-q", zip_path, "-d", os.path.join(WORK, "plantseg")],
        label="unzip")

run([sys.executable, os.path.join(DP, "prepare_plantseg.py"),
     "--root", os.path.join(WORK, "plantseg"),
     "--annos", os.path.join(WORK, "PlantSeg", "coco_annotations"),
     "--out", os.path.join(WORK, "plantseg_tomato"), "--plant", "tomato"],
    env=env, label="prepare_plantseg")

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

# ---------------------------------------------------------------- 6. summary ---
SPLIT = os.path.join(WORK, "cropguard_v7_split")
summary = {"splits": {}, "plantseg_zip_gb": round(size / 1e9, 2)}
for split in ("train", "val", "test"):
    d = os.path.join(SPLIT, "images", split)
    n = len(os.listdir(d)) if os.path.isdir(d) else 0
    summary["splits"][split] = n
print("\n=== derive summary ===")
print(json.dumps(summary, indent=2), flush=True)
with open(os.path.join(WORK, "derive_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print("\nDONE. Publish /kaggle/working/cropguard_v7_split as a dataset, then run",
      "the v7 train kernel against it.", flush=True)
