"""CropGuard v7 -- repack the derived corpus for publishing (no GPU).

`cropguard-v7-plantseg-derive` runs in /kaggle/working, which also holds the
3 GB Zenodo zip, its 3 GB extraction and the annotation repo -- ~8 GB of
download artefacts that must not be pulled down just to publish a ~60 MB corpus.

Kaggle can mount another kernel's OUTPUT as read-only input (`kernel_sources`),
so this kernel copies only the derived corpus out of that mount and emits it as
its own (small) output. No GPU, so it burns no quota, and it does not re-derive
anything.

Outputs:
  cropguard_v7_split/       the training corpus (images + masks_lesion +
                            masks_leaf + labels, deduped, per split)
  cropguard_v7_negatives/   the not_tomato set (dedup_split drops these, but the
                            abstention/OOD story needs them)
"""
import glob
import os
import shutil
import subprocess
import sys

IN = "/kaggle/input"
WORK = "/kaggle/working"


def find_dir(name):
    cands = [os.path.join(IN, name)] + sorted(
        glob.glob(os.path.join(IN, "**", name), recursive=True))
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


CROP = find_dir("cropguard_v7_split")
print("inputs:", sorted(os.listdir(IN)), flush=True)
print("cropguard_v7_split ->", CROP, flush=True)
if CROP is None:
    # help the next person: show what the mount actually contains
    for dirpath, dirnames, _ in os.walk(IN):
        print("  mount:", dirpath, dirnames[:6], flush=True)
        if dirpath.count(os.sep) > 6:
            dirnames[:] = []
    sys.exit("cropguard_v7_split not found in the mounted kernel output")


def counts(root):
    return {s: len(os.listdir(os.path.join(root, "images", s)))
            for s in ("train", "val", "test")
            if os.path.isdir(os.path.join(root, "images", s))}


def run(cmd, label):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    print(f"[{label}] exit {rc}", flush=True)
    return rc


out = os.path.join(WORK, "cropguard_v7_split")
run(["cp", "-r", CROP, out], "copy-corpus")
print("corpus per-split image counts:", counts(out), flush=True)

for sub in ("masks_lesion", "masks_leaf", "labels"):
    for split in ("train", "val", "test"):
        d = os.path.join(out, sub, split)
        print(f"  {sub}/{split}: {len(os.listdir(d)) if os.path.isdir(d) else 0}",
              flush=True)

# negatives live beside the pre-dedup corpus (dedup_split deliberately skips them)
NEG = find_dir("cropguard_v7")
if NEG is not None and os.path.isdir(os.path.join(NEG, "images", "negatives")):
    src = os.path.join(NEG, "images", "negatives")
    dst = os.path.join(WORK, "cropguard_v7_negatives")
    run(["cp", "-r", src, dst], "copy-negatives")
    print("negatives:", len(os.listdir(dst)), flush=True)
else:
    print("negatives: not found (skipped)", flush=True)

# Emit TARBALLS, not loose trees. Kaggle's output-list API pages ~100 files at
# a time and the list endpoint stops answering after ~33 pages, so a 2500-file
# split (plus 6558 negatives) cannot be enumerated -- `kaggle kernels output`
# crawls and the paginated API hangs. Two archives = two files to fetch.
for name in ("cropguard_v7_split", "cropguard_v7_negatives"):
    d = os.path.join(WORK, name)
    if not os.path.isdir(d):
        continue
    run(["tar", "czf", os.path.join(WORK, name + ".tar.gz"), "-C", WORK, name],
        "tar-" + name)
    shutil.rmtree(d)                     # keep only the archive in the output

run(["ls", "-la", WORK], "ls-output")
print("\nDONE. Output = cropguard_v7_split.tar.gz (+ negatives tarball).",
      flush=True)
