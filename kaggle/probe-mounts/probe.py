"""Probe: where do kernel_sources actually mount in this runtime?

Prints the /kaggle/input tree (depth 4) and every *.pt found under it, using the
SAME dataset_sources + kernel_sources as cropguard-v9-eval.
"""
import os

DEPTH = 4
BASE = "/kaggle/input"


def walk(max_depth=DEPTH):
    out = []
    if not os.path.isdir(BASE):
        return out
    base_depth = BASE.rstrip("/").count(os.sep)
    for dp, dns, fns in os.walk(BASE):
        if dp.count(os.sep) - base_depth >= max_depth:
            dns[:] = []
        out.append((dp, sorted(fns)[:12], len(fns)))
    return out


print("=== /kaggle/input tree (depth %d) ===" % DEPTH, flush=True)
for dp, fns, n in walk():
    rel = dp[len(BASE):] or "/"
    print(f"  {rel}   [{n} files] {fns}", flush=True)

print("\n=== every *.pt under /kaggle/input ===", flush=True)
found = 0
for dp, dns, fns in os.walk(BASE):
    for f in fns:
        if f.endswith(".pt"):
            p = os.path.join(dp, f)
            print(f"  {p}  ({os.path.getsize(p)} bytes)", flush=True)
            found += 1
print(f"  total .pt files: {found}", flush=True)

print("\n=== candidate mount paths for the training outputs ===", flush=True)
for cand in (
    "/kaggle/input/cropguard-v8-train",
    "/kaggle/input/cropguard-v9-train",
    "/kaggle/input/cropguard-v10-train-detection-enabled",
    "/kaggle/input/datasets/ganrajlongadge/cropguard-v8-train",
    "/kaggle/input/datasets/ganrajlongadge/cropguard-v9-train",
    "/kaggle/input/datasets/ganrajlongadge/cropguard-v10-train-detection-enabled",
):
    print(f"  {cand}  isdir={os.path.isdir(cand)}", flush=True)

print("\n=== top level of /kaggle/input ===", flush=True)
try:
    print("  ", sorted(os.listdir(BASE)), flush=True)
except Exception as e:
    print("  ERR", e, flush=True)
