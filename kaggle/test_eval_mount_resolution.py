"""Regression test for the eval checkpoint-mount bug (found 2026-10-12).

The first v9-eval / v10-eval run on Kaggle reported every metric as "not
measured" because both resolvers only listed the flat `/kaggle/input/<slug>`
and `/kaggle/input/datasets/<owner>/<slug>` forms. A probe kernel showed the
real layout is:

    dataset_sources -> /kaggle/input/datasets/<owner>/<slug>
    kernel_sources  -> /kaggle/input/notebooks/<owner>/<slug>      <-- was missing

This test extracts the REAL resolver functions out of both eval scripts (no
copy-paste, so it cannot drift) and runs them against a simulated mount tree,
by redirecting "/kaggle/input" to a temp directory via an `os` shim.

It also pins the two subtler requirements:
  * v8-train and v9-train both ship `best_v8_raw.pt` (v9 reuses the v8
    filenames on purpose), so they must be told apart by PATH, not filename;
  * `isdir()` is not sufficient -- an EMPTY directory at a candidate path must
    not win over a directory that actually contains a checkpoint.

Usage:  python3 kaggle/test_eval_mount_resolution.py
Exit code 0 = all PASS.
"""
import ast
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {
    "v9": (os.path.join(HERE, "v9-eval", "cropguard-v9-eval.py"),
           {"CKPT_NAMES", "TRAIN_OUT_ROOTS", "V8_TRAIN_ROOTS"},
           {"_walk_input", "log_pt_inventory", "_find_ckpt_dir",
            "find_train_out", "find_v8_train_out"}),
    "v10": (os.path.join(HERE, "v10-eval", "cropguard-v10-eval.py"),
            {"CKPT_NAMES_V9", "CKPT_NAMES_V10", "V9_OUT_ROOTS", "V10_OUT_ROOTS"},
            {"_walk_input", "log_pt_inventory", "_find_ckpt_dir",
             "find_train_out", "find_v9_train_out"}),
}

failures = []


# --------------------------------------------------------------- os shim ----
class _PathShim:
    """/kaggle/input -> tmpdir translation for os.path queries."""

    def __init__(self, translate):
        self._translate = translate

    def isdir(self, p):
        return os.path.isdir(self._translate(p))

    def isfile(self, p):
        return os.path.isfile(self._translate(p))

    def getsize(self, p):
        return os.path.getsize(self._translate(p))

    def __getattr__(self, name):
        return getattr(os.path, name)


class _OsShim:
    def __init__(self, fake_root):
        self.fake_root = fake_root
        self.path = _PathShim(self._translate)

    def _translate(self, p):
        if p == "/kaggle/input":
            return self.fake_root
        if p.startswith("/kaggle/input/"):
            return os.path.join(self.fake_root, p[len("/kaggle/input/"):])
        return p

    def walk(self, p, *a, **k):
        return os.walk(self._translate(p), *a, **k)

    def listdir(self, p):
        return os.listdir(self._translate(p))

    def __getattr__(self, name):
        return getattr(os, name)


def load_resolvers(path, wanted_assigns, wanted_funcs):
    """Exec only the named top-level nodes of an eval script, under a shimmed os."""
    src = open(path).read()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & wanted_assigns:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            keep.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)  # `import os` etc.
    missing = (wanted_assigns | wanted_funcs)
    got = set()
    for n in keep:
        if isinstance(n, ast.Assign):
            got |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        elif isinstance(n, ast.FunctionDef):
            got.add(n.name)
    missing -= got
    if missing:
        raise AssertionError(f"{path}: could not extract {sorted(missing)}")
    return ast.Module(body=keep, type_ignores=[])


def make_tree(root):
    """Simulate the measured layout, including an EMPTY decoy at a flat path."""
    def touch(*parts):
        p = os.path.join(root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\x00" * 16)

    # kernel_sources -> /kaggle/input/notebooks/<owner>/<slug>
    for slug, ckpts in (
        ("cropguard-v8-train", ["best_v8_raw.pt", "last_v8.pt"]),
        ("cropguard-v9-train", ["best_v8_raw.pt", "last_v8.pt"]),
        ("cropguard-v10-train-detection-enabled", ["best_v10_raw.pt", "last_v10.pt"]),
    ):
        n = os.path.join("notebooks", "ganrajlongadge", slug)
        os.makedirs(os.path.join(root, n), exist_ok=True)
        for c in ckpts:
            touch(n, c)
        touch(n, "stage_summary.json")

    # dataset_sources -> /kaggle/input/datasets/<owner>/<slug>
    os.makedirs(os.path.join(root, "datasets", "ganrajlongadge",
                             "cropguard-v9-filled-masks"), exist_ok=True)

    # DECOY: an existing but EMPTY directory at the old flat path. The buggy
    # `if os.path.isdir(base): return base` would happily return this.
    for slug in ("cropguard-v8-train", "cropguard-v9-train",
                 "cropguard-v10-train-detection-enabled"):
        os.makedirs(os.path.join(root, slug), exist_ok=True)


def check(label, got, want):
    ok = (got or "").replace("\\", "/").endswith(want)
    print(f"  {label:28s} -> {got}\n{'':32s}    want ...{want}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{label}: got {got!r}, want suffix {want!r}")


def make_empty_decoys(root):
    """PART B tree: ONLY empty dirs at the candidate paths, no .pt anywhere.

    This is the exact failure mode of the old `if os.path.isdir(base): return
    base` -- it would return an empty directory and then die later on a missing
    checkpoint. Correct behaviour is None + an empty inventory.
    """
    for slug in ("cropguard-v8-train", "cropguard-v9-train",
                 "cropguard-v10-train-detection-enabled"):
        os.makedirs(os.path.join(root, slug), exist_ok=True)
        os.makedirs(os.path.join(root, "notebooks", "ganrajlongadge", slug),
                    exist_ok=True)


def make_unlisted_tree(root):
    """PART C tree: checkpoints live SOMEWHERE THE ROOT LISTS DO NOT NAME.

    Pins the walk fallback. Two dirs hold `best_v8_raw.pt` at the same depth,
    and only one has the slug in its path, so this also pins that the fallback
    disambiguates v8-train from v9-train by path instead of filename.
    """
    def touch(*parts):
        p = os.path.join(root, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\x00" * 16)

    touch("uploads", "ganrajlongadge", "cropguard-v9-train", "best_v8_raw.pt")
    touch("uploads", "ganrajlongadge", "cropguard-v8-train", "best_v8_raw.pt")
    touch("uploads", "ganrajlongadge", "cropguard-v10-train-detection-enabled",
          "best_v10_raw.pt")
    # same depth, same filename, hint absent -> must never win the v9 lookup
    touch("uploads", "ganrajlongadge", "unrelated-archive", "best_v8_raw.pt")


def run_script(name, root):
    """Extract + run one eval script's resolvers under the shim; returns ns."""
    path, want_assigns, want_funcs = SCRIPTS[name]
    module = load_resolvers(path, want_assigns, want_funcs)
    ns = {"log": lambda m: None}
    exec(compile(module, path, "exec"), ns)
    # Install the shim AFTER exec: the extracted script body contains its own
    # `import os`, which would otherwise clobber it. Functions look `os` up in
    # their globals (== ns) at call time, so setting it here is what counts.
    ns["os"] = _OsShim(root)
    return ns


def new_fn(name):
    return "find_train_out"


def old_fn(name):
    return "find_v8_train_out" if name == "v9" else "find_v9_train_out"


NEW_WANT = {
    "v9": "notebooks/ganrajlongadge/cropguard-v9-train",
    "v10": "notebooks/ganrajlongadge/cropguard-v10-train-detection-enabled",
}
OLD_WANT = {
    "v9": "notebooks/ganrajlongadge/cropguard-v8-train",
    "v10": "notebooks/ganrajlongadge/cropguard-v9-train",
}


def main():
    # ---------------------------------------------------------------- A ----
    root = tempfile.mkdtemp(prefix="fakemount_A_")
    try:
        make_tree(root)
        print("=== PART A: measured Kaggle layout (notebooks/) + EMPTY flat decoys ===")
        print(f"simulated /kaggle/input at {root}")
        print(f"  notebooks/ : {sorted(os.listdir(os.path.join(root, 'notebooks', 'ganrajlongadge')))}")
        print()
        for name in ("v9", "v10"):
            print(f"--- {name}-eval resolvers ---")
            ns = run_script(name, root)
            check(f"{name} {new_fn(name)} (NEW)", ns[new_fn(name)](), NEW_WANT[name])
            check(f"{name} {old_fn(name)} (OLD)", ns[old_fn(name)](), OLD_WANT[name])
            print()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # ---------------------------------------------------------------- B ----
    root = tempfile.mkdtemp(prefix="fakemount_B_")
    try:
        make_empty_decoys(root)
        print("=== PART B: only EMPTY dirs at candidate paths -> must be None ===")
        for name in ("v9", "v10"):
            ns = run_script(name, root)
            for fn, label in ((new_fn(name), "NEW"), (old_fn(name), "OLD")):
                got = ns[fn]()
                ok = got is None
                print(f"  {name} {fn:22s} ({label}) -> {got}  {'PASS' if ok else 'FAIL'}")
                if not ok:
                    failures.append(f"B: {name} {fn} returned {got!r}, want None")
        print()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # ---------------------------------------------------------------- C ----
    root = tempfile.mkdtemp(prefix="fakemount_C_")
    try:
        make_unlisted_tree(root)
        print("=== PART C: checkpoints at an UNLISTED path -> walk fallback ===")
        for name in ("v9", "v10"):
            ns = run_script(name, root)
            check(f"{name} {new_fn(name)} via walk", ns[new_fn(name)](),
                  f"uploads/ganrajlongadge/{NEW_WANT[name].split('/')[-1]}")
            check(f"{name} {old_fn(name)} via walk", ns[old_fn(name)](),
                  f"uploads/ganrajlongadge/{OLD_WANT[name].split('/')[-1]}")
        print()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("=" * 60)
    if failures:
        print(f"OVERALL: FAIL ({len(failures)})")
        for f in failures:
            print("  -", f)
        return 1
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
