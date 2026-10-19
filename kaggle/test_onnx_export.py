"""Regression test for the ONNX export device bug (found 2026-10-12).

Both v9-eval and v10-eval reported "ONNX export failed" on Kaggle and produced
no .onnx artifact at all. The traceback ended at the first conv:

    RuntimeError: Input type (torch.FloatTensor) and weight type
    (torch.cuda.FloatTensor) should be the same

because `load_ckpt()` moves the model to CUDA while the tracing `dummy` input
was built unconditionally on CPU.

This test extracts the REAL `OnnxWrap` / `export_onnx` functions out of the two
eval scripts (no copy-paste, so it cannot drift) and runs the full export path
on CPU:

  1. it must still succeed on CPU (the fix must not regress the CPU path);
  2. the exported FP32 graph must agree numerically with PyTorch;
  3. INT8 static quantization must still produce a file;
  4. `dummy` must be built on the MODEL's device -- pinned by running the
     export with the model on a non-CPU device stand-in. There is no GPU on
     this host, so the device rule is asserted structurally instead of by
     running CUDA: the source must derive the device from the model rather
     than hardcoding a CPU tensor.

Usage:  python3 kaggle/test_onnx_export.py
Exit code 0 = all PASS.
"""
import ast
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = {
    "v9": os.path.join(HERE, "v9-eval", "cropguard-v9-eval.py"),
    "v10": os.path.join(HERE, "v10-eval", "cropguard-v10-eval.py"),
}
WANT_FUNCS = {"export_onnx", "ensure_onnxruntime"}
WANT_CLASSES = {"OnnxWrap"}
WANT_ASSIGNS = {"SIZE", "NOT_MEASURED"}

failures = []


def load_export(path):
    """Exec only SIZE/NOT_MEASURED/OnnxWrap/export_onnx from an eval script."""
    tree = ast.parse(open(path).read())
    keep, got = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & WANT_ASSIGNS:
                keep.append(node)
                got |= names & WANT_ASSIGNS
        elif isinstance(node, ast.FunctionDef) and node.name in WANT_FUNCS:
            keep.append(node)
            got.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in WANT_CLASSES:
            keep.append(node)
            got.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.append(node)
    missing = (WANT_FUNCS | WANT_CLASSES | WANT_ASSIGNS) - got
    if missing:
        raise AssertionError(f"{path}: could not extract {sorted(missing)}")
    ns = {"log": lambda m: print(f"    [log] {m}")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
    return ns


def check_device_rule(name, path):
    """Structural pin: the tracing input must follow the model's device."""
    src = re.sub(r"#.*", "", open(path).read())            # ignore comments
    body = src[src.index("def export_onnx"):]
    body = body[:body.index("\ndef ", 10)]                 # just this function
    flat = re.sub(r"\s+", " ", body)
    has_dev = "next(model.parameters()).device" in flat
    dummy_on_dev = re.search(r"torch\.randn\(1, 3, SIZE, SIZE, device=dev\)", flat)
    bare_dummy = re.search(r"torch\.randn\(1, 3, SIZE, SIZE\)[^\n]*\n", flat)
    ok = bool(has_dev and dummy_on_dev)
    print(f"  {name}: derives device from model = {has_dev}, "
          f"dummy on that device = {bool(dummy_on_dev)}, "
          f"un-deviced randn left in export path = {bool(bare_dummy)}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{name}: export_onnx does not build the tracing input "
                        f"on the model's device")


def check_ensure_ort(name, ns):
    """ensure_onnxruntime() must be a no-op guard when ort is present, and must
    respect NO_ORT=1 rather than pip-installing behind the user's back."""
    saved = os.environ.get("NO_ORT")
    try:
        os.environ["NO_ORT"] = "1"
        guarded = ns["ensure_onnxruntime"]()
        os.environ.pop("NO_ORT", None)
        present = ns["ensure_onnxruntime"]()
    finally:
        if saved is None:
            os.environ.pop("NO_ORT", None)
        else:
            os.environ["NO_ORT"] = saved
    ok = (guarded is False) and (present is True)
    print(f"  {name}: NO_ORT=1 -> {guarded}, ort present -> {present}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"{name}: ensure_onnxruntime guard wrong "
                        f"(NO_ORT {guarded}, present {present})")


def run_export(name, ns):
    import torch
    sys.path.insert(0, os.path.abspath("model"))
    from architecture.v7_model import CropGuardMultiTaskV7

    torch.manual_seed(0)
    model = CropGuardMultiTaskV7(nc=8, pretrained=False).eval()
    # Load the real v9 checkpoint when present, so the export runs on the
    # actual weights rather than a random net.
    ckpt = "/home/logiclocus/tmp/kout_v9/best_v8_raw.pt"
    loaded = False
    if os.path.isfile(ckpt):
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = obj.get("model", obj)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded = (not missing and not unexpected)
    print(f"  {name}: real v9 checkpoint loaded = {loaded} "
          f"({os.path.basename(ckpt) if loaded else 'random weights'})")

    size = ns["SIZE"]
    calib = [torch.randn(3, size, size) for _ in range(4)]

    outdir = tempfile.mkdtemp(prefix=f"onnx_{name}_")
    fp32 = os.path.join(outdir, f"{name}_fp32.onnx")
    int8 = os.path.join(outdir, f"{name}_int8.onnx")
    try:
        res = ns["export_onnx"](model, fp32, int8, calib)
        ok_fp32 = os.path.isfile(fp32) and os.path.getsize(fp32) > 0
        diffs = res.get("max_abs_diff_vs_pytorch", {})
        logit_keys = [k for k in diffs if k != "n_images"]
        worst = max((diffs[k] for k in logit_keys), default=float("nan"))
        ok_agree = bool(logit_keys) and worst < 1e-3
        ok_int8 = res.get("int8_path") == int8 and os.path.isfile(int8)
        print(f"  {name}: fp32 written = {ok_fp32} ({res.get('fp32_mb')} MB)")
        print(f"  {name}: max |ONNX - PyTorch| = {worst:.3e} over "
              f"{diffs.get('n_images')} images  -> {'PASS' if ok_agree else 'FAIL'}")
        print(f"  {name}: int8 written = {ok_int8} ({res.get('int8_mb')} MB)"
              f"{'' if ok_int8 else '  err=' + str(res.get('int8_error'))}")
        if not ok_fp32:
            failures.append(f"{name}: no fp32 onnx produced")
        if not ok_agree:
            failures.append(f"{name}: ONNX/PyTorch disagreement {worst}")
        if not ok_int8:
            failures.append(f"{name}: no int8 onnx produced "
                            f"({res.get('int8_error')})")
    finally:
        shutil.rmtree(outdir, ignore_errors=True)


def main():
    if not os.path.isdir("model"):
        print("run this from the project root (needs ./model)")
        return 2
    for name, path in SCRIPTS.items():
        print(f"=== {name}-eval export_onnx ===")
        ns = load_export(path)
        check_device_rule(name, path)
        check_ensure_ort(name, ns)
        run_export(name, ns)
        print()
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
