"""Export CropGuard to ONNX and measure CPU latency (edge proxy).

    python3 model/training/export_onnx.py --ckpt runs/v7/student_best.pt \
        --out cropguard_v7.onnx --compare

Export modes:
  full  -- {det,cls,seg} heads + analytic severity (default)
  cascade -- det+cls only (the healthy-early-exit path; run seg/severity only
             when pred != healthy && conf above threshold)
Static 1x3x320x320 input. Dynamic H/W (post-quantization INT8) is left for a
follow-up; keep the static size for now so runtime ORT can fuse efficiently.
"""
import argparse
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def build(arch, ckpt_path=None, device="cpu", nc=8):
    if arch == "v7":
        from model.architecture.v7_model import CropGuardMultiTaskV7 as M
    else:
        from model.architecture.model import CropGuardMultiTask as M
    m = M(nc=nc, pretrained=False).to(device).eval()
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ck.get("model", ck)
        m.load_state_dict(sd, strict=True)
        print(f"loaded {ckpt_path}")
    return m


class CascadeWrapper(torch.nn.Module):
    """det+cls ONLY export path (the fast, healthy-early-exit mode)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x, active_tasks={"det", "cls"})
        return out["cls_logits"], out["det_cls"], out["det_box"]


class FullWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x, with_analysis=True)
        return (out["cls_logits"], out["det_cls"], out["det_box"],
                out["seg_logits"], out["sev_cont"])


def export(arch, ckpt, out, mode, nc=8):
    m = build(arch, ckpt, nc=nc)
    x = torch.randn(1, 3, 320, 320)
    wrap = (CascadeWrapper if mode == "cascade" else FullWrapper)(m)
    torch.onnx.export(
        wrap, x, out,
        input_names=["image"],
        output_names=["cls_logits", "det_cls", "det_box"] +
                     (["seg_logits", "sev_cont"] if mode == "full" else []),
        opset_version=17, dynamo=False,
        do_constant_folding=True,
    )
    print(f"exported -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


def bench_cpu(onnx_path, n_warm=5, n_run=30):
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1          # RPi-class single core
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 320, 320).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {"image": x})
    t = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        sess.run(None, {"image": x})
        t.append(time.perf_counter() - t0)
    t = np.array(t)
    return {"mean_ms": t.mean() * 1e3, "p95_ms": np.percentile(t, 95) * 1e3,
            "max_ms": t.max() * 1e3, "min_ms": t.min() * 1e3}


def quantize_int8(onnx_path, int8_path, weight_type="int8"):
    """Dynamic INT8 quantization (post-training, no data needed).

    Weights are quantized to int8/uint8 with per-channel scales; activations are
    quantized at runtime. This is the right first step for edge (RPi-class) CPU:
    it needs no calibration set and shrank the FP32 graph's dominant cost
    (Conv/MatMul) with negligible accuracy loss in our local checks. Static
    (calibrated) quantization can be a later refinement if latency demands it.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    # torch.onnx.export (with constant folding) leaves some Conv/MatMul weights
    # as Constant nodes rather than graph initializers, which makes
    # quantize_dynamic fail with "Expected onnx::Conv_XXX to be an initializer".
    # quant_pre_process runs symbolic shape inference + graph optimizations that
    # promote those to initializers (onnxruntime itself prints a warning telling
    # you to do this). Fall back to the raw model if pre-processing misfires.
    src = onnx_path
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = onnx_path.replace(".onnx", "_pre.onnx")
        quant_pre_process(onnx_path, pre, skip_symbolic_shape=False)
        src = pre
    except Exception as e:  # noqa: BLE001
        print(f"quant pre-process skipped ({type(e).__name__}: {e})")
    wt = QuantType.QInt8 if weight_type == "int8" else QuantType.QUInt8
    quantize_dynamic(model_input=src, model_output=int8_path, weight_type=wt)
    print(f"quantized (dynamic) -> {int8_path} "
          f"({os.path.getsize(int8_path)/1e6:.1f} MB)")
    return int8_path


def make_calib_reader(root, n=32, size=320):
    """CalibrationDataReader over real images for STATIC quantization.

    Static quantization needs activation ranges, so it needs representative
    data -- random tensors give meaningless scales and wreck accuracy. Falls
    back to None (=> dynamic only) if the corpus is unavailable.
    """
    import numpy as np
    from onnxruntime.quantization import CalibrationDataReader
    from torch.utils.data import DataLoader

    from dataset import CropGuardDataset, collate  # same dir

    ds = CropGuardDataset(root, "train", size, augment=False, max_images=n)
    loader = DataLoader(ds, batch_size=1, collate_fn=collate, shuffle=False)

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(loader)

        def get_next(self):
            try:
                b = next(self._it)
            except StopIteration:
                return None
            return {"image": np.asarray(b["image"], dtype=np.float32)}

    return _Reader()


def quantize_int8_static(onnx_path, int8_path, calib_reader,
                         quant_format="qdq"):
    """Static INT8 quantization (calibrated). Integer Conv kernels, so this is
    the mode that can actually beat fp32 on CPU -- unlike dynamic, which wraps
    every Conv/MatMul in float<->int casts and lost badly in our measurements.
    """
    from onnxruntime.quantization import (
        quantize_static, QuantFormat, QuantType)
    from onnxruntime.quantization.shape_inference import quant_pre_process
    src = onnx_path
    try:
        pre = onnx_path.replace(".onnx", "_pre.onnx")
        if not os.path.exists(pre):
            quant_pre_process(onnx_path, pre, skip_symbolic_shape=False)
        src = pre
    except Exception as e:  # noqa: BLE001
        print(f"quant pre-process skipped ({type(e).__name__}: {e})")
    fmt = QuantFormat.QDQ if quant_format == "qdq" else QuantFormat.QOperator
    quantize_static(model_input=src, model_output=int8_path,
                    calibration_data_reader=calib_reader,
                    quant_format=fmt, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8)
    print(f"quantized (static/{quant_format}) -> {int8_path} "
          f"({os.path.getsize(int8_path)/1e6:.1f} MB)")
    return int8_path


def bench_cascade(model, n_warm=5, n_run=30):
    """PyTorch-side cascade latency: det+cls always; seg only if diseased.
    Uses the trained model to decide -- measures the ACTUAL save, not theory."""
    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        for _ in range(n_warm):
            model(x, active_tasks={"det", "cls"})
        t = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            out = model(x, active_tasks={"det", "cls"})
            conf, pred = out["cls_logits"].softmax(-1).max(-1)
            t.append(time.perf_counter() - t0)
            if pred.item() != 0:                     # not healthy -> run seg too
                t0 = time.perf_counter()
                model(x, with_analysis=True)
                t[-1] += time.perf_counter() - t0
    import numpy as np
    t = np.array(t)
    return {"mean_ms": t.mean() * 1e3, "p95_ms": np.percentile(t, 95) * 1e3,
            "max_ms": t.max() * 1e3, "min_ms": t.min() * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="v7", choices=["v5", "v7"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="cropguard_v7.onnx")
    ap.add_argument("--mode", default="full", choices=["full", "cascade"])
    ap.add_argument("--compare", action="store_true",
                    help="bench full vs cascade on CPU(ORT single-thread)")
    ap.add_argument("--torch-only", action="store_true",
                    help="skip ONNX; just bench the cascade in PyTorch")
    ap.add_argument("--quantize", action="store_true", default=True,
                    help="also emit an INT8 (dynamic) model + bench it")
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    ap.add_argument("--int8-out", default=None,
                    help="INT8 output path (default: <out>_int8.onnx)")
    ap.add_argument("--quant-mode", default="static",
                    choices=["dynamic", "static", "both"],
                    help="static = calibrated int Conv (faster on CPU); "
                         "dynamic = weights-only")
    ap.add_argument("--calib-root", default=os.environ.get("CROPGUARD_DATA"),
                    help="corpus root for static-quantization calibration")
    ap.add_argument("--nc", type=int, default=8, help="number of classes")
    args = ap.parse_args()

    results = {}
    if not args.torch_only:
        export(args.arch, args.ckpt, args.out, args.mode, nc=args.nc)
        # Each bench is isolated: a missing onnxruntime or an op the quantizer
        # can't handle must not discard the exported FP32 model.
        try:
            results["onnx_cpu_1thread_fp32"] = bench_cpu(args.out)
            print("onnx fp32 cpu latency:", results["onnx_cpu_1thread_fp32"])
        except Exception as e:  # noqa: BLE001
            print(f"fp32 bench skipped ({type(e).__name__}: {e})")
        if args.quantize:
            base = args.int8_out or args.out.replace(".onnx", "_int8.onnx")
            modes = (["static", "dynamic"] if args.quant_mode == "both"
                     else [args.quant_mode])
            for mode in modes:
                path = (base if len(modes) == 1
                        else base.replace("_int8", f"_int8_{mode}"))
                try:
                    if mode == "static":
                        reader = make_calib_reader(args.calib_root,
                                                   args.calib_images, 320)
                        quantize_int8_static(args.out, path, reader)
                    else:
                        quantize_int8(args.out, path)
                    results[f"onnx_cpu_1thread_int8_{mode}"] = bench_cpu(path)
                    print(f"onnx int8({mode}) cpu latency:",
                          results[f"onnx_cpu_1thread_int8_{mode}"])
                except Exception as e:  # noqa: BLE001
                    print(f"quantization[{mode}] skipped "
                          f"({type(e).__name__}: {e})")
    if args.compare or args.torch_only:
        m = build(args.arch, args.ckpt)
        results["torch_cascade"] = bench_cascade(m)
        results["torch_full"] = None
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            for _ in range(5):
                m(x, with_analysis=True)
            t = []
            for _ in range(30):
                t0 = time.perf_counter()
                m(x, with_analysis=True)
                t.append(time.perf_counter() - t0)
        import numpy as np
        t = np.array(t)
        results["torch_full"] = {
            "mean_ms": t.mean() * 1e3, "p95_ms": np.percentile(t, 95) * 1e3}
        print("torch cascade:", results["torch_cascade"])
        print("torch full   :", results["torch_full"])

    import json
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
