"""On-device ONNX latency benchmark (run this ON the phone, e.g. Termux).

    pip install onnxruntime numpy
    python3 bench_onnx_phone.py --model models/cropguard_v7_fp32.onnx

Same warmup (3) + timed loop as the browser bench, so the two are comparable.
Prints median / mean / p90 / min / max milliseconds and the runtime + provider.
"""
import argparse
import platform
import statistics
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=1, help="0 = ORT default")
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(1, 3, 320, 320).astype(np.float32)

    for _ in range(3):
        sess.run(None, {name: x})

    ts = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1e3)

    ts.sort()
    p90 = ts[min(len(ts) - 1, int(0.9 * len(ts)))]
    print(f"device     {platform.machine()} / {platform.system()}")
    print(f"python     {platform.python_version()}")
    print(f"onnxruntime {ort.__version__}  providers={sess.get_providers()}")
    print(f"model      {args.model}  threads={args.threads}")
    print(f"runs       {args.runs}")
    print(f"median     {statistics.median(ts):.1f} ms")
    print(f"mean       {statistics.mean(ts):.1f} ms")
    print(f"p90        {p90:.1f} ms")
    print(f"min / max  {ts[0]:.1f} / {ts[-1]:.1f} ms")


if __name__ == "__main__":
    main()
