# CropGuard v7 — real-phone latency benchmark

Requirement: report latency measured on **an actual phone**, not a desktop CPU or
a Kaggle VM. A Kaggle T4/CPU number says nothing about what a farmer's phone
does, and the two runtimes disagree (static INT8 is *faster* than FP32 on a
Ryzen but *slower* on a T4) — so the number has to come from the target device.

This folder runs the shipped ONNX model **inside the phone's own browser** via
`onnxruntime-web` (WASM). Nothing to install, works on Android and iOS.

## Models

| file | size | what |
|---|---|---|
| `models/cropguard_v7_fp32.onnx` | 10.0 MB | FP32 reference |
| `models/cropguard_v7_fp32_int8_static.onnx` | 3.0 MB | calibrated INT8 (int Conv) |
| `models/cropguard_v7_fp32_int8_dynamic.onnx` | 2.8 MB | weights-only INT8 |

All three take `image [1,3,320,320]` float32 in `[0,1]` (RGB, direct resize, no
ImageNet normalisation — matching training).

## Run it

1. On the laptop, serve this folder over HTTP (WASM + `fetch` need HTTP, not
   `file://`):

   ```bash
   cd mobile/bench
   python3 -m http.server 8000
   ```

2. Find the laptop's LAN IP:

   ```bash
   hostname -I | awk '{print $1}'
   ```

3. Put the phone on the **same Wi-Fi** and open
   `http://<laptop-ip>:8000/phone_bench.html`.

4. Pick a model, tap **Run benchmark**. It warms up (3 runs) then times 30/50/100
   runs and prints median / mean / p90 / min / max plus the device user-agent.

5. Record the numbers here (this is what goes in the paper's latency table):

| model | device | median ms | p90 ms | notes |
|---|---|---|---|---|
| FP32 | _fill in_ | | | |
| INT8 static | _fill in_ | | | |
| INT8 dynamic | _fill in_ | | | |

> Browser WASM single-thread is a **conservative** proxy: a native app using the
> NNAPI / XNNPACK delegate is usually faster. Mark it as such in the paper.

## Alternative: Termux (Android, native ORT)

If the phone has Termux, measure against the native runtime instead:

```bash
pkg install python
pip install onnxruntime numpy pillow
python3 bench_onnx_phone.py --model models/cropguard_v7_fp32.onnx
```

`bench_onnx_phone.py` runs the same warmup + timed loop with ONNX Runtime's
Android CPU provider and prints the same statistics.
