"""Watchdog + heartbeat helpers for the CropGuard Kaggle script kernels.

Why this exists
---------------
A Kaggle script kernel is a black box: its log is only visible in the Kaggle UI
while it runs, and there is no API to read that log. A child process that hangs
therefore costs the entire session and tells us nothing (this already happened
once: `nvidia-smi -l` under `subprocess.run()` never returns, and the smoke
kernel sat silent for 31 minutes with 0 B of output).

`run_guarded` gives every long step three things:

  * live relay of the child's stdout/stderr, line-buffered, straight into the
    kernel log (so the UI shows training progress as it happens);
  * a heartbeat line every `heartbeat` seconds saying how long the step has run
    and when it last produced output or wrote a file -- proof of liveness even
    when the child is quiet;
  * a stall/hard timeout that kills the child, and before killing it sends
    SIGABRT, which `faulthandler` (enabled via `-X faulthandler`) turns into a
    full Python traceback of every thread in the log. So a hang reports *where*
    instead of just timing out.

The kernel always finishes within (stall|hard) bounds, so whatever checkpoints
already exist in /kaggle/working stay downloadable.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

__all__ = ["run_guarded", "PREFLIGHT_SRC", "best_checkpoint"]


def _newest_mtime(dirs) -> float:
    """Newest mtime under `dirs`, or 0.0. Used as a progress signal for steps
    whose only output is a file (e.g. a checkpoint written each epoch)."""
    newest = 0.0
    for d in dirs:
        for root, _sub, files in os.walk(d):
            for f in files:
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(root, f)))
                except OSError:
                    pass
    return newest


def run_guarded(cmd, env=None, label=None, watch_dirs=(), heartbeat=60,
                stall_timeout=None, hard_timeout=None):
    """Run `cmd`, relaying output live, with heartbeat + stall/hard timeouts.

    Returns the child's exit code. A stall kill returns -6 (SIGABRT) for
    attribution in the log; a hard timeout kill returns -15/-9.
    """
    label = label or os.path.basename(str(cmd[0]))
    env = dict(os.environ if env is None else env)
    env["PYTHONUNBUFFERED"] = "1"          # pipe instead of tty: keep it live
    env.setdefault("PYTHONFAULTHANDLER", "1")

    cmd = list(cmd)
    # -X faulthandler: makes the SIGABRT dump below actually print a traceback.
    if cmd[:1] == [sys.executable] and "-X" not in cmd[1:3]:
        cmd = [cmd[0], "-X", "faulthandler"] + cmd[1:]

    t0 = time.time()
    print(f"[guard] {label}: start (stall_timeout={stall_timeout}s, "
          f"hard_timeout={hard_timeout}s)", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    state = {"last_output": t0, "lines": 0, "killed": None}
    lock = threading.Lock()

    def relay():
        assert proc.stdout is not None
        for line in proc.stdout:
            with lock:
                state["last_output"] = time.time()
                state["lines"] += 1
            sys.stdout.write(line)
            sys.stdout.flush()

    reader = threading.Thread(target=relay, daemon=True)
    reader.start()

    def progress() -> float:
        with lock:
            last = state["last_output"]
        return max(last, _newest_mtime(watch_dirs), t0)

    def kill(reason: str, sig: int, code: int) -> None:
        with lock:
            state["killed"] = reason
        print(f"\n[guard] {label}: {reason}; sending {signal.Signals(sig).name} "
              f"to pid {proc.pid} (awaiting faulthandler traceback)", flush=True)
        try:
            os.kill(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            print(f"[guard] {label}: still alive after {signal.Signals(sig).name}"
                  f", SIGKILL", flush=True)
            proc.kill()

    while True:
        rc = proc.poll()
        if rc is not None:
            break
        elapsed = time.time() - t0
        if stall_timeout and time.time() - progress() > stall_timeout:
            kill(f"no output or file activity for {stall_timeout}s "
                 f"(elapsed {elapsed:.0f}s)", signal.SIGABRT, -6)
            break
        if hard_timeout and elapsed > hard_timeout:
            kill(f"hard timeout {hard_timeout}s exceeded", signal.SIGTERM, -15)
            break
        # heartbeat: cheap proof of life between the child's own prints
        if int(elapsed) % heartbeat < 1:
            print(f"[guard] {label}: t={elapsed:.0f}s alive, "
                  f"{state['lines']} lines, last activity "
                  f"{time.time() - progress():.0f}s ago", flush=True)
        time.sleep(1)

    reader.join(timeout=10)
    rc = proc.returncode
    with lock:
        killed, lines = state["killed"], state["lines"]
    dt = time.time() - t0
    print(f"[guard] {label}: exit {rc} after {dt:.0f}s ({lines} lines"
          f"{', killed: ' + killed if killed else ''})", flush=True)
    return rc


def best_checkpoint(out_dir: str):
    """Best available checkpoint, so a run that dies mid-schedule still yields
    something to evaluate instead of nothing."""
    for name in ("student_best.pt", "best_stage5.pt", "best_stage4.pt",
                 "best_stage3.pt", "best_stage2.pt", "best_stage1.pt", "last.pt"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            return p
    return None


# --------------------------------------------------------------------------
# Preflight: fail in 2 minutes on the Kaggle-specific landmines (missing GPU,
# no internet for the pretrained backbone, a broken v7 package) instead of
# discovering them 4 hours into a training session.
# --------------------------------------------------------------------------
PREFLIGHT_SRC = r'''
import os, sys, time
sys.path.insert(0, os.environ["CROPGUARD_WORK"])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available(), "no CUDA device"
print("gpu", torch.cuda.get_device_name(0), flush=True)

from model.architecture.v7_model import CropGuardMultiTaskV7
t0 = time.time()
model = CropGuardMultiTaskV7(num_classes=11, pretrained=True)   # downloads weights
print(f"built v7 pretrained in {time.time()-t0:.1f}s; "
      f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

dev = "cuda"
model = model.to(dev).train()
x = torch.rand(4, 3, 320, 320, device=dev)
cls = torch.randint(0, 11, (4,), device=dev)
tgt = {"boxes": [torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=dev) for _ in range(4)],
       "labels": [torch.tensor([2], device=dev) for _ in range(4)]}
out = model(x, gt_mask=torch.rand(4, 2, 320, 320, device=dev), gt_mask_prob=0.8)
print("forward keys:", sorted(out.keys()), flush=True)
loss = out.get("total")
if loss is None:
    raise SystemExit(f"no total loss in output: {sorted(out)}")
loss.backward()
gn = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f"loss {loss.item():.4f} grad_norm {gn:.2e}", flush=True)
assert gn > 0, "zero gradients"
out["logits"].sum().backward if False else None
print("PREFLIGHT PASS", flush=True)
'''


def preflight(work_dir: str, env: dict, extra_env: dict | None = None,
              script_path: str | None = None, stall_timeout: int = 420) -> int:
    """Run PREFLIGHT_SRC (or `script_path`) under the guard; returns exit code."""
    env = dict(env)
    env["CROPGUARD_WORK"] = work_dir
    env.update(extra_env or {})
    path = script_path or os.path.join(work_dir, "_cg_preflight.py")
    with open(path, "w") as fh:
        fh.write(PREFLIGHT_SRC)
    return run_guarded([sys.executable, path], env=env, label="preflight",
                       stall_timeout=stall_timeout, hard_timeout=stall_timeout * 2)
