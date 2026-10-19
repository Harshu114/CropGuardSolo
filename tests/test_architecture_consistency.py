"""Guard against drift between the canonical architecture and the serving copy.

`app/api/model.py` is a vendored copy of `model/architecture/model.py` because the
API runs as a script and is containerised with `app/api` as its Docker build
context, so the repo-root `model/` package is not importable there. That
duplication is deliberate, but it is also exactly the failure mode that broke this
project: the serving copy had drifted into describing a GUM module and task
adapters that were never trained, so the API could not load its own checkpoint.

This test fails the moment the two definitions disagree, so drift cannot be
committed silently.

Run directly:  python3 tests/test_architecture_consistency.py
Or under pytest.
"""
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CKPT = os.path.join(ROOT, "app", "api", "cropguard_model_v5.pt")
NC = 11


def _load_serving():
    """Import app/api/model.py by path (it is not part of a package)."""
    path = os.path.join(ROOT, "app", "api", "model.py")
    spec = importlib.util.spec_from_file_location("cropguard_serving_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canonical():
    from model.architecture.model import CropGuardMultiTask
    return CropGuardMultiTask


def _load_serving_v7():
    """Import app/api/model_v7_serving.py by path (not a package)."""
    path = os.path.join(ROOT, "app", "api", "model_v7_serving.py")
    spec = importlib.util.spec_from_file_location("cropguard_serving_model_v7",
                                                 path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canonical_v7():
    from model.architecture.v7_model import CropGuardMultiTaskV7
    return CropGuardMultiTaskV7


def test_v7_serving_and_canonical_have_identical_state_dicts():
    """Same guard as the v5 pair, for the new architecture."""
    serving = _load_serving_v7().CropGuardMultiTaskV7(nc=NC, pretrained=False)
    canonical = _canonical_v7()(nc=NC, pretrained=False)
    a, b = serving.state_dict(), canonical.state_dict()

    assert set(a) == set(b), (
        "v7 key sets differ\n"
        f"  only in serving:   {sorted(set(a) - set(b))[:8]}\n"
        f"  only in canonical: {sorted(set(b) - set(a))[:8]}")
    bad = {k for k in a if a[k].shape != b[k].shape}
    assert not bad, f"v7 shape mismatches: {sorted(bad)[:8]}"


def test_v7_forward_agrees_between_the_two_copies():
    torch.manual_seed(0)
    serving = _load_serving_v7().CropGuardMultiTaskV7(nc=NC, pretrained=False).eval()
    canonical = _canonical_v7()(nc=NC, pretrained=False).eval()
    serving.load_state_dict(canonical.state_dict())

    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        o1 = serving(x, with_analysis=True, return_abstain=True)
        o2 = canonical(x, with_analysis=True, return_abstain=True)
    assert set(o1) == set(o2)
    for k in ("cls_logits", "det_cls", "det_box", "seg_logits", "sev_cont",
              "lesion_ratio", "cls_energy"):
        assert torch.allclose(o1[k], o2[k], atol=1e-5), f"v7 output {k!r} diverges"
    # v7 seg is 2-channel (lesion, leaf) -- the whole point of the two-channel
    # severity definition
    assert o2["seg_logits"].shape[1] == 2, "v7 seg must be 2-channel"


def test_v7_cls_loss_does_not_grad_into_seg_head():
    """The lesion gate is detached: classification must not train the seg head."""
    canonical = _canonical_v7()(nc=NC, pretrained=False).train()
    out = canonical(torch.randn(1, 3, 320, 320))
    out["cls_logits"].sum().backward()
    g = sum(p.grad.abs().sum() for p in canonical.seg_head.parameters()
            if p.grad is not None)
    assert float(g) == 0.0, f"cls loss leaked into seg head (grad={float(g)})"


def test_serving_and_canonical_have_identical_state_dicts():
    serving = _load_serving().CropGuardMultiTask(nc=NC, pretrained=False)
    canonical = _canonical()(nc=NC, pretrained=False)
    a, b = serving.state_dict(), canonical.state_dict()

    assert set(a) == set(b), (
        "key sets differ\n"
        f"  only in serving:   {sorted(set(a) - set(b))[:8]}\n"
        f"  only in canonical: {sorted(set(b) - set(a))[:8]}")
    bad = {k for k in a if a[k].shape != b[k].shape}
    assert not bad, f"shape mismatches: {sorted(bad)[:8]}"


def test_shipped_checkpoint_loads_strictly_into_both():
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    if not hasattr(sd, "keys"):
        sd = sd.get("state_dict", sd)

    serving = _load_serving().CropGuardMultiTask(nc=NC, pretrained=False)
    canonical = _canonical()(nc=NC, pretrained=False)
    # strict=True is the point: a stale/partial load must fail loudly here rather
    # than produce plausible-looking garbage at request time.
    serving.load_state_dict(sd, strict=True)
    canonical.load_state_dict(sd, strict=True)

    assert len(sd) == 292, f"expected the v5 checkpoint to have 292 tensors, got {len(sd)}"
    n = sum(p.numel() for p in serving.parameters()) / 1e6
    assert 2.5 < n < 2.8, f"unexpected parameter count {n:.2f}M (MODEL_MANIFEST says 2.63M)"


def test_forward_agrees_between_the_two_copies():
    """Same input -> same outputs, proving the duplication is behaviourally inert."""
    torch.manual_seed(0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    if not hasattr(sd, "keys"):
        sd = sd.get("state_dict", sd)

    serving = _load_serving().CropGuardMultiTask(nc=NC, pretrained=False).eval()
    canonical = _canonical()(nc=NC, pretrained=False).eval()
    serving.load_state_dict(sd, strict=True)
    canonical.load_state_dict(sd, strict=True)

    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        o1 = serving(x, with_severity=True)
        o2 = canonical(x, with_severity=True)

    assert set(o1) == set(o2), f"output keys differ: {sorted(o1)} vs {sorted(o2)}"
    for k in ("det_cls", "det_box", "cls_logits", "seg_logits", "sev_cont", "sev_ord"):
        assert torch.allclose(o1[k], o2[k], atol=1e-5), f"output {k!r} diverges"
    # anchors are deterministic constants derived from the feature-map sizes
    assert torch.allclose(o1["anchors"], o2["anchors"], atol=1e-5)
    assert o1["anchors"].shape == (2100, 3)


def _equivalent(a, b):
    """The predicate the drift guard actually relies on: same names AND shapes.

    Checking key names alone is not enough. Changing the class count, for
    instance, leaves every parameter *name* identical and only changes the shape
    of two output layers -- a names-only comparison would report "no drift" for a
    genuinely different model.
    """
    if set(a) != set(b):
        return False
    return all(a[k].shape == b[k].shape for k in a)


def test_the_check_can_actually_fail():
    """A guard that cannot fail is worthless, so prove this one detects a change.

    Building the serving copy with the wrong class count must be reported as
    different, or the comparison in the tests above is vacuous.
    """
    serving = _load_serving().CropGuardMultiTask(nc=NC, pretrained=False)
    canonical = _canonical()(nc=NC, pretrained=False)
    wrong = _load_serving().CropGuardMultiTask(nc=NC - 1, pretrained=False)

    a, b, c = serving.state_dict(), canonical.state_dict(), wrong.state_dict()
    assert _equivalent(a, b), "the two real copies should match"
    # names are identical here -- only shapes differ, which is exactly the
    # failure mode a names-only guard would miss.
    assert set(a) == set(c)
    assert not _equivalent(a, c), "nc change was not detected -- guard is vacuous"


def test_consistency_loss_is_autocast_safe():
    """Regression: F.binary_cross_entropy is autocast-'unsafe' and RAISES under
    torch.amp.autocast("cuda") even when its inputs are fp32. The v7 consistency
    term needs BOTH cls_logits and seg_logits, so it only runs from stage 4 on --
    stages 1-3 trained fine and every full v7 run then died at the first joint
    stage. This asserts the term runs (and backprops) under autocast.
    """
    from model.training.consistency import consistency_loss
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seg = torch.randn(2, 2, 32, 32, device=dev, requires_grad=True)
    cls = torch.randn(2, NC, device=dev, requires_grad=True)
    with torch.amp.autocast(dev, enabled=dev == "cuda"):
        loss = consistency_loss({"seg_logits": seg, "cls_logits": cls})
    assert torch.isfinite(loss), "consistency loss is non-finite"
    assert float(loss.detach()) > 0.0, "consistency term should be active"
    loss.backward()  # must not raise, and must produce a gradient
    assert seg.grad is not None and torch.isfinite(seg.grad).all()


class Skipped(Exception):
    """Raised when a test needs an artifact that is legitimately absent.

    Only for large gitignored blobs (the archived v9/v10 checkpoints live under
    `model/weights/`, which is manifest-managed, NOT committed). A skip is
    reported separately and never counted as a pass, so it cannot hide a
    failure.
    """


def _newest_v7_family_checkpoint():
    """Newest archived v7-family best checkpoint, or None if none are on disk."""
    import glob
    cands = []
    for p in (os.path.join(ROOT, "app", "api", "cropguard_model_v7.pt"),
              *sorted(glob.glob(os.path.join(ROOT, "model", "weights", "v*",
                                             "best_*.pt")))):
        if os.path.exists(p):
            cands.append((os.path.getmtime(p), p))
    return max(cands)[1] if cands else None


def test_trained_v7_family_checkpoint_loads_into_the_serving_copy():
    """The point of the serving copy: a REAL trained checkpoint must load.

    The key-name/shape guard above proves the two definitions agree with each
    other; this proves they agree with the weights that were actually trained.
    Both halves are needed -- the decoder drift that broke this file passed a
    name-only comparison until the shapes were checked, and a checkpoint trained
    against a *third* variant would still fail here.

    Derives `nc` from the checkpoint instead of hardcoding it, so a future class
    count change is covered too.
    """
    ckpt = _newest_v7_family_checkpoint()
    if ckpt is None:
        raise Skipped(
            "no trained v7-family checkpoint on disk (expected "
            "app/api/cropguard_model_v7.pt or model/weights/v*/best_*.pt); "
            "run the v9/v10 eval kernels + model/weights/organize_weights.py")

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    nc = sd["cls_head.mlp.2.weight"].shape[0]

    serving = _load_serving_v7().CropGuardMultiTaskV7(nc=nc, pretrained=False)
    # strict=True is the point: 0 missing / 0 unexpected, or fail loudly.
    serving.load_state_dict(sd, strict=True)

    serving.eval()
    with torch.no_grad():
        out = serving(torch.randn(1, 3, 320, 320), with_analysis=True,
                      return_abstain=True)
    assert out["seg_logits"].shape[1] == 2, "v7 seg must be 2-channel"
    for k in ("cls_logits", "seg_logits", "sev_cont", "cls_energy"):
        assert k in out, f"real-weight forward is missing {k!r}"


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = skipped = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Skipped as e:
            skipped += 1
            print(f"  SKIP  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - report, don't mask
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    ran = len(tests) - skipped
    print(f"\n{ran - failed}/{ran} passed"
          + (f", {skipped} skipped (not counted as passes)" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
