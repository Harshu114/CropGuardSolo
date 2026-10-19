"""Export the v11 checkpoint to a MINIMAL ONNX graph for on-device inference.

    python3 model/training/export_v11_mobile.py \
        --ckpt model/weights/v11/best_v11_raw.pt \
        --out  model/weights/v11/cropguard_v11_mobile.onnx

Why not `export_onnx.py --mode full`? That graph also carries the detection head
(`det_cls`, `det_box`, 2100 anchors), and v11 has **no trained detection head** --
the server derives boxes from connected components of the predicted lesion mask
(`boxes_from_mask`) and never calls the det head. Carrying it would add size and
compute for outputs nothing reads.

What the graph computes (identical arithmetic to the server's forward with
`active_tasks={"cls","seg"}`, minus the unused anchor/det branches):

    c2,c3,c4,c5 = backbone(x)
    p3,p4,p5    = neck([c3,c4,c5])
    seg_logits  = seg_head(p3,p4,p5,c2,out_size=x.shape[-2:])
    cls_logits  = cls_head([p3,p4,p5], sigmoid(seg_logits[:,0:1]))[0]

Note the classification head is *lesion-gated*: it consumes sigmoid(seg ch0) as
its pooling mask, so cls_logits and seg_logits must come from ONE forward pass.
Everything else the API reports (softmax, temperature, energy/abstention gate,
severity, boxes, mask PNGs) is post-processing and is reimplemented in Dart --
ONNX cannot carry LayerCAM, which needs gradients and is therefore server-only.

Input  1x3x320x320 float32, RGB, letterboxed on gray-114, scaled /255 (no mean/std).
Output cls_logits 1x8 float32
       seg_logits 1x2x320x320 float32   (0 = lesion, 1 = leaf; apply sigmoid)
"""
import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.architecture.v7_model import CropGuardMultiTaskV7  # noqa: E402

NC = 8
IMG_SIZE = 320
LETTERBOX_FILL = 114


class MobileWrapper(torch.nn.Module):
    """cls + seg only, and without forward()'s det_sizes/anchors bookkeeping.

    Those branches are dead weight here: `_anchors()` builds meshgrids for three
    feature levels on every call, and the server never reads them for v11.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        c2, c3, c4, c5 = self.m.backbone(x)
        p3, p4, p5 = self.m.neck([c3, c4, c5])
        seg_logits = self.m.seg_head(p3, p4, p5, c2, out_size=x.shape[-2:])
        mask_prob = torch.sigmoid(seg_logits[:, 0:1])
        cls_logits, _ = self.m.cls_head([p3, p4, p5], mask_prob)
        return cls_logits, seg_logits


def load(ckpt_path: str, device="cpu"):
    model = CropGuardMultiTaskV7(nc=NC, pretrained=False).to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model"] if "model" in ck else (ck["state_dict"] if "state_dict" in ck else ck)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, ck


def letterbox(rgb: np.ndarray) -> np.ndarray:
    """Same letterbox the server does, via PIL/numpy (no cv2 dependency here)."""
    import cv2
    orig_h, orig_w = rgb.shape[:2]
    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    cw = max(1, int(round(orig_w * scale)))
    ch = max(1, int(round(orig_h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (cw, ch), interpolation=interp)
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), LETTERBOX_FILL, dtype=np.uint8)
    px, py = (IMG_SIZE - cw) // 2, (IMG_SIZE - ch) // 2
    canvas[py:py + ch, px:px + cw] = resized
    return canvas


def parity(model, onnx_path, x: torch.Tensor, tag: str) -> float:
    """Max abs difference between PyTorch and ONNX Runtime on the same input."""
    import onnxruntime as ort
    with torch.no_grad():
        t_cls, t_seg = model(x)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o_cls, o_seg = sess.run(None, {"image": x.numpy().astype(np.float32)})
    d_cls = float(np.abs(t_cls.numpy() - o_cls).max())
    d_seg = float(np.abs(t_seg.numpy() - o_seg).max())
    print(f"  {tag:<22} cls max|Δ|={d_cls:.3e}   seg max|Δ|={d_seg:.3e}")
    return max(d_cls, d_seg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, "model/weights/v11/best_v11_raw.pt"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "model/weights/v11/cropguard_v11_mobile.onnx"))
    ap.add_argument("--image", default=None, help="optional real image for a second parity pass")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    print(f"checkpoint: {args.ckpt}")
    model, ck = load(args.ckpt)
    epoch = ck.get("epoch") if isinstance(ck, dict) else None
    print(f"  epoch={epoch}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    wrap = MobileWrapper(model).eval()
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        wrap, x, args.out,
        input_names=["image"],
        output_names=["cls_logits", "seg_logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"exported -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

    worst = parity(wrap, args.out, x, "random input")

    if args.image and os.path.exists(args.image):
        import cv2
        bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(letterbox(rgb).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        worst = max(worst, parity(wrap, args.out, t, os.path.basename(args.image)))

    # Also confirm the graph runs at the batch=1 static shape it will see on device.
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()[0]
    print(f"  input  {ins.name} {ins.shape} {ins.type}")
    for o in sess.get_outputs():
        print(f"  output {o.name:<12} {o.shape} {o.type}")

    if worst > args.tolerance:
        print(f"FAIL: parity {worst:.3e} exceeds tolerance {args.tolerance:.1e}")
        return 1
    print(f"OK: worst parity {worst:.3e} <= {args.tolerance:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
