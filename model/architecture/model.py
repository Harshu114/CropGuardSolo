"""CropGuard multi-task model assembly.

Backbone:  timm fasternet_t0, features_only, strides 8/16/32 -> C3(80) C4(160) C5(320)
Neck:      BiFPN-lite on P3/P4/P5 with learnable softmax-normalized fusion weights
Heads:     decoupled detection (cls + DFL box), classification (SE -> GAP -> MLP,
           pooled from P3+P4+P5), LightHam-lite segmentation, gated multimodal
           severity

This is the architecture that `cropguard_model_v5.pt` was trained with, and the
single source of truth for it. Two things were removed during the earlier
"modularization" pass because they were never part of the trained checkpoint
(verified: the v5 state_dict has no `guide.*` and no `adapters.*` keys) but had
been left in the serving copy, where they made it impossible to load the model:

  * a "Guided Upsampling Module (GUM)" that passed P2 as a segmentation guide,
  * per-task residual adapters (TaskAdapter), which does not exist in heads.py.

If those are reinstated as research, they must be added here AND the checkpoint
must be retrained -- the shipped weights have no such parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone, _STRIDES
from .neck import BiFPNLite
from .heads import (
    DecoupledDetHead,
    ClassificationHead,
    LightHamLiteDecoder,
    SeverityHead,
)


def _anchors(stride, hw, device=None):
    """Anchor centers (x, y) in pixels + stride; shape (h*w, 3)."""
    h, w = hw
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device) + 0.5,
        torch.arange(w, device=device) + 0.5, indexing="ij")
    a = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1) * stride
    return torch.cat([a, torch.full((h * w, 1), float(stride), device=device)], dim=1)


class CropGuardMultiTask(nn.Module):
    def __init__(self, nc=11, width=96, pretrained=True, reg_max=8):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.width = width
        self.backbone = Backbone(pretrained)
        self.neck = BiFPNLite(self.backbone.channels, width)
        self.det_head = DecoupledDetHead(width, nc, reg_max)
        self.cls_head = ClassificationHead(width, nc)
        self.seg_head = LightHamLiteDecoder(width)
        # projects 1-channel seg logits to a width vector for severity fusion
        self.seg_embed = nn.Sequential(nn.Conv2d(1, width, 1), nn.GELU())
        self.sev_head = SeverityHead(width, width, width)

    def forward(self, x, with_severity=False, active_tasks=None):
        """Returns dict with det cls/dfl (concat levels), cls logits, seg logits,
        anchors; optionally severity outputs.

        active_tasks: subset of {"det","cls","seg"} to actually compute.
        Inactive heads are NOT run at all -- prevents frozen-head activation
        overflow (FP16 Inf) and BN-stat poisoning during staged training.
        None = run everything (inference/validation).
        """
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.neck([c3, c4, c5])
        act = active_tasks or {"det", "cls", "seg"}
        out = {}
        if "det" in act:
            dets = [self.det_head(p) for p in (p3, p4, p5)]
            out["det_cls"] = torch.cat(
                [d[0].flatten(2).transpose(1, 2) for d in dets], dim=1)  # B,A,nc
            out["det_box"] = torch.cat(
                [d[1].flatten(2).transpose(1, 2) for d in dets], dim=1)  # B,A,4*reg_max
        out["det_sizes"] = [(p3.shape[-2], p3.shape[-1]),
                            (p4.shape[-2], p4.shape[-1]),
                            (p5.shape[-2], p5.shape[-1])]
        if "cls" in act:
            # Multi-scale: pool P3+P4+P5 (the v5 head), not P5 alone.
            y_cls, z_c = self.cls_head([p3, p4, p5])
            out["cls_logits"], out["z_cls"] = y_cls, z_c
        seg_logits = None
        if "seg" in act:
            seg_logits = self.seg_head(p3, p4, p5, out_size=x.shape[-2:])
            out["seg_logits"] = seg_logits
        out["anchors"] = torch.cat(
            [_anchors(s, hw, x.device) for s, hw in zip(_STRIDES, out["det_sizes"])],
            dim=0)
        if with_severity and seg_logits is not None:
            z_s = self.seg_embed(seg_logits)
            z_s = F.adaptive_avg_pool2d(z_s, 1).flatten(1)               # B,width
            z_d = F.adaptive_avg_pool2d(p5, 1).flatten(1)                # B,width
            prob = torch.sigmoid(seg_logits)
            ratio = prob.mean(dim=(1, 2, 3))                             # leaf area ratio
            # z_c is required by SeverityHead; if cls was skipped, fall back to P5.
            z_c_final = out.get("z_cls", F.adaptive_avg_pool2d(p5, 1).flatten(1))
            sev_cont, sev_ord = self.sev_head(z_s, z_d, z_c_final, ratio)
            out["sev_cont"], out["sev_ord"], out["lesion_ratio"] = sev_cont, sev_ord, ratio
        return out
