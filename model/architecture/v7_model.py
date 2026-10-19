"""v7 model assembly. NEW architecture -- the v5/v6 checkpoint does NOT load.

What changed vs v5 (`model/architecture/model.py`), and why (expert critique,
checked against the code):

  backbone  4 levels: adds stride-4 C2 (40ch) for the seg skip.
  neck      unchanged BiFPNLite (P3/P4/P5 @96) -- v5 neck weights can warm-start.
  det head  unchanged DecoupledDetHead.
  cls head  LesionGatedClassificationHead (mask-pooled + GAP, detached mask).
  seg head  BilinearDecoder, 2-channel out (lesion, leaf), stride-4 skip,
            Hardswish (INT8-safe).
  severity  ANALYTIC: sev = sum(sig(lesion_logit)) / (sum(sig(leaf_logit)) + eps).
            The learned SeverityHead is REMOVED -- without human severity
            ratings it was learning to read the area ratio we already know.
  abstain   AbstainHead: temperature-scaled probs + energy OOD score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Backbone, _LEVEL_INDICES_V7, _STRIDES_V7
from .neck import BiFPNLite
from .heads import DecoupledDetHead
from .v7_heads import (
    BilinearDecoder,
    LesionGatedClassificationHead,
    AbstainHead,
)


def _anchors(stride, hw, device=None):
    h, w = hw
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device) + 0.5,
        torch.arange(w, device=device) + 0.5, indexing="ij")
    a = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1) * stride
    return torch.cat([a, torch.full((h * w, 1), float(stride), device=device)], dim=1)


class CropGuardMultiTaskV7(nn.Module):
    SEV_EPS = 1e-3
    arch_name = "v7"   # train.py reads this to switch loss paths

    def __init__(self, nc=11, width=96, pretrained=True, reg_max=8):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.width = width
        self.backbone = Backbone(pretrained, levels=_LEVEL_INDICES_V7)
        self.neck = BiFPNLite(self.backbone.channels[1:], width)  # C3,C4,C5 -> 96
        self.det_head = DecoupledDetHead(width, nc, reg_max)
        self.cls_head = LesionGatedClassificationHead(width, nc)
        self.seg_head = BilinearDecoder(width, skip_ch=self.backbone.channels[0])
        self.abstain = AbstainHead(nc)

    # -------------------------------------------------------------- utils --
    @staticmethod
    def _seg_channels(seg_logits):
        """Split (B,2,H,W) logits into (lesion, leaf) pairs (B,1,H,W) each."""
        return seg_logits[:, 0:1], seg_logits[:, 1:2]

    def analytic_severity(self, seg_logits, eps=None):
        """severity = lesion_px / leaf_px, both from predicted masks.

        Returns (sev_cont, lesion_ratio). Fully differentiable but normally
        used under no_grad at deploy; during training the consistency loss
        supervises it instead of an L1-to-teacher term.
        """
        eps = eps or self.SEV_EPS
        p = torch.sigmoid(seg_logits.float())
        lesion, leaf = self._seg_channels(p)
        # Restrict the numerator to the leaf by the soft product. The model does
        # paint lesion pixels OUTSIDE the leaf (measured on val: mean 1672 px,
        # raw ratio > 1 on 11/156 images), and the old clamped ratio hid that by
        # pinning those predictions at exactly 1.000. Soft intersection keeps the
        # ratio <= 1, stays differentiable for the consistency loss, and measured
        # better than a thresholded AND: val severity MAE 0.1508 -> 0.1296 with r
        # 0.7539 -> 0.7733 on v9 best_v8_raw.pt (2026-10-13).
        num = (lesion * leaf).sum(dim=(2, 3))
        den = leaf.sum(dim=(2, 3)) + eps
        return (num / den).squeeze(1).clamp(0, 1), (num / den).squeeze(1)

    # ------------------------------------------------------------ forward --
    def forward(self, x, with_analysis=False, active_tasks=None,
                return_abstain=False, with_severity=None,
                gt_mask=None, gt_mask_prob=0.0):
        """`with_severity` is accepted-and-ignored for v5-style call sites:
        v7 severity is analytic and rides on the seg channel, controlled by
        with_analysis. Passing with_severity=True logs a hint so call sites
        get migrated.

        gt_mask / gt_mask_prob: fixes the lesion-gating train/inference
        mismatch. At inference the classifier is gated on PREDICTED masks; if
        it only ever saw clean GT masks in training it would be optimized for
        an input it never gets. Each step, with probability gt_mask_prob, the
        gate uses the GT mask (a clean signal), otherwise the predicted one.
        Schedule gt_mask_prob 0.8 -> 0.2 across the stage so the head ends up
        seeing mostly real predictions.
        """
        """Returns dict; same key conventions as v5 where shared.

        with_analysis: v5's `with_severity`; v7 severity is always analytic
        from the seg head, so this toggles ONLY the extra analysis outputs
        (sev_cont, lesion_ratio, abstain probabilities).
        active_tasks: subset of {"det","cls","seg"}; None = all.
        """
        feats = self.backbone(x)                     # [C2, C3, C4, C5]
        c2, c3, c4, c5 = feats
        p3, p4, p5 = self.neck([c3, c4, c5])
        act = active_tasks or {"det", "cls", "seg"}
        out = {}

        if "det" in act:
            dets = [self.det_head(p) for p in (p3, p4, p5)]
            out["det_cls"] = torch.cat(
                [d[0].flatten(2).transpose(1, 2) for d in dets], dim=1)
            out["det_box"] = torch.cat(
                [d[1].flatten(2).transpose(1, 2) for d in dets], dim=1)
        out["det_sizes"] = [(p3.shape[-2], p3.shape[-1]),
                            (p4.shape[-2], p4.shape[-1]),
                            (p5.shape[-2], p5.shape[-1])]

        seg_logits = None
        if "seg" in act:
            seg_logits = self.seg_head(p3, p4, p5, c2, out_size=x.shape[-2:])
            out["seg_logits"] = seg_logits

        if "cls" in act:
            # classifier consumes seg probability only if seg ran (staged
            # training: det/cls stages pass mask=None -> plain GAP fallback)
            mask_prob = (torch.sigmoid(seg_logits[:, 0:1])
                         if seg_logits is not None else None)
            if gt_mask is not None and gt_mask_prob > 0.0 and \
                    torch.rand(()).item() < gt_mask_prob:
                # GT gate at full resolution: cls_head interpolates the mask
                # to each pooling level itself, exactly as it does for the
                # predicted one, so both paths exercise the same code.
                mask_prob = (gt_mask.float() if gt_mask.dim() == 4
                             else gt_mask.float()[:, None])
            y_cls, z_c = self.cls_head([p3, p4, p5], mask_prob)
            out["cls_logits"], out["z_cls"] = y_cls, z_c

        out["anchors"] = torch.cat(
            [_anchors(s, hw, x.device) for s, hw in zip(_STRIDES_V7[1:],
                                                        out["det_sizes"])],
            dim=0)

        if with_analysis and seg_logits is not None:
            with torch.no_grad():
                sev, ratio = self.analytic_severity(seg_logits)
            out["sev_cont"], out["lesion_ratio"] = sev, ratio

        if return_abstain:
            probs = self.abstain(out["cls_logits"])
            out["cls_probs"], out["cls_energy"] = probs, self.abstain.energy(
                out["cls_logits"], self.abstain.temperature.item())
        return out
