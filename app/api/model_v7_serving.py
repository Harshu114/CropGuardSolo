"""CropGuard v7 model -- SERVING COPY (self-contained).

The API is run as a script from this directory (and containers use `app/api`
as build context), so the repo-root `model/` package is not importable here:
this file repeats the v7 definitions rather than importing them. After v7
training completes, `tests/test_architecture_consistency.py` will be extended
to assert this copy builds the same state_dict as `model/architecture/v7_model.py`
(same mechanism as the v5 pair).

Load with:
    from model_v7_serving import CropGuardMultiTaskV7
    m = CropGuardMultiTaskV7(nc=11)
    m.load_state_dict(torch.load("cropguard_model_v7.pt", weights_only=True))

NOTE: no v7 checkpoint exists yet -- this lands with the v7 training run.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

_LEVEL_INDICES = (1, 2, 3)   # C3 @/8, C4 @/16, C5 @/32   (v5-style neck inputs)
_LEVEL_INDICES_V7 = (0, 1, 2, 3)   # adds C2 @/4 for the seg skip
_STRIDES = (8, 16, 32)
_STRIDES_V7 = (4, 8, 16, 32)


class Backbone(nn.Module):
    def __init__(self, pretrained=True, levels=_LEVEL_INDICES_V7):
        super().__init__()
        try:
            self.net = timm.create_model("fasternet_t0", pretrained=pretrained,
                                         features_only=True, out_indices=levels)
        except Exception as e:
            if not pretrained:
                raise
            print(f"[Backbone] pretrained download failed ({e}); random init")
            self.net = timm.create_model("fasternet_t0", pretrained=False,
                                         features_only=True, out_indices=levels)
        self.channels = self.net.feature_info.channels()

    def forward(self, x):
        return self.net(x)


def _hardswish():
    return nn.Hardswish()


class BiFPNLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, c))
        self.eps = 1e-4
        self.conv = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()

    def fuse(self, x1, x2):
        w = torch.relu(self.w)
        w = w / (w.sum(dim=0, keepdim=True) + self.eps)
        return self.act(self.bn(self.pw(self.conv(
            w[0][None, :, None, None] * x1 + w[1][None, :, None, None] * x2))))

    forward = fuse


class BiFPNLite(nn.Module):
    def __init__(self, channels=(80, 160, 320), width=96):
        super().__init__()
        self.in_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, width, 1), nn.BatchNorm2d(width), nn.GELU())
             for c in channels])
        self.lat = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(2)])
        self.td = nn.ModuleList([BiFPNLayer(width) for _ in range(2)])
        self.bu = nn.ModuleList([BiFPNLayer(width) for _ in range(2)])
        self.out_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                           nn.Conv2d(width, width, 1), nn.BatchNorm2d(width), nn.GELU())
             for _ in range(3)])

    def forward(self, feats):
        c3, c4, c5 = [p(f) for p, f in zip(self.in_proj, feats)]
        p5 = c5
        p4 = self.td[0](self.lat[0](c4), F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3 = self.td[1](self.lat[1](c3), F.interpolate(p4, scale_factor=2, mode="nearest"))
        p4 = self.bu[0](c4, F.max_pool2d(p3, 2))
        p5 = self.bu[1](c5, F.max_pool2d(p4, 2))
        return [o(p) for o, p in zip(self.out_proj, (p3, p4, p5))]


class DecoupledDetHead(nn.Module):
    def __init__(self, width=96, nc=11, reg_max=8):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        def branch():
            return nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                nn.Conv2d(width, width, 1), nn.BatchNorm2d(width), nn.GELU())

        self.cls_b = branch()
        self.box_b = branch()
        self.cls_out = nn.Conv2d(width, nc, 1)
        self.box_out = nn.Conv2d(width, 4 * reg_max, 1)
        nn.init.constant_(self.cls_out.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.constant_(self.box_out.bias, 1.0)

    def forward(self, p):
        return self.cls_out(self.cls_b(p)), self.box_out(self.box_b(p))


class BilinearDecoder(nn.Module):
    """P3 + up(P4) + up(P5) + stride-4 C2 skip -> progressive upsampling.

    MUST stay identical to model/architecture/v7_heads.py:BilinearDecoder --
    tests/test_architecture_consistency.py compares this copy's state_dict
    against the canonical one, and the two diverged once already (the canonical
    gained this progressive up1/up2 path and a wider skip, the serving copy kept
    the old /4-wide single-interpolate decoder, so no v7 checkpoint could load
    here). Progressive upsampling:
      - Fuses multiscale P3/P4/P5 with stride-4 C2 skip at 80x80 (48ch)
      - Bilinear x2 + 3x3 conv -> 160x160 (32ch)
      - Bilinear to out_size + 3x3 conv -> 320x320 (24ch) -> 1x1 head
      - Keeps crisp lesion boundary features instead of one big bilinear blur.
    """

    def __init__(self, width=96, skip_ch=None, out_ch=2):
        super().__init__()
        self.out_ch = out_ch
        self.reduce = nn.Sequential(
            nn.Conv2d(width * 3, width, 1), nn.BatchNorm2d(width), _hardswish())
        # stride-4 skip: project C2 (40ch on fasternet_t0) down to width//2 (48ch)
        self.skip = nn.Sequential(
            nn.Conv2d(skip_ch if skip_ch is not None else 40, width // 2, 1),
            nn.BatchNorm2d(width // 2), _hardswish(),
            nn.Conv2d(width // 2, width // 2, 3, padding=1,
                      groups=width // 2, bias=False),
            nn.Conv2d(width // 2, width // 2, 1), nn.BatchNorm2d(width // 2),
            _hardswish())
        fused_ch = width // 2   # 48ch
        self.fuse_skip = nn.Sequential(
            nn.Conv2d(width + width // 2, fused_ch, 1),
            nn.BatchNorm2d(fused_ch), _hardswish())
        self.bw = nn.Sequential(
            nn.Conv2d(fused_ch, fused_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(fused_ch), _hardswish())
        # Progressive stage 1: 80x80 -> 160x160
        self.up1 = nn.Sequential(
            nn.Conv2d(fused_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), _hardswish())
        # Progressive stage 2: 160x160 -> 320x320
        self.up2 = nn.Sequential(
            nn.Conv2d(32, 24, 3, padding=1, bias=False),
            nn.BatchNorm2d(24), _hardswish())
        self.head = nn.Conv2d(24, out_ch, 1)

    def forward(self, p3, p4, p5, skip_feat, out_size):
        x = self.reduce(torch.cat([
            p3,
            F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))
        s = self.skip(skip_feat)                                   # /4, 48ch
        x4 = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
        x = self.fuse_skip(torch.cat([x4, s], dim=1))              # /4, 48ch
        x = self.bw(x)
        h4, w4 = x.shape[-2:]
        x = F.interpolate(x, size=(h4 * 2, w4 * 2), mode="bilinear",
                          align_corners=False)                     # /2, 160x160
        x = self.up1(x)                                            # 32ch
        x = F.interpolate(x, size=out_size, mode="bilinear",
                          align_corners=False)                     # 1x, 320x320
        x = self.up2(x)                                            # 24ch
        return self.head(x)


class LesionGatedClassificationHead(nn.Module):
    def __init__(self, width=96, nc=11, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        self.fuse = nn.Sequential(
            nn.Linear(width * n_levels * 2, width), _hardswish())
        self.mlp = nn.Sequential(nn.Linear(width, width), _hardswish(),
                                 nn.Linear(width, nc))

    @staticmethod
    def _masked_avg(feat, mask_small, eps=1e-4):
        wsum = mask_small.sum(dim=(2, 3), keepdim=True)
        weight = torch.where(
            wsum > eps, mask_small / (wsum + eps),
            torch.full_like(mask_small, 1.0 / (mask_small.shape[-1] * mask_small.shape[-2])))
        return (feat * weight).sum(dim=(2, 3))

    def forward(self, levels, mask=None):
        zs = []
        m = mask.detach() if mask is not None else None
        for p in levels:
            mi = (F.interpolate(m, size=p.shape[-2:], mode="bilinear",
                                align_corners=False) if m is not None else None)
            lesion = (self._masked_avg(p, mi) if mi is not None
                      else F.adaptive_avg_pool2d(p, 1).flatten(1))
            gap = F.adaptive_avg_pool2d(p, 1).flatten(1)
            zs.append(torch.cat([lesion, gap], dim=1))
        z = self.fuse(torch.cat(zs, dim=1))
        return self.mlp(z), z


class AbstainHead(nn.Module):
    def __init__(self, nc=11, init_temp=1.0):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(init_temp))

    @staticmethod
    def energy(logits, T=1.0):
        return -torch.logsumexp(logits.float() / T, dim=-1)

    def forward(self, cls_logits, return_probs=True):
        logits_t = cls_logits / self.temperature
        return torch.softmax(logits_t, dim=-1) if return_probs else logits_t

    def should_abstain(self, cls_logits, energy_thresh=None, prob_thresh=0.60):
        probs = self.forward(cls_logits)
        conf, pred = probs.max(-1)
        e = self.energy(cls_logits, self.temperature.item())
        reason = torch.zeros_like(conf, dtype=torch.bool)
        if energy_thresh is not None:
            reason = reason | (e > energy_thresh)
        return (reason | (conf < prob_thresh)), pred, conf, e


class CropGuardMultiTaskV7(nn.Module):
    SEV_EPS = 1e-3
    arch_name = "v7"

    def __init__(self, nc=11, width=96, pretrained=True, reg_max=8):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.width = width
        self.backbone = Backbone(pretrained, levels=_LEVEL_INDICES_V7)
        self.neck = BiFPNLite(self.backbone.channels[1:], width)
        self.det_head = DecoupledDetHead(width, nc, reg_max)
        self.cls_head = LesionGatedClassificationHead(width, nc)
        self.seg_head = BilinearDecoder(width, skip_ch=self.backbone.channels[0])
        self.abstain = AbstainHead(nc)

    @staticmethod
    def _seg_channels(seg_logits):
        return seg_logits[:, 0:1], seg_logits[:, 1:2]

    def analytic_severity(self, seg_logits, eps=None):
        # Arithmetically identical to model/architecture/v7_model.py by design
        # (tests/test_architecture_consistency.py compares forward, which includes
        # sev_cont). The numerator is intersected with the leaf by the SOFT
        # product because the model paints lesion pixels outside the leaf (val:
        # mean 1672 px, raw ratio > 1 on 11/156 images) and the old clamp hid
        # that by pinning those predictions at 1.000. Measured on v9:
        # val severity MAE 0.1508 -> 0.1296, r 0.7539 -> 0.7733.
        eps = eps or self.SEV_EPS
        p = torch.sigmoid(seg_logits.float())
        lesion, leaf = self._seg_channels(p)
        lden = (lesion * leaf).sum(dim=(2, 3))
        lfdn = leaf.sum(dim=(2, 3)) + eps
        sev = (lden / lfdn).clamp(0, 1)
        return sev.flatten(), (lden / (lfdn)).squeeze(1)

    def forward(self, x, with_analysis=False, active_tasks=None,
                return_abstain=False, with_severity=None):
        feats = self.backbone(x)                     # [C2, C3, C4, C5]
        c2, c3, c4, c5 = feats
        p3, p4, p5 = self.neck([c3, c4, c5])
        act = active_tasks or {"det", "cls", "seg"}
        out = {}
        if "det" in act:
            dets = [self.det_head(p) for p in (p3, p4, p5)]
            out["det_cls"] = torch.cat([d[0].flatten(2).transpose(1, 2) for d in dets], dim=1)
            out["det_box"] = torch.cat([d[1].flatten(2).transpose(1, 2) for d in dets], dim=1)
        out["det_sizes"] = [(p3.shape[-2], p3.shape[-1]),
                            (p4.shape[-2], p4.shape[-1]),
                            (p5.shape[-2], p5.shape[-1])]
        seg_logits = None
        if "seg" in act:
            seg_logits = self.seg_head(p3, p4, p5, c2, out_size=x.shape[-2:])
            out["seg_logits"] = seg_logits
        if "cls" in act:
            mask_prob = (torch.sigmoid(seg_logits[:, 0:1])
                         if seg_logits is not None else None)
            y_cls, z_c = self.cls_head([p3, p4, p5], mask_prob)
            out["cls_logits"], out["z_cls"] = y_cls, z_c
        out["anchors"] = torch.cat(
            [_anchors(s, hw, x.device) for s, hw in zip(_STRIDES_V7[1:],
                                                        out["det_sizes"])], dim=0)
        if with_analysis and seg_logits is not None:
            with torch.no_grad():
                sev, ratio = self.analytic_severity(seg_logits)
            out["sev_cont"], out["lesion_ratio"] = sev, ratio
        if return_abstain:
            probs = self.abstain(out["cls_logits"])
            out["cls_probs"] = probs
            out["cls_energy"] = self.abstain.energy(out["cls_logits"],
                                                    self.abstain.temperature.item())
        return out


def boxes_from_mask(lesion_prob, min_area=24, max_components=8, thr=0.5):
    """Boxes straight from the predicted lesion mask -- no detector needed.

    This is the same connected-component geometry the training labels were
    derived from (derive_dataset.boxes_from_lesion_mask), so it is consistent
    with the ground truth by construction. The detection head is trained and
    kept for the drone / wide-shot case, but on-device this path drops the
    assigner, 2,100 anchors and NMS entirely.

    lesion_prob: (H,W) numpy array in [0,1]. Returns normalized (cx,cy,w,h)
    plus area fractions, or [] when the mask is empty.
    """
    import cv2
    import numpy as np
    m = (np.asarray(lesion_prob) > thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return []
    h, w = m.shape[:2]
    out = []
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:max_components]:
        x, y, bw, bh, area = stats[i + 1]
        if area < min_area:
            continue
        out.append({"cx": (x + bw / 2) / w, "cy": (y + bh / 2) / h,
                    "w": bw / w, "h": bh / h,
                    "area_frac": float(area) / (h * w)})
    return out


def _anchors(stride, hw, device=None):
    h, w = hw
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device) + 0.5,
        torch.arange(w, device=device) + 0.5, indexing="ij")
    a = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1) * stride
    return torch.cat([a, torch.full((h * w, 1), float(stride), device=device)], dim=1)
