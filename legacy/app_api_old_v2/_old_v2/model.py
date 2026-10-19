"""CropGuard multi-task model (paper: Crop Disease Detection Using Hybrid Deep Learning Model).

Backbone:  timm fasternet_t0, features_only, strides 8/16/32 -> C3(80) C4(160) C5(320)
Neck:      BiFPN-lite on P3/P4/P5 with learnable softmax-normalized fusion weights
Heads:     decoupled detection (cls+box DFL), classification (SE->GAP->MLP),
           LightHam-lite segmentation, gated multimodal severity
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- backbone ---
class Backbone(nn.Module):
    """fasternet_t0 feature pyramid: C3 @ /8 (80ch), C4 @ /16 (160ch), C5 @ /32 (320ch)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            self.net = timm.create_model(
                "fasternet_t0", pretrained=pretrained, features_only=True,
                out_indices=(1, 2, 3),
            )
        except Exception as e:  # HF Hub unreachable (Kaggle Internet off)
            if pretrained:
                print(f"[Backbone] pretrained download failed ({e}); "
                      "falling back to random init")
            self.net = timm.create_model(
                "fasternet_t0", pretrained=False, features_only=True,
                out_indices=(1, 2, 3),
            )
        self.channels = self.net.feature_info.channels()  # [80, 160, 320]

    def forward(self, x):
        return self.net(x)  # [C3, C4, C5]

# ---------------------------------------------------------------- attention ---
class SEBlock(nn.Module):
    """Selective channel attention (paper sec 2.2 / 2.4.2)."""

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 8), 1), nn.GELU(),
            nn.Conv2d(max(c // r, 8), c, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)

# ---------------------------------------------------------------- BiFPN-lite ---
class BiFPNLayer(nn.Module):
    """Weighted cross-scale fusion with softmax-normalized learnable weights."""

    def __init__(self, c: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, c))
        self.eps = 1e-4
        self.conv = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)  # depthwise
        self.pw = nn.Conv2d(c, c, 1)                                     # pointwise
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()

    def fuse(self, x1, x2):
        w = torch.relu(self.w)                       # (2, C)
        w = w / (w.sum(dim=0, keepdim=True) + self.eps)
        w0 = w[0][None, :, None, None]               # (1,C,1,1) for channel-wise
        w1 = w[1][None, :, None, None]
        return self.act(self.bn(self.pw(self.conv(w0 * x1 + w1 * x2))))

    forward = fuse


class BiFPNLite(nn.Module):
    """3-level bidirectional fusion: top-down then bottom-up, one round (lite)."""

    def __init__(self, channels=(80, 160, 320), width=96):
        super().__init__()
        self.in_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c, width, 1), nn.BatchNorm2d(width), nn.GELU())
             for c in channels]
        )
        self.lat = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(2)])
        self.td = nn.ModuleList([BiFPNLayer(width) for _ in range(2)])
        self.bu = nn.ModuleList([BiFPNLayer(width) for _ in range(2)])
        self.out_proj = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                           nn.Conv2d(width, width, 1), nn.BatchNorm2d(width), nn.GELU())
             for _ in range(3)]
        )

    def forward(self, feats):
        c3, c4, c5 = [p(f) for p, f in zip(self.in_proj, feats)]
        p5 = c5
        p4 = self.td[0](self.lat[0](c4), F.interpolate(p5, scale_factor=2, mode="nearest"))
        p3 = self.td[1](self.lat[1](c3), F.interpolate(p4, scale_factor=2, mode="nearest"))
        p4 = self.bu[0](c4, F.max_pool2d(p3, 2))
        p5 = self.bu[1](c5, F.max_pool2d(p4, 2))
        p3, p4, p5 = [o(p) for o, p in zip(self.out_proj, (p3, p4, p5))]
        return [p3, p4, p5]

# ------------------------------------------------------------- det head ---
class DecoupledDetHead(nn.Module):
    """YOLOv8-style: per level, separate cls and box branches (DFL box encoding)."""

    def __init__(self, width=96, nc=16, reg_max=8):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

        def branch():
            return nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
                nn.Conv2d(width, width, 1), nn.BatchNorm2d(width), nn.GELU(),
            )

        self.cls_b = branch()
        self.box_b = branch()
        self.cls_out = nn.Conv2d(width, nc, 1)
        self.box_out = nn.Conv2d(width, 4 * reg_max, 1)

    def forward(self, p):
        """Returns (cls_logits B,nc,H,W, box_dfl B,4*reg_max,H,W)."""
        return self.cls_out(self.cls_b(p)), self.box_out(self.box_b(p))

# ------------------------------------------------------------- cls head ---
class ClassificationHead(nn.Module):
    """SE -> GAP -> GELU MLP (paper eqs 1-3)."""

    def __init__(self, width=96, nc=16):
        super().__init__()
        self.se = SEBlock(width, r=8)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, nc))

    def forward(self, p5):
        z = self.se(p5)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.mlp(z), z  # (logits, embedding for severity fusion)

# ------------------------------------------------------------- seg head ---
class Hamburger(nn.Module):
    """LightHam-lite global context unit (channel-attention variant)."""

    def __init__(self, c, r=4):
        super().__init__()
        h = max(c // r, 16)
        self.inp = nn.Conv2d(c, c, 1)
        self.u = nn.Linear(c, h, bias=False)
        self.v = nn.Linear(c, h, bias=False)
        self.out = nn.Linear(h, c, bias=False)
        self.scale = h ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, hh, ww = x.shape
        x = self.inp(x)
        q = self.u(x.flatten(2).transpose(1, 2))          # B,N,h
        k = self.v(x.flatten(2).transpose(1, 2))          # B,N,h
        attn = torch.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)
        y = attn @ k                                       # B,N,h
        y = self.out(y).transpose(1, 2).reshape(b, c, hh, ww)
        return x + self.gamma * y


class LightHamLiteDecoder(nn.Module):
    """Concat pyramid -> Hamburger bottleneck -> transposed-conv upsampling to 1x."""

    def __init__(self, width=96):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(width * 3, width, 1), nn.BatchNorm2d(width), nn.GELU())
        self.hamburger = Hamburger(width)
        self.up1 = nn.ConvTranspose2d(width, width // 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(width // 2, width // 4, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(width // 4, width // 8, 2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(width // 8, width // 8, 3, padding=1, groups=width // 8, bias=False),
            nn.Conv2d(width // 8, width // 8, 1), nn.BatchNorm2d(width // 8), nn.GELU())
        self.head = nn.Conv2d(width // 8, 1, 1)

    def forward(self, p3, p4, p5, out_size):
        """out_size: (H, W) of the input image; logits upsampled to exactly that."""
        x = self.reduce(torch.cat([
            p3,
            F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1))
        x = self.hamburger(x)
        x = self.up1(x)                       # /4
        x = self.up2(x)                       # /2
        x = self.up3(x)                       # /1
        x = self.refine(x)
        x = self.head(x)
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

# ------------------------------------------------------------- sev head ---
class SeverityHead(nn.Module):
    """Gated multimodal fusion: [z_seg; z_det; z_cls; r] -> sigmoid gate -> MLP.
    Outputs continuous severity in [0,1] and ordinal logits (K=4)."""

    def __init__(self, z_seg=96, z_det=96, z_cls=96, k_ord=4):
        super().__init__()
        d = z_seg + z_det + z_cls + 1
        self.gate = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU())
        self.cont = nn.Linear(64, 1)
        self.ord = nn.Linear(64, k_ord)

    def forward(self, z_s, z_d, z_c, ratio):
        z = torch.cat([z_s, z_d, z_c, ratio.unsqueeze(1)], dim=1)
        g = torch.sigmoid(self.gate(z))
        h = self.mlp(g * z)
        return torch.sigmoid(self.cont(h)).squeeze(1), self.ord(h)

# ------------------------------------------------------------- full model ---
class CropGuardMultiTask(nn.Module):
    def __init__(self, nc=16, width=96, pretrained=True, reg_max=8):
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

        active_tasks: subset of {"det", "cls", "seg"} to actually compute.
        Inactive heads are NOT run at all -- prevents frozen-head activation
        overflow (FP16 Inf) and BN-stat poisoning during staged training.
        None = run everything (inference/validation)."""
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
            y_cls, z_c = self.cls_head(p5)
            out["cls_logits"], out["z_cls"] = y_cls, z_c
        seg_logits = None
        if "seg" in act:
            seg_logits = self.seg_head(p3, p4, p5, out_size=x.shape[-2:])
            out["seg_logits"] = seg_logits
        out["anchors"] = torch.cat(
            [_anchors(s, hw, x.device) for s, hw in zip([8, 16, 32], out["det_sizes"])],
            dim=0)
        if with_severity and seg_logits is not None:
            z_s = self.seg_embed(seg_logits)
            z_s = F.adaptive_avg_pool2d(z_s, 1).flatten(1)               # B,width
            z_d = F.adaptive_avg_pool2d(p5, 1).flatten(1)                # B,width
            prob = torch.sigmoid(seg_logits)
            ratio = prob.mean(dim=(1, 2, 3))                             # lesion area ratio
            sev_cont, sev_ord = self.sev_head(z_s, z_d, z_c, ratio)
            out["sev_cont"], out["sev_ord"], out["lesion_ratio"] = sev_cont, sev_ord, ratio
        return out


def _anchors(stride, hw, device=None):
    """Anchor centers (x,y) in pixels + stride; shape (h*w, 3)."""
    h, w = hw
    ys, xs = torch.meshgrid(
        torch.arange(h, device=device) + 0.5,
        torch.arange(w, device=device) + 0.5, indexing="ij")
    a = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=1) * stride
    return torch.cat([a, torch.full((h * w, 1), float(stride), device=device)], dim=1)
