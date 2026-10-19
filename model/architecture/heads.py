import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class DecoupledDetHead(nn.Module):
    """YOLOv8-style: per level, separate cls and box branches (DFL box encoding)."""

    def __init__(self, width=96, nc=11, reg_max=8):
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
        # YOLOv8-style priors. Without the cls bias the initial detection
        # classification loss is enormous (~745 in the first epochs) because
        # 2100 anchors all start at p=0.5 against an almost all-background
        # target; that swamps the box/dfl gradients and the detector stalls.
        nn.init.constant_(self.cls_out.bias, -math.log((1 - 0.01) / 0.01))
        nn.init.constant_(self.box_out.bias, 1.0)

    def forward(self, p):
        """Returns (cls_logits B,nc,H,W, box_dfl B,4*reg_max,H,W)."""
        return self.cls_out(self.cls_b(p)), self.box_out(self.box_b(p))

# ------------------------------------------------------------- attention ---
class SEBlock(nn.Module):
    """Selective channel attention (paper sec 2.2 / 2.4.2).

    NOTE: this class was dropped during the model modularization refactor while
    ClassificationHead below kept referencing it, so `heads.py` raised
    `NameError: name 'SEBlock' is not defined` on construction. Restored here
    verbatim from _old_v2/model.py so the parameter names (`fc.1`, `fc.3`) and
    therefore the v5 checkpoint keys still line up.
    """

    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 8), 1), nn.GELU(),
            nn.Conv2d(max(c // r, 8), c, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)

# ------------------------------------------------------------- cls head ---
class ClassificationHead(nn.Module):
    """SE -> GAP -> GELU MLP (paper eqs 1-3), on ALL pyramid levels.

    Originally this read only P5 (stride 32 -> a 10x10 map at 320px). Tomato
    diseases are distinguished largely by lesion *texture* (early vs late
    blight vs septoria are all brown spots), and P5 has already thrown that
    detail away. Pooling P3 (40x40), P4 (20x20) and P5 (10x10) together gives
    the classifier both fine texture and semantic context. The fused vector is
    projected back to `width` so the severity head's dims are unchanged.
    """

    def __init__(self, width=96, nc=11, n_levels=3):
        super().__init__()
        self.n_levels = n_levels
        # r=8 (not the SEBlock default 16): this is the width the v5 checkpoint
        # was trained with -- cls_head.se.{0,1,2}.fc.1 is (12, width, 1, 1).
        self.se = nn.ModuleList([SEBlock(width, r=8) for _ in range(n_levels)])
        self.fuse = nn.Sequential(nn.Linear(width * n_levels, width), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, nc))

    def forward(self, levels):
        zs = [F.adaptive_avg_pool2d(se(p), 1).flatten(1)
              for se, p in zip(self.se, levels)]
        z = self.fuse(torch.cat(zs, dim=1))
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

