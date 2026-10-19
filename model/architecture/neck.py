import torch
import torch.nn as nn
import torch.nn.functional as F
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

