import torch
import torch.nn as nn
import timm

# Strides for the three levels we expose: C3 @ /8, C4 @ /16, C5 @ /32.
# These feed the neck, and their channel counts are read from timm at runtime
# (fasternet_t0 -> [80, 160, 320]).
_LEVEL_INDICES = (1, 2, 3)
_STRIDES = (8, 16, 32)

# v7: an extra stride-4 level for the segmentation skip connection.
# out_indices (0, 1, 2, 3) on fasternet_t0 yields [40, 80, 160, 320];
# v7 takes the extra C2 (stride 4, 40ch) and feeds it to BilinearDecoder.skip
# so small lesions survive upsampling. v5 keeps (1, 2, 3).
_LEVEL_INDICES_V7 = (0, 1, 2, 3)
_STRIDES_V7 = (4, 8, 16, 32)


class Backbone(nn.Module):
    """fasternet_t0 feature pyramid: C3 @ /8 (80ch), C4 @ /16 (160ch), C5 @ /32 (320ch).

    Emits exactly THREE feature maps for v5. This matters: the neck builds one
    input projection per level from `channels`, and the multi-task model unpacks
    `c3, c4, c5 = backbone(x)`. Requesting `out_indices=(0, 1, 2, 3)` (as the
    first version of this file did) yields FOUR maps [40, 80, 160, 320] and the
    unpack raises, while the offline fallback below used `(1, 2, 3)` -> the two
    code paths disagreed on arity. v7 asks for (0, 1, 2, 3) on purpose and
    unpacks four.

    pretrained weights: a DOWNLOAD FAILURE IS A HARD ERROR by default. The old
    silent fallback to random init meant a Kaggle run with Internet off could
    train a 30-epoch model on random features and report normal-looking losses,
    wasting the whole budget. Pass allow_random=True to opt out explicitly
    (smoke tests only).
    """

    def __init__(self, pretrained: bool = True, levels=_LEVEL_INDICES,
                 allow_random: bool = False):
        super().__init__()
        try:
            self.net = timm.create_model(
                "fasternet_t0", pretrained=pretrained, features_only=True,
                out_indices=levels,
            )
        except Exception as e:  # HF Hub unreachable (e.g. Kaggle with Internet off)
            if not pretrained or not allow_random:
                raise RuntimeError(
                    f"fasternet_t0 pretrained weights unavailable ({e}). "
                    "Pre-download them and attach as a Kaggle dataset, or pass "
                    "allow_random=True to train from random init (not advised)."
                ) from e
            print(f"[Backbone] pretrained download failed ({e}); "
                  "falling back to random init (allow_random=True)")
            # Same out_indices as the primary path -- only the weights change.
            self.net = timm.create_model(
                "fasternet_t0", pretrained=False, features_only=True,
                out_indices=levels,
            )
        self.channels = self.net.feature_info.channels()  # v5 [80,160,320]

    def forward(self, x):
        return self.net(x)  # v5: [C3, C4, C5];  v7: [C2, C3, C4, C5]
