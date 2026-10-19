"""Cross-task consistency for v7 -- ONE term only.

Expert correction (accepted): the box<->mask consistency term was vacuous.
Our boxes are DERIVED from the lesion masks (connected components), so they
agree with the masks by construction and the loss taught nothing. The same
argument kills "severity vs detection density": detection density was itself
computed from a head trained on mask-derived boxes.

What survives is genuinely informative and cannot be satisfied by copying one
signal into the other:

    cls_cons: p(not healthy) must equal "there is lesion somewhere".

The classifier sees pooled features; the segmenter sees per-pixel evidence.
Neither can satisfy this by construction, so the term actually constrains.
"""
import torch
import torch.nn.functional as F

HEALTHY_IDX = 0          # index 0 of the 11-class tomato schema
LAMBDAS = {"cls_cons": 0.5}


def _soft_has_lesion(seg_logits_ch0, k=25.0):
    """Differentiable 'there is lesion somewhere', in [0,1].

    BUGFIX: the first version was `(mean(p * |2p-1|) * 20).clamp(0, 1)`, whose
    clamp SATURATES for any normal mask (mean ~0.1-0.25 -> 2-5 -> 1.0), so the
    gradient into the seg head was exactly zero and the term constrained
    nothing. This form is smooth and strictly increasing with no saturating
    clamp: small lesion areas move it a lot (any lesion should flip the
    classifier), large areas approach 1 asymptotically.
    """
    p = torch.sigmoid(seg_logits_ch0.float())
    mean_p = p.mean(dim=(1, 2, 3))
    return 1.0 - torch.exp(-k * mean_p)


def consistency_loss(outputs, batch=None, weights=None):
    """Scalar loss; 0.0 when the required heads did not run this step."""
    w = dict(LAMBDAS)
    if weights:
        w.update(weights)
    if not w or "cls_logits" not in outputs or "seg_logits" not in outputs:
        return torch.zeros((), device=outputs.get(
            "cls_logits", torch.zeros(1)).device)

    has_les = _soft_has_lesion(outputs["seg_logits"][:, 0:1])
    p_health = torch.softmax(outputs["cls_logits"].float(), -1)[:, HEALTHY_IDX]
    has_dis = 1.0 - p_health
    # BUGFIX: F.binary_cross_entropy is on torch's autocast "unsafe" list and
    # RAISES under torch.amp.autocast("cuda") even though both operands are
    # already fp32 (the .float() casts above). It only runs at stages >= 4
    # (the first joint det+cls+seg stage -- the term needs BOTH cls_logits and
    # seg_logits), so stages 1-3 trained fine and the crash appeared only at
    # stage 4, which is what killed every full v7 run. Compute the identical
    # formula directly on the clamped fp32 probabilities -- autocast-safe.
    a = has_dis.clamp(1e-4, 1 - 1e-4)
    b = has_les.clamp(0, 1)
    bce = -(b * torch.log(a) + (1.0 - b) * torch.log(1.0 - a)).mean()
    return w["cls_cons"] * bce
