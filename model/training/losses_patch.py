def _dice_bce(logits, target):
    if target.dim() == 3:
        target = target[:, None]
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:],
                               mode="bilinear", align_corners=False)

    # EXCLUSION: Only compute loss on pixels where target != -1.0
    # target is (B, 1, H, W), logits is (B, 1, H, W)
    valid_mask = (target != -1.0).float()

    # BCE loss: manually weighted by valid_mask
    # F.binary_cross_entropy_with_logits has no mask, so we do:
    bce_raw = F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1), reduction="none")
    bce = (bce_raw * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    prob = logits.sigmoid()
    # Dice loss: only sum intersection/union over valid pixels
    inter = (prob * target.clamp(0, 1) * valid_mask).sum(dim=(2, 3))
    union = (prob * valid_mask).sum(dim=(2, 3)) + (target.clamp(0, 1) * valid_mask).sum(dim=(2, 3))
    dice = 1 - ((2 * inter + 1) / (union + 1)).mean()

    return bce + dice
