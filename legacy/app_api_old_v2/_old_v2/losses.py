"""Multi-task losses for CropGuard (paper eqs. 4, 10, 11) + PCGrad.

Detection:  TAL assignment (simplified YOLOv8) + BCE cls + CIoU box + DFL box
Cls:        CrossEntropy (ignore background)
Seg:        Dice + BCE (logits)
Severity:   L1 (continuous) + CrossEntropy (ordinal stages)
Total:      weighted sum, stage-dependent lambdas; PCGrad optional.
"""
import torch
import torch.nn.functional as F

# ------------------------------------------------------------- box utils ---
def box_xywh_to_xyxy(b):
    """(...,4) xywh -> xyxy (normalized or pixels, agnostic)."""
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def ciou_loss(pred, target, eps=1e-3):
    """Pairwise or elementwise CIoU loss (1 - CIoU). Boxes xyxy, same shape.
    Geometry calculations are done in FP32 to avoid numerical instability
    under AMP/FP16 (predicted boxes can become extremely thin)."""
    pred = pred.float()
    target = target.float()
    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    # intersection
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    parea = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    tarea = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union = parea + tarea - inter + eps
    iou = inter / union
    # smallest enclosing box
    cx1 = torch.min(px1, tx1); cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2); cy2 = torch.max(py2, ty2)
    c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + eps
    pcx = (px1 + px2) / 2; pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2; tcy = (ty1 + ty2) / 2
    d2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    # Stable aspect-ratio calculation with clamped dimensions
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)
    v = (4 / (torch.pi ** 2)) * ((torch.atan(pw / ph) -
                                 torch.atan(tw / th)) ** 2)
    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)
    return (1 - iou + alpha * (d2 / c2 + v)).clamp(min=0)


def dfl_decode(box_dfl, anchors, img_size, reg_max=8):
    """DFL bins -> normalized xywh boxes.
    box_dfl: (B,A,4*reg_max); anchors: (A,3) x,y,stride (pixels)."""
    B, A, _ = box_dfl.shape
    prob = box_dfl.view(B, A, 4, reg_max).softmax(-1)
    reg = (prob * torch.arange(reg_max, device=box_dfl.device).float()).sum(-1)  # B,A,4 ltrb in strides
    l, t, r, b = reg.unbind(-1)
    x, y, s = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    x1 = (x - l * s) / img_size
    y1 = (y - t * s) / img_size
    x2 = (x + r * s) / img_size
    y2 = (y + b * s) / img_size
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1).clamp(min=1e-4)
    h = (y2 - y1).clamp(min=1e-4)
    return torch.stack([cx, cy, w, h], dim=-1)  # B,A,4 normalized xywh

# ------------------------------------------------------------- assigner ---
@torch.no_grad()
def tal_assign(pred_xywh, pred_cls, gt_boxes, gt_labels, gt_mask, topk=10,
               alpha=0.5, beta=2.0):
    """Simplified TaskAlignedAssigner (YOLOv8-style), batched.
    beta=2 (not 6): with IoU~0.6 typical early in training, IoU**6 ~ 0.05
    crushes the align metric and hence the soft cls targets -- the model then
    learns to output conf ~0.2 for TRUE positives, wrecking PR ranking.
    pred_xywh: (B,A,4) normalized decoded boxes
    pred_cls:  (B,A,nc) sigmoid scores
    gt_boxes:  (B,G,4) normalized xywh, padded
    gt_labels: (B,G) long, padded
    gt_mask:   (B,G) bool (True = real gt)
    Returns: fg_mask (B,A), gt_idx (B,A) long (-1 bg), align (B,A,G) metric."""
    B, A, nc = pred_cls.shape
    G = gt_boxes.shape[1]
    device = pred_cls.device

    pred_xyxy = box_xywh_to_xyxy(pred_xywh)                    # B,A,4
    gt_xyxy = box_xywh_to_xyxy(gt_boxes.clamp(min=0, max=1))   # B,G,4

    # pairwise IoU B,A,G
    px1, py1, px2, py2 = pred_xyxy.unbind(-1)
    tx1, ty1, tx2, ty2 = gt_xyxy.unbind(-1)
    ix1 = torch.max(px1[:, :, None], tx1[:, None, :])
    iy1 = torch.max(py1[:, :, None], ty1[:, None, :])
    ix2 = torch.min(px2[:, :, None], tx2[:, None, :])
    iy2 = torch.min(py2[:, :, None], ty2[:, None, :])
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
    parea = ((px2 - px1) * (py2 - py1)).clamp(min=0)[:, :, None]
    tarea = ((tx2 - tx1) * (ty2 - ty1)).clamp(min=0)[:, None, :]
    iou = inter / (parea + tarea - inter + 1e-7)               # B,A,G

    # center-in-gt mask
    acx, acy = px1 + (px2 - px1) / 2, py1 + (py2 - py1) / 2    # B,A
    in_x = (acx[:, :, None] >= tx1[:, None, :]) & (acx[:, :, None] <= tx2[:, None, :])
    in_y = (acy[:, :, None] >= ty1[:, None, :]) & (acy[:, :, None] <= ty2[:, None, :])
    inside = in_x & in_y                                       # B,A,G

    # align metric: cls score for the gt class ^alpha * iou ^beta
    gt_cls_safe = gt_labels.clamp(min=0)
    score = pred_cls.gather(
        2, gt_cls_safe[:, None, :].expand(B, A, G)
    )                                                          # B,A,G
    align = (score ** alpha) * (iou.clamp(min=0) ** beta)
    align = align * inside * gt_mask[:, None, :]

    # top-k anchors per gt
    k = min(topk, A)
    topv, topi = align.topk(k, dim=1)                          # B,k,G
    cand = torch.zeros_like(align, dtype=torch.bool)
    cand.scatter_(1, topi, topv > 0)
    # resolve conflicts: anchor keeps gt with highest align
    best_align, best_gt = align.max(dim=2)                     # B,A
    fg_mask = cand.any(dim=2) & (best_align > 0)               # B,A
    gt_idx = torch.where(fg_mask, best_gt, torch.full_like(best_gt, -1))
    return fg_mask, gt_idx, align

# ------------------------------------------------------------- det loss ---
class DetectionLoss(torch.nn.Module):
    def __init__(self, nc=16, reg_max=8):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max

    def forward(self, det_cls, det_box, anchors, targets, img_size):
        """det_cls (B,A,nc) logits; det_box (B,A,4*reg_max);
        targets: list of (n_i,5) tensors [cls,cx,cy,w,h] normalized."""
        B, A, _ = det_cls.shape
        device = det_cls.device
        G = max(int(t.shape[0]) for t in targets) or 1
        gt_boxes = torch.zeros(B, G, 4, device=device)
        gt_labels = torch.zeros(B, G, dtype=torch.long, device=device)
        gt_mask = torch.zeros(B, G, dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            n = t.shape[0]
            if n:
                gt_boxes[i, :n] = t[:, 1:5].to(device)
                gt_labels[i, :n] = t[:, 0].long().to(device)
                gt_mask[i, :n] = True

        pred_xywh = dfl_decode(det_box, anchors, img_size, self.reg_max)
        fg_mask, gt_idx, align = tal_assign(pred_xywh, det_cls.sigmoid(),
                                            gt_boxes, gt_labels, gt_mask)
        n_fg = fg_mask.sum().clamp(min=1).float()
        bi, ai = fg_mask.nonzero(as_tuple=True)
        gi = gt_idx[bi, ai]

        # ---- classification: one-hot * normalized align (v8-style soft targets)
        target_scores = torch.zeros(B, A, self.nc, device=device)
        if fg_mask.any():
            # PER-GT normalization: each GT's best anchor targets 1.0. The old
            # batch-max scaling left soft targets ~0.2 early in training, so
            # TRUE positives learned low confidence -> bad PR ranking (the
            # r2 diagnosis: recall@0.5 = 0.95 but TP scores mean 0.23).
            align_per_gt = align / align.amax(dim=1, keepdim=True).clamp(min=1e-9)
            a_norm = torch.zeros(B, A, device=device)
            a_norm[bi, ai] = align_per_gt[bi, ai, gi]
            onehot = F.one_hot(gt_labels.clamp(min=0), self.nc).float()        # B,G,nc
            per_anchor_cls = onehot[
                torch.arange(B, device=device)[:, None],
                gt_idx.clamp(min=0)]                                           # B,A,nc
            target_scores = per_anchor_cls * (a_norm[..., None] * fg_mask[..., None])
        if fg_mask.any():
            loss_cls = F.binary_cross_entropy_with_logits(
                det_cls, target_scores, reduction="sum") / n_fg
        else:
            # all-background batch: teach full background suppression, mean-scaled
            loss_cls = F.binary_cross_entropy_with_logits(
                det_cls, target_scores, reduction="mean")

        # ---- box + dfl on fg anchors (zero-grad placeholder keeps graph alive)
        loss_box = det_cls.sum() * 0.0
        loss_dfl = det_cls.sum() * 0.0
        if fg_mask.any():
            bi, ai = fg_mask.nonzero(as_tuple=True)
            gi = gt_idx[bi, ai]
            tgt_xywh = gt_boxes[bi, gi]                                        # Nf,4
            pred_xywh_fg = pred_xywh[bi, ai]
            tgt_xyxy = box_xywh_to_xyxy(tgt_xywh) * img_size
            pred_xyxy = box_xywh_to_xyxy(pred_xywh_fg) * img_size
            loss_box = ciou_loss(pred_xyxy, tgt_xyxy).sum() / n_fg

            # dfl target: ltrb distances from the anchor to each box edge,
            # in stride units. The assigned anchor lies INSIDE the gt box
            # (TAL center-in-gt), so all four are >= 0 BEFORE clamping:
            #   l = (ax - x1)/s    decode: x1 = ax - l*s
            #   t = (ay - y1)/s            y1 = ay - t*s
            #   r = (x2 - ax)/s            x2 = ax + r*s
            #   b = (y2 - ay)/s            y2 = ay + b*s
            s = anchors[ai, 2]
            cx, cy, w, h = tgt_xywh.unbind(-1)
            x1, y1 = (cx - w / 2) * img_size, (cy - h / 2) * img_size
            x2, y2 = (cx + w / 2) * img_size, (cy + h / 2) * img_size
            ltrb = torch.stack([(anchors[ai, 0] - x1) / s,
                                (anchors[ai, 1] - y1) / s,
                                (x2 - anchors[ai, 0]) / s,
                                (y2 - anchors[ai, 1]) / s],
                               dim=-1).clamp(0, self.reg_max - 1 - 1e-4)
            # standard two-bin DFL: floor & ceil bins with linear weights
            prob = det_box[bi, ai].view(-1, 4, self.reg_max).softmax(-1)       # Nf,4,R
            logp = prob.clamp(min=1e-7).log()
            l = ltrb.floor().clamp(0, self.reg_max - 2).long()                 # Nf,4
            frac = ltrb - l.float()                                            # Nf,4 in [0,1)
            ce_l = torch.gather(logp, 2, l.unsqueeze(-1)).squeeze(-1)          # log p(floor)
            ce_u = torch.gather(logp, 2, (l + 1).unsqueeze(-1)).squeeze(-1)    # log p(ceil)
            loss_dfl = (-((1 - frac) * ce_l + frac * ce_u)).sum() / n_fg

        return {"det_cls": loss_cls, "det_box": loss_box, "det_dfl": loss_dfl}

# ------------------------------------------------------------- seg loss ---
def seg_loss_fn(seg_logits, mask):
    """Dice + BCE on logits; mask (B,1?,H,W) or (B,H,W)."""
    if mask.dim() == 3:
        mask = mask[:, None]
    bce = F.binary_cross_entropy_with_logits(seg_logits, mask)
    prob = seg_logits.sigmoid()
    inter = (prob * mask).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = 1 - ((2 * inter + 1) / (union + 1)).mean()
    return bce + dice

# ------------------------------------------------------------- sev loss ---
def sev_loss_fn(sev_cont, sev_ord, sev_cont_t, sev_ord_t):
    l1 = F.l1_loss(sev_cont, sev_cont_t)
    ce = F.cross_entropy(sev_ord, sev_ord_t)
    return l1 + ce

# ------------------------------------------------------------- PCGrad ---
class PCGrad:
    """Project conflicting task gradients (paper sec 2.5). Call .step(losses)."""

    def __init__(self, optimizer, scaler):
        self.optimizer = optimizer
        self.scaler = scaler

    @staticmethod
    def _flatten(params):
        gs = [p.grad for p in params if p.grad is not None]
        return torch.cat([g.reshape(-1) for g in gs]) if gs else None

    def step(self, task_losses):
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        total = sum(p.numel() for p in params)
        device = task_losses[0].device
        grads = []
        for lt in task_losses:
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(lt).backward(retain_graph=True)
            g = torch.zeros(total, device=device)
            off = 0
            for p in params:
                if p.grad is not None:
                    g[off:off + p.numel()] = p.grad.reshape(-1)
                off += p.numel()
            grads.append(g)
        n = len(grads)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                dot = grads[i] @ grads[j]
                if dot < 0:
                    grads[i] -= (dot / (grads[j] @ grads[j] + 1e-12)) * grads[j]
        self.optimizer.zero_grad(set_to_none=True)
        off = 0
        for p in params:
            numel = p.numel()
            if p.requires_grad:
                p.grad = (sum(g[off:off + numel] for g in grads) / n).view_as(p)
            off += numel
        # Gradient clipping: same protection as normal training path
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in params if p.requires_grad], 10.0)
        self.scaler.step(self.optimizer)

# ------------------------------------------------------------- total ---
def multitask_loss(outputs, batch, model_out_size, nc=16, reg_max=8, lambdas=None,
                   train_sev=True, active_tasks=None):
    """Weighted sum of task losses (paper eq. 11).

    active_tasks: subset of {"det","cls","seg"} present in `outputs`.
    Only computes losses for heads that actually ran (staged training skips
    inactive heads entirely to avoid frozen-head instability)."""
    lam = {"det": 1.0, "cls": 1.0, "seg": 1.0, "sev": 1.0}
    if lambdas:
        lam.update(lambdas)
    act = active_tasks or {"det", "cls", "seg"}
    parts = {}
    zero = None
    for name in ("det", "cls", "seg"):
        if name not in act:
            continue
        if name == "det":
            det_fn = DetectionLoss(nc=nc, reg_max=reg_max)
            targets = [bb[:n] for bb, n in zip(batch["det_boxes"],
                                               batch["n_boxes"].tolist())]
            parts.update(det_fn(outputs["det_cls"], outputs["det_box"],
                                outputs["anchors"], targets, model_out_size))
        elif name == "cls":
            valid = batch["cls"] >= 0
            if valid.any():
                parts["cls"] = F.cross_entropy(outputs["cls_logits"][valid],
                                               batch["cls"][valid])
            else:
                parts["cls"] = outputs["cls_logits"].sum() * 0.0
        elif name == "seg":
            parts["seg"] = seg_loss_fn(outputs["seg_logits"], batch["mask"])
    if train_sev and "sev_cont" in outputs:
        parts["sev"] = sev_loss_fn(outputs["sev_cont"], outputs["sev_ord"],
                                   batch["sev_cont"], batch["sev_ord"])
    total = None
    if "det_cls" in parts:   # NB: det keys are det_cls/det_box/det_dfl
        t = lam["det"] * (parts["det_cls"] + parts["det_box"] + parts["det_dfl"])
        total = t if total is None else total + t
    if "cls" in parts:
        t = lam["cls"] * parts["cls"]
        total = t if total is None else total + t
    if "seg" in parts:
        t = lam["seg"] * parts["seg"]
        total = t if total is None else total + t
    if "sev" in parts:
        total = total + lam["sev"] * parts["sev"]
    assert total is not None, "multitask_loss: no active task produced a loss"
    return total, parts
