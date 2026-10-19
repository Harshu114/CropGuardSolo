"""Inference utilities extracted from training pipeline (no dataset dependency)."""
import torch
import torch.nn.functional as F
from losses import dfl_decode, box_xywh_to_xyxy


@torch.no_grad()
def nms(pred_xyxy, scores, iou_thr=0.55):
    """Simple NMS. pred_xyxy (N,4) pixels; scores (N,). Returns keep idx."""
    if pred_xyxy.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = pred_xyxy.unbind(-1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1 = torch.max(x1[i], x1[rest])
        iy1 = torch.max(y1[i], y1[rest])
        ix2 = torch.min(x2[i], x2[rest])
        iy2 = torch.min(y2[i], y2[rest])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long)