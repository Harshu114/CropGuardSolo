# Data audit: `cropguard-corpus-claude` labels are degenerate

Date: 2026-10-11. Scope: the corpus both v6 and v7 were trained and scored on.

**Headline: the corpus's segmentation masks are copies of the detection boxes,
not leaf segmentations. Two of the six reported task metrics are therefore
vacuous on this corpus, and one (severity) cannot be fixed without new labels.**

## What was found

### 1. `masks/` is the filled detection boxes

For 20/20 sampled test images the mask PNG is **pixel-identical** to the union of
the image's YOLO boxes:

```
images checked: 20; leaf-mask == filled-boxes (>99% pixels): 20
  img0: pixel-match 100.0%  IoU(leaf,boxfill)=1.000  320x320  nboxes=1
  img2: pixel-match 100.0%  IoU(leaf,boxfill)=1.000  320x320  nboxes=5
```

Every mask is `320x320` — generated at model resolution, never the native image
size, which is what a real annotation would have been. A real GrabCut/SAM leaf
mask would also be irregular; these are exact rectangles.

Reproduce:

```bash
python3 - <<'EOF'
import os, glob, numpy as np, cv2
ROOT="<corpus>"; eq=0; n=0
for lp in sorted(glob.glob(f"{ROOT}/labels/test/*.txt"))[:20]:
    stem=os.path.basename(lp)[:-4]; mp=f"{ROOT}/masks/test/{stem}.png"
    if not os.path.exists(mp): continue
    raw=cv2.imread(mp,0); H,W=raw.shape
    bm=np.zeros((H,W),np.uint8)
    for line in open(lp):
        _,cx,cy,w,h=map(float,line.split()[:5])
        bm[int(max(0,(cy-h/2)*H)):int(min(H,(cy+h/2)*H)),
           int(max(0,(cx-w/2)*W)):int(min(W,(cx+w/2)*W))]=255
    eq += float(((raw==255)==(bm==255)).mean())>0.99; n+=1
print(f"{eq}/{n} masks identical to filled boxes")
EOF
```

### 2. Ground-truth severity is constant

`dataset.py` reads severity as `lesion_area / leaf_area`. Because the "leaf
mask" *is* the box union, `box_mask.mean()/leaf_frac ≡ 1.0` and the ratio is
clipped to 1.0 by `np.clip(ratio, 0.0, 1.0)`.

Measured on the full test split (1286 images):

```
GT grade hist : [3, 0, 0, 1283]      # 3 healthy, 1283 "severe"
GT sev_cont distinct values: 2       # {0.0: 3, 1.0: 1283}
```

## Consequences

| metric | status |
|---|---|
| `mAP@0.5`, `mAP@0.5:0.95` | **valid** — boxes/labels are real YOLO annotations |
| `cls_top1`, `cls_macro_f1`, `cls_ece` | **valid** — class labels are real |
| `seg_mIoU`, `seg_dice` | **inflated** — the target is axis-aligned rectangles, not lesions/leaves. Still usable for *relative* v6-vs-v7 comparison (both scored against the same rectangles), not as a leaf-segmentation claim |
| `sev_mae`, `sev_rmse`, `sev_r2` | **vacuous** — measures `|1 − prediction|` only |
| `sev_qwk` | **exactly 0 by construction** — when every ground-truth grade is identical, `Σw·O` always equals `Σw·E` in `quadratic_weighted_kappa`, so the score is 0 regardless of the model |

This is why `sev_mae 0.0428` "matches the paper target of 0.042" — the match is
coincidental, not evidence of severity skill.

## Why the PlantSeg derivation is the fix

The PlantSeg corpus is derived rather than borrowed, and its labels do not have
this failure mode (`data_pipeline/scripts/derive_dataset.py`):

- **boxes** — connected components of the real lesion mask polygons in
  `coco_annotations/*.json` (the annotations carry `segmentation` polygons);
- **leaf mask** — GrabCut on the actual image pixels (`--leaf-backend grabcut`),
  which cannot collapse onto the boxes;
- **severity** — `lesion area / leaf area` using those two *independent* sources,
  so it is non-degenerate by construction;
- **negative class** — the 33 non-tomato hosts in PlantSeg become `not_tomato`.

PlantSeg tomato covers 7 of the 11 CropGuard classes and 901 images
(623 train / 97 val / 181 test) — see `prepare_plantseg.py --self-test`.
The remaining 4 classes have no PlantSeg tomato source and can only come from
the borrowed corpus; that gap is a property of the data, not of the pipeline.

`derive_dataset.py --self-test` and `prepare_plantseg.py --self-test`
(annotation path) both pass locally.
