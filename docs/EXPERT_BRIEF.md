# CropGuard — architecture & open problems (brief for review)

Multi-task tomato-leaf disease diagnosis from a single forward pass:
**detection + classification + segmentation + severity**. Report against a paper
targeting mAP@0.5 0.892, cls_top1 0.987, seg_mIoU 0.863, sev_mae 0.042.
Efficiency-motivated: 2.63 M params, 320×320 RGB, CPU-capable.

Everything below was verified against the trained checkpoint's state_dict
(292 tensors, 2.63 M params) and the code that produced it.

---

## 1. Architecture

**Backbone** — `timm` `fasternet_t0`, `features_only`, `out_indices=(1,2,3)`:
C3 @/8 = 80 ch, C4 @/16 = 160 ch, C5 @/32 = 320 ch. (No C2 — nothing consumes it.)

**Neck** — `BiFPN-lite`, 3 levels (P3/P4/P5), width 96, **one** top-down + **one**
bottom-up round:
- per level: 1×1 conv → BN → GELU to width 96
- `BiFPNLayer` fusion: `w = relu(w_learnable); w = w / (w.sum + 1e-4)` →
  `w0·x1 + w1·x2` → depthwise 3×3 → pointwise 1×1 → BN → GELU. Weights are
  per-channel `(2, C)`, softmax-free (relu-normalized).
- separate lateral/top-down/bottom-up modules (2 each) for the 3 levels.

**Heads** (all width 96):

| head | structure | output |
|---|---|---|
| Detection | `DecoupledDetHead` per level: separate depthwise-separable cls and box branches; `cls_out` 1×1 → 11, `box_out` 1×1 → 32 (4×`reg_max`, reg_max=8, DFL) | `det_cls (B,2100,11)`, `det_box (B,2100,32)` |
| Classification | SE→GAP→GELU-MLP on **all** levels: 3× `SEBlock(r=8)` → GAP each → concat(288) → Linear(288→96)+GELU → MLP(96→96→11) | `cls_logits (B,11)` + embedding `z_cls (B,96)` |
| Segmentation | `LightHamLiteDecoder`: concat[P3, P4↑, P5↑] (288) → 1×1→96 → **Hamburger** (channel-attention: q,k → Linear to 24, softmax attn, Linear back, `x + γ·y`, γ init 0) → ConvTranspose ×3 (/4 → /2 → /1) → depthwise+pointwise+BN+GELU refine → 1×1 → 1 ch logits → bilinear to input size | `seg_logits (B,1,320,320)` |
| Severity | gated multimodal fusion: `z_s`=GAP(Conv1×1(seg)→96), `z_d`=GAP(P5), `z_c`=cls embedding, `r`=sigmoid(seg).mean → concat(289) → `gate=sigmoid(Linear(289→289))`, `h=MLP(gate·z)(289→128→64)` | `sev_cont`=sigmoid(Linear(64→1)), `sev_ord`=Linear(64→4) |

Anchors: **2100** = 40×40 + 20×20 + 10×10 at strides 8/16/32.

**Losses** — `DetectionLoss` = TaskAlignedAssigner (topk 10, α 0.5, β 6) + BCE-cls
with soft targets (one-hot × normalized alignment) + CIoU + 2-bin DFL.
Seg = BCE + Dice. Sev = L1(cont) + CE(4-way ordinal).
Weights: `det_cls 0.5, det_box 7.5, det_dfl 1.5, cls 1.0, seg 1.0, sev 1.0`.
Optional PCGrad from stage 4.

**Training** — 5 stages × 10 epochs, AdamW wd 1e-4, AMP FP16, cosine + warmup,
batch 32:
`s1` backbone+neck+det_head (det only) → `s2` cls_head only → `s3` seg_head only →
`s4` all of det+cls+seg → `s5` all + sev_head/seg_embed.
LRs `1e-3 / 3e-4 / 3e-4 / 3e-4 / 1e-4`. ~350–400 s/epoch, peak 6.4 GB.
Stages 2/3 train **heads only** (frozen backbone) — deliberate: an earlier version
that trained backbone+neck in s2/s3 destroyed the s1 detector (per-image best IoU
1→ fell 0.938 → 0.079, mAP@0.5 → 0.0 on the s2 checkpoint).

## 2. Data

32,897 images / 11 classes (29,611 train / 2,000 val / 1,286 test), all native
**320×320**, merged from 3 Roboflow Universe projects (CC-BY-4.0).
Per image: YOLO box label (`.txt`) + a `.png` mask.

Class order: healthy, bacterial_spot, early_blight, late_blight, leaf_miner,
leaf_mold, mosaic_virus, septoria_leaf_spot, spider_mites, target_spot,
yellow_leaf_curl_virus.

**Label pathologies (measured, this is the crux):**
- Masks are **binary {0,255} filled leaf bounding boxes**, not lesions. The
  `128 = lesion` value that the loader supports **never occurs** (in the 200 masks
  I sampled; the repo's own note says the same for all 32,897).
- Box geometry (`box_geometry_audit.json`, train): 71,996 boxes, mean 2.44/image,
  max 100. **5,845 boxes ≥90% of the frame, 18,772 ≥50%.**
  `bacterial_spot` median box area = **0.889** of the frame (47.6% ≥90%);
  `target_spot` median 0.598. Others ~0.02–0.22.
- Class imbalance: `healthy` 12,220 boxes vs **`target_spot` 252** (13 test images).
- Duplicates: 1,139 dhash groups / 2,396 train images (8.09%), 15 groups with
  **conflicting** labels. Leakage: 179 groups span splits (val 7.45%).
- Severity target = `box_area / leaf_area`, binned 0 healthy / <0.10 mild /
  <0.30 moderate / ≥0.30 severe.

## 3. Measured results

| metric | reported (1,286 test) | leak-free re-baseline (1,240) | paper target |
|---|---|---|---|
| mAP@0.5 | 0.511 | 0.524 | 0.892 |
| mAP@0.5:0.95 | 0.431 | 0.442 | 0.624 |
| cls_top1 | 0.853 | 0.850 | 0.987 |
| cls_macro_f1 | 0.843 | 0.839 | — |
| seg_mIoU | 0.880 | 0.883 | 0.863 |
| seg_dice | 0.936 | 0.938 | 0.912 |
| **sev_r2** | **−1.41** | **−1.33** | — |
| sev_mae | 0.068 | 0.068 | 0.042 |
| sev_qwk | 0.800 | 0.800 | — |

Serving adds horizontal-flip TTA: val mAP@0.5 0.4972 → 0.5101.

---

## 4. Problems — what I want expert input on

**P1. The severity head is degenerate (r² −1.41, worse than a constant predictor:
a constant scores mae 0.009 vs the model's 0.068).**
Root cause is the label, not the head: `ratio = box_area / leaf_area`, and the
masks are near-full-frame leaf regions, so `ratio ≥ 0.30` for 1991/2000 val images
→ the target is ~constant "severe" and the head learns nothing.
- *Question:* how do multi-task/crop papers define a defensible **severity** target
  without pixel-level lesion masks? Is area-ratio-from-boxes ever acceptable? Would
  an ordinal formulation (or dropping the head) be more honest?

**P2. The segmentation metric does not measure what it claims.**
Because masks are leaf boxes, `seg_mIoU 0.880 / Dice 0.936` is **leaf-region** IoU,
not lesion segmentation — and the decoder is effectively learning "where is the
leaf". It also feeds severity (`z_s`, `r`), so P1 and P2 are coupled.
- *Question:* is it defensible to report this as segmentation with a clearly
  renamed metric, or must it be replaced with a real lesion-mask dataset before
  the number means anything?

**P3. Detection is furthest from target (0.511 vs 0.892) — label problem or model?**
The GT boxes are frequently whole-leaf (median 0.889 for one class, 5,845 ≥90% of
frame). A detector trained on whole-object boxes cannot produce lesion-localised
output, and mAP against huge boxes is a very different regime from standard
small-object detection.
- *Question:* how should anchor design / assignment change when GT boxes routinely
  span the whole image (currently a standard 3-level 8/16/32 tiling, 2100 anchors)?
  Is mAP the right headline metric for full-object boxes, and how should results be
  reported so the gap to the paper's 0.892 is interpreted fairly?

**P4. Classification 0.853 vs 0.987 target.**
Errors concentrate among the visually similar brown-lesion classes
(early blight / late blight / septoria). The head already pools P3+P4+P5 with SE to
recover texture.
- *Question:* for texture-dominated fine-grained classes, is multi-scale SE→GAP→MLP
  the right head, or is bilinear/second-order pooling, a small transformer, or
  training on detected-box crops rather than the full frame materially better?
  (Note the corpus is single-leaf close-ups, so the whole frame ≈ the leaf.)

**P5. Multi-task interference and the staged schedule.**
Severity consumes seg logits and the cls embedding, so seg errors propagate.
Stages 2/3 freeze the backbone entirely (head-only) to protect the stage-1
detector — a fix for a real regression, but it means cls/seg never co-adapt with
features.
- *Question:* is head-only staged training the right call, or should
  backbone LR be scaled down (discriminative fine-tuning) instead of frozen? Does
  PCGrad help or hurt here? Is the severity gate (sigmoid on a 289-dim line) a
  sound fusion, or should reliability be estimated explicitly?

**P6. Evaluation/leakage.**
I removed 1,425 duplicates and confined duplicate groups to one split; results were
unchanged-or-better, so leakage was not inflating them. Remaining caveat: dhash
only catches exact perceptual duplicates, so near-duplicate leakage is a lower
bound.
- *Question:* what is the standard practice for reporting under
  near-duplicate contamination in merged web-scraped corpora?

**P7. Practical constraints.** Target hardware is a 4 GB laptop GPU (local) and
Kaggle T4 ×2 (T4 required; the pipeline rejects compute capability < 7.0 for a
reproducible P100 bug). Full run ≈ 5–6 h.

---

### The one-line summary

Detection and classification are credible (0.52 mAP@0.5, 0.85 top-1). **Severity is
non-functional and segmentation is measuring the wrong thing — both because the
corpus annotates whole leaves rather than lesions**, so no architecture change or
extra epoch fixes them; they need different labels.
