# CropGuard v7 — architecture + problem brief (for expert review)

> Companion to `docs/EXPERT_BRIEF.md`, which documents the **v5** architecture and
the dataset audit that motivated v7 (masks are filled leaf bounding boxes, not
lesions; class imbalance; duplicate leakage). This file covers what changed in
v7, what was measured, and what still needs expert input.

## 1. Task

Multi-task tomato-leaf disease model, one forward pass, four outputs:
**detection + classification + segmentation + severity**. Deployment target is a
phone (later a drone); tomato only, 11 PlantVillage classes.

## 2. Model, as actually built (not as previously described)

Total **2.49 M params** (v7) / 2.63 M (v5, shipped).

```
input 320x320
└─ FasterNet-T0 @224/320 (timm, features_only)
   v5: C3/C4/C5  @/8,/16,/32   (80/160/320ch)
   v7: adds C2   @/4  (40ch)  -> segmentation skip
└─ BiFPN-lite (3 levels, 1 top-down + 1 bottom-up round, width 96, fast
   normalized fusion w = relu(w)/sum(w))
   ├─ DecoupledDetHead (YOLOv8-style): 3 levels, cls + DFL box, 2100 anchors,
   │  cls bias init -log((1-0.01)/0.01), TAL assigner in losses.py
   ├─ ClassificationHead (v5) / LesionGatedClassificationHead (v7)
   │  v5 : SE(r=8) -> GAP on P3+P4+P5 -> fuse(288->96) -> MLP -> 11
   │  v7 : masked average pool over PREDICTED lesion pixels + plain GAP per
   │       level, concat(6x96) -> fuse -> MLP; mask is DETACHED (cls never
   │       trains seg); GT/predicted mask mix 0.8->0.2 across the stage
   ├─ Seg: LightHam decoder (v5) / BilinearDecoder (v7, 2-channel,
   │  /4 skip, Hardswish, no transposed convs)
   └─ Severity: learned gated fusion head (v5) / ANALYTIC (v7):
      sev = sum(sig(lesion_logit)) / (sum(sig(leaf_logit)) + 1e-3)
```

Extra v7 pieces: `AbstainHead` (temperature-scaled softmax + energy OOD),
`boxes_from_mask()` (boxes straight from the predicted lesion mask), cross-task
consistency term (only: *p(not healthy) must equal "lesion somewhere"*).

## 3. What we deliberately removed and why

- **Learned severity head (v5)** — its target was lesion/leaf *area*. Feeding it
  the area ratio as an input feature meant it was mostly learning to pass the
  ratio through. v7 computes the ratio from the masks directly.
- **Box<->mask consistency loss** — boxes are derived from the masks, so the two
  agree by construction and the term taught nothing (it also made the det head's
  supervision partly circular).
- **Hamburger/transposed-conv decoder** — no early skip, quant-hostile.
- **GELU in new modules** — Hardswish, for INT8 safety.

## 4. Training recipe

Staged: 1 det → 2 cls → 3 seg → 4 joint → 5 joint+severity. AdamW, cosine with
warmup, AMP, per-stage LR (1e-3 → 3e-4 → 1e-4), BN-safe rollback on non-finite
loss, class-balanced sampling, EMA, label smoothing 0.05.

Data (v7): single PlantSeg-tomato corpus with **all** labels derived —
class from filename (verified against `class_map.yaml`), boxes from lesion-mask
connected components, leaf masks from GrabCut/SAM, severity = lesion/leaf area
**only on close-up single-leaf images**, plus 33 non-tomato hosts as free
"not tomato" negatives for abstention. Perceptual-hash dedup, duplicate groups
never straddle splits.

## 5. Measured (not claimed)

- v7 params 2.49 M; smoke-tested through all 5 stages on both v5 and v7 paths.
- Grad check: cls loss contributes **exactly 0** gradient to the seg head.
- v5 checkpoint still loads strictly (292 tensors) — the shipped API is unaffected.
- ONNX (v7, opset 17, 320², CPU 1 thread): **mean 26.1 ms, p95 28.3 ms**.
  Cascade measured in PyTorch on the same box: full 21.3 ms vs cascade 38.9 ms —
  the cascade ran seg on nearly every synthetic sample, so that number is an
  upper bound for the worst case, not a win. Honest statement: the cascade only
  skips the seg decoder; backbone+BiFPN always run.
- v6 baseline to beat (same val protocol): `cls 0.285, seg 0.280, sev 0.065`.

## 6. Open problems we would like reviewed

1. **Severity validity.** Even restricted to close-ups, PlantSeg's web photos
   have ambiguous leaf extent. Is lesion/leaf defensible, and should grade
   thresholds be per-disease (published scales) rather than one global 5%/25%?
2. **Detector redundancy.** With boxes derivable from the predicted mask for
   free, is keeping the detector justified at all for a phone build, given the
   assigner + 2100 anchors + NMS cost? We kept it for the future drone case.
3. **Lesion gating** when the segmenter is weak early: gating on predicted masks
   at stage 2 measured worse than GAP, so gating is enabled only from stage 4
   (after the seg head exists). Is there a better warm-up schedule than
   GT 0.8 → 0.2?
4. **Gate input.** The classifier currently consumes the *lesion* channel only.
   Should leaf-gated features (lesion-normalised) also feed it?
5. **INT8.** Depthwise-heavy FasterNet + Hardswish: is post-training
   quantization likely to hold accuracy, or should we plan QAT?
6. **Abstention.** Energy-based OOD with temperature scaling fitted on val
   (target 5% abstain). Is per-class temperature needed, or is a single scalar
   fine for a farmer-facing "I don't know"?

## 7. Evaluation protocol

mAP@0.5 / @0.5:0.95, Top-1, macro-F1, **per-class F1**, **ECE (10-bin)**,
seg mIoU/Dice, severity MAE/RMSE/R²/QWK — all on field-style data, plus ONNX
latency. Held-out test split is duplicate-aware (no near-duplicate leakage).
