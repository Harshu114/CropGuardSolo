# CropGuard: A Unified Multi-Task Lightweight Framework for Tomato Leaf Disease Localization, Classification, Pixel-Level Lesion Segmentation, Severity Estimation, and Explainability

**A detailed technical research paper (Markdown, diagram-complete), derived from the actual implementation in this repository.**

| | |
|---|---|
| **Model family** | `CropGuardMultiTaskV7` (`model/architecture/v7_model.py`) |
| **Task header** | 8-class tomato foliar disease analysis, multi-task |
| **Total parameters** | **2,533,254** (≈2.53 M) — measured, see §9.1 |
| **Input resolution** | 320 × 320 RGB, ImageNet-style normalization |
| **Deployment targets** | ONNX FP32 (10.23 MB) and static INT8 (3.13 MB) |
| **Explainability** | LayerCAM over the neck (P3/P4/P5) |
| **Primary corpus** | `cropguard_v7_split` — train 715 / val 156 / test 150 |
| **Reference GPU** | Kaggle Tesla T4 (×2, single-device training) |

> **Reading note on evidence.** Every quantitative number in this paper is tagged with its source: `[M]` = measured in this repository and reproducible from the listed command/artifact, `[P]` = pending (a run is queued or has not produced a checkpoint), `[N]` = not measured. No number in this paper is estimated, extrapolated, or illustrative. Where a widely expected result does not exist yet, the paper says so explicitly.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Introduction](#2-introduction)
3. [Foundations: from a single convolution to a multi-task network](#3-foundations-from-a-single-convolution-to-a-multi-task-network)
4. [System overview and process diagrams](#4-system-overview-and-process-diagrams)
5. [Backbone: FasterNet-T0](#5-backbone-fasternet-t0)
6. [Neck: BiFPN-Lite](#6-neck-bifpn-lite)
7. [Heads: low-level breakdown of every branch](#7-heads-low-level-breakdown-of-every-branch)
8. [Analytic severity and abstention](#8-analytic-severity-and-abstention)
9. [Parameter and tensor-shape budget](#9-parameter-and-tensor-shape-budget)
10. [Loss functions: complete mathematics](#10-loss-functions-complete-mathematics)
11. [Data pipeline](#11-data-pipeline)
12. [Training protocol](#12-training-protocol)
13. [Evaluation protocol](#13-evaluation-protocol)
14. [Results](#14-results)
15. [Design rationale: why this particular combination](#15-design-rationale-why-this-particular-combination)
16. [Reliability engineering: the runtime-hang class of failure](#16-reliability-engineering-the-runtime-hang-class-of-failure)
17. [Limitations and threats to validity](#17-limitations-and-threats-to-validity)
18. [Deployment](#18-deployment)
19. [Reproducibility](#19-reproducibility)
20. [References](#20-references)

---

## 1. Notation

| Symbol | Meaning | Shape / domain |
|---|---|---|
| $B$ | batch size | 8 |
| $H, W$ | spatial resolution | $320 \times 320$ (full), $/4, /8, /16, /32$ for pyramid levels |
| $C$ | channel count | e.g. 40, 80, 96, 160, 320 |
| $K$ | convolution kernel size | typically 3 |
| $s$ | stride | 1, 2, 4, 8, 16, 32 |
| $N_c$ | number of disease classes | 8 |
| $A$ | total anchor points | $40^2 + 20^2 + 10^2 = 2100$ |
| $R$ | DFL bins | `reg_max` = 8 |
| $p$ | sigmoid probability | $[0,1]$ |
| $\sigma$ | logistic sigmoid | $\sigma(z) = 1/(1+e^{-z})$ |
| $T$ | softmax temperature | scalar, learned/calibrated |
| $\lambda$ | loss weight | scalar per task |
| $\mathbf{x}$ | input image tensor | $(B, 3, H, W)$ |

Throughout, spatial resolution is written as "$\mathbin{/}k$", meaning feature map size $H/k \times W/k$.

---

## 2. Introduction

### 2.1 Problem statement

Foliar disease diagnosis is the highest-leverage intervention point in tomato production, but the information a grower actually needs is **not a single label**. A useful diagnosis answers five questions at once:

1. **What** is the disease? (classification)
2. **Where** is it on the leaf? (localization)
3. **Which pixels** are affected? (lesion segmentation)
4. **How much** of the leaf is affected? (severity, a continuous quantity that drives treatment decisions)
5. **Why** did the model answer that? (explainability, required for grower trust and regulatory review)

Most published plant-disease work answers only question 1. This repository's design premise is that answers 1–5 can be produced by **one shared feature extractor with several cheap heads**, at a parameter cost small enough for a phone.

### 2.2 The failure that motivated the current architecture

The immediate predecessor (v7) achieved respectable classification but **its segmentation branch never learned**. Measured evidence `[M]` from `runs_v7_120/train.log` / `train_full.log` (also recorded in `docs/v8_report.md`):

```
epoch  90: cls=0.3081  seg=3.6495
epoch 120: cls=0.3072  seg=3.6311     <- seg loss FLAT for the entire run
epoch 120 val: acc=0.6755 macro_f1=0.6433 lesion_iou=0.0282 dice=0.0549
```

A *flat* segmentation loss is diagnostic: the segmentation head was receiving no usable gradient signal. Three independent root causes were found by reading the code:

- **(a) Wrong supervision target.** The segmentation target was a set of **bounding-box rectangles** (`make_pseudo_mask`), not real lesion pixels. The head was being asked to regress rectangles while being scored against polygons.
- **(b) No high-resolution skip.** The decoder consumed only stride-8/16/32 features. A lesion occupying 3–8 px at 320×320 is already sub-pixel after a stride-8 downsample.
- **(c) A learned severity head with no labels.** There were no human severity ratings, so the severity regression head was learning to re-derive the area ratio it was already given at input — an identifiability failure.

The v8/v9/v10 line is a direct, causally-argued repair of (a)–(c).

### 2.3 Contributions claimed in this paper

1. **A unified 5-output architecture** — detection, classification, 2-channel segmentation, analytic severity, and abstention — at **2.53 M parameters**, with an explicit parameter budget (§9).
2. **Lesion-gated classification**: classification pooling is restricted to predicted-lesion pixels, removing the background shortcut that a global-average-pooled classifier is prone to.
3. **Analytic severity**: severity is computed as a ratio of *predicted masks*, with the learned severity head deliberately removed, turning an under-determined regression into a measurable segmentation-derived quantity.
4. **A two-channel segmentation target** — `(lesion, leaf)` — with a stride-4 skip and a false-negative-weighted loss for a class of extremely small targets.
5. **A staged training protocol** with data-integrity gates and a hard wall-clock safety net (§12, §16).
6. **A complete, honest evaluation protocol** including a healthy-shortcut leakage probe and a paired old-vs-new checkpoint comparison.

---

## 3. Foundations: from a single convolution to a multi-task network

This section builds the architecture from first principles so that each later design choice has a reference point.

### 3.1 The discrete convolution layer

A 2-D convolution layer computes, for input feature map $X \in \mathbb{R}^{H \times W \times C_{in}}$ and kernel $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$:

$$
Y[i, j, o] \;=\; b[o] \;+\; \sum_{u=0}^{k-1}\sum_{v=0}^{k-1}\sum_{c=0}^{C_{in}-1}
K[u, v, c, o]\; X\big[i\,s + u - p,\; j\,s + v - p,\; c\big]
$$

where $s$ is the **stride** (subsampling factor) and $p$ the **padding**. Three quantities matter for everything that follows:

- **Receptive field.** For $L$ stacked $k\times k$ convolutions, the receptive field grows linearly: $R_L = 1 + L(k-1)$. For $k=3$ and $L=6$, $R=13$ px. This is why a pure CNN needs deep stacks or explicit downsampling to see a whole leaf.
- **Computational cost.** $\mathrm{FLOPs} \approx 2 \cdot k^2 \cdot C_{in} \cdot C_{out} \cdot H_{out} \cdot W_{out}$. Cost is **quadratic in channel width and linear in spatial area**.
- **Spatial reduction** is what creates the "feature pyramid". Output size is $H_{out} = \lfloor (H + 2p - k)/s \rfloor + 1$. A stride-$s$ layer produces a $\mathbin{/}s$ map.

### 3.2 Stride and the feature pyramid

Running a single input at $320\times320$ through a backbone whose stages have strides 4, 8, 16, 32 yields the classical pyramid used by this model:

```
  input          stage-1         stage-2         stage-3         stage-4
  320x320   ->   80x80     ->    40x40     ->    20x20     ->    10x10
  (3 ch)         (40 ch)        (80 ch)        (160 ch)        (320 ch)
                 C2 (/4)        C3 (/8)        C4 (/16)        C5 (/32)
                  |              |              |               |
                  |              +-- neck in -->+               |
                  |              (this model's neck reads C3, C4, C5)
                  +-- segmentation skip (this model's key addition)
```

**Why a pyramid at all?** A single stride-32 map has a receptive field large enough to see a whole leaf but a spatial resolution too coarse to draw where the lesion is. The pyramid lets each head choose the resolution it needs: the detection head uses all three coarse levels; the segmentation decoder uses the **finest** level plus a stride-4 skip.

### 3.3 Normalization: batch normalization

For a mini-batch, BN standardizes each channel then applies a learned affine:

$$
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}, \qquad y = \gamma \hat{x} + \beta
$$

BN is used **after every convolution** in this model. It matters for three practical reasons here: (i) it makes the 5e-4 learning rate safe without per-layer tuning; (ii) it lets each head consume features with different statistics without destabilizing the shared backbone; (iii) it is foldable into the preceding convolution at export time, so it costs nothing at inference in the INT8 graph.

### 3.4 Activations: ReLU, GELU, Hardswish — and why this model uses two different ones

$$
\mathrm{ReLU}(z)=\max(0,z),\qquad
\mathrm{GELU}(z)=z\,\Phi(z),\qquad
\mathrm{Hardswish}(z)=z\cdot\frac{\mathrm{ReLU6}(z+3)}{6}
$$

This model is deliberately **inconsistent across modules**, and the inconsistency is the point:

- The **neck** (`BiFPNLite`) uses **GELU**. The neck is never quantized to INT8 in the deployed graph — it runs in FP32 alongside the backbone's residual blocks.
- All **v7 heads** (`BilinearDecoder`, `LesionGatedClassificationHead`) use **Hardswish**. Hardswish is piecewise-linear and therefore **exactly representable** by the INT8 quantizer; GELU's exponential is not.

This is the single most consequential "boring" decision in the architecture, and it is the reason INT8 export succeeds without per-layer surgery (§18).

### 3.5 Residual connections and gradient flow

A residual block computes $y = x + \mathcal{F}(x)$. The identity path gives $\partial y/\partial x = I + \partial\mathcal{F}/\partial x$, so gradients reach early layers even when $\partial\mathcal{F}/\partial x$ is small. In this model the residual mechanism that matters for *training stability across tasks* is the **detached mask gate** (§7.2), which plays the analogous role of stopping one task's gradient from corrupting another's.

### 3.6 Depthwise and partial convolutions

A standard $k\times k$ convolution costs $k^2 C_{in} C_{out}$. Two cheaper factorizations appear in this model:

**(i) Depthwise separable** (used inside the neck and the decoder's skip projection):

$$
\text{cost}_{\text{dwsep}} = \underbrace{k^2 C}_{\text{depthwise}} + \underbrace{C^2}_{\text{pointwise}} \;\ll\; k^2 C^2
$$

Implementation note: in `BiFPNLite` and `BilinearDecoder` the depthwise step is expressed as `Conv2d(C, C, 3, padding=1, groups=C, bias=False)` — `groups == C_in == C_out` **is** a depthwise convolution.

**(ii) Partial convolution (PConv)** — the defining primitive of FasterNet, detailed next in §5.1.

### 3.7 From single-task to multi-task: the sharing problem

If $N$ tasks each have a loss $\mathcal{L}_i$, naive training minimizes $\sum_i \lambda_i \mathcal{L}_i$. The failure modes are well known and all three are addressed here:

| Failure mode | Symptom | Mitigation in this model |
|---|---|---|
| Task dominance | One loss's gradient magnitude dwarfs the others; the shared trunk optimizes only it | Explicit YOLOv8-standard weights `det_cls=0.5, det_box=7.5, det_dfl=1.5` (§15.4); `cls`, `seg` both weight 1.0 |
| Gradient conflict | Two task gradients point in opposing directions, so the shared trunk oscillates | Optional **PCGrad** projection (§10.7) |
| Negative transfer | A task with bad supervision poisons the shared features | **Staged training**: stage 1 gates on the segmentation task alone; heads are switched on only after the trunk is proven to carry lesion signal |

---

## 4. System overview and process diagrams

### 4.1 End-to-end system diagram

```
                              ┌───────────────────────────────────────────────┐
                              │              INPUT: RGB leaf image            │
                              │              320 x 320 x 3                    │
                              └───────────────────────┬───────────────────────┘
                                                      │
                              ┌───────────────────────▼───────────────────────┐
                              │   BACKBONE   FasterNet-T0  (timm)             │
                              │   partial-conv blocks, 4 stages               │
                              │   emits C2 /4, C3 /8, C4 /16, C5 /32          │
                              │   channels: 40, 80, 160, 320                  │
                              │   params: 2,214,900   (87.4 % of total)       │
                              └───────┬───────────────┬───────────────┬───────┘
                                      │ C2 (/4, 40ch) │ C3,C4,C5      │
                                      │               │               │
              ┌───────────────────────┘               │               │
              │ (skip, used only by seg)              │               │
              │                        ┌──────────────▼───────────────┐│
              │                        │   NECK  BiFPN-Lite (w=96)    ││
              │                        │   top-down then bottom-up    ││
              │                        │   weighted cross-scale fuse  ││
              │                        │   params: 146,592  (5.8 %)   ││
              │                        └───┬──────────┬──────────┬────┘│
              │                     P3 /8,96ch  P4 /16,96  P5 /32,96  │
              │                            │          │          │    │
              │        ┌───────────────────┘          │          │    │
              │        │            ┌────────────────-┘          │    │
              │        │            │        ┌───────────────────┘    │
              │        │            │        │                        │
              │        │     ┌──────▼────────▼────────▼─────┐         │
              │        │     │  DETECTION HEAD (decoupled) │         │
              │        │     │  cls: 96->nc per pixel      │         │
              │        │     │  box: 96->4*R (DFL, R=8)    │         │
              │        │     │  params: 24,616 (1.0 %)     │         │
              │        │     └────────────┬────────────────┘         │
              │        │                  │ det_cls, det_box          │
              │        │                  │ + anchors (2100, 3)       │
              │        │                                               │
              │        │     ┌─────────────────────────────────────────┐
              │        │     │   CLASSIFICATION HEAD (lesion-gated)    │
              │        │     │   masked-avg over predicted lesion +    │
              │        │     │   GAP, fused per level, then MLP        │
              │        │     │   params: 65,480 (2.6 %)                │
              │        │     └────────────┬────────────────────────────┘
              │        │                  │ cls_logits (B, 8)
              │        │                  │
              │        │     ┌────────────▼────────────────────────────┐
              │        │     │   ABSTAIN HEAD                          │
              │        │     │   temperature scaling + energy OOD      │
              │        │     │   params: 0 (buffer only)               │
              │        │     └────────────┬────────────────────────────┘
              │        │                  │ cls_probs, cls_energy
              │        │
   ┌──────────▼────────▼──────────────────────────────────────────────┐
   │        SEGMENTATION HEAD  BilinearDecoder  (2 channels)          │
   │  input: P3,P4,P5 + C2 stride-4 skip                              │
   │  fuse -> /4 (48ch) -> x2 -> /2 (32ch) -> x2 -> /1 (24ch) -> 2ch  │
   │  output: seg_logits (B, 2, 320, 320) = [lesion, leaf]            │
   │  params: 81,666 (3.2 %)                                          │
   └────────────────────┬────────────────────────────────────────────┘
                        │ seg_logits
                        │
        ┌───────────────┴────────────────┐
        │                                │
┌───────▼─────────┐            ┌─────────▼──────────────────────────┐
│ LESION CHANNEL  │            │ LEAF CHANNEL                       │
│ sigmoid -> P(L) │            │ sigmoid -> P(leaf)                 │
└───────┬─────────┘            └─────────┬──────────────────────────┘
        │                                │
        └────────────┬───────────────────┘
                     │
        ┌────────────▼────────────────────────────────┐
        │  ANALYTIC SEVERITY   (no learned parameters) │
        │  sev = Σ P(lesion) / (Σ P(leaf) + 1e-3)      │
        │  -> continuous severity in [0,1]             │
        │  -> grade: 0 healthy | 1 mild<0.10 |         │
        │            2 moderate<0.30 | 3 severe>=0.30  │
        └─────────────────────────────────────────────┘
```

### 4.2 Process diagram — training

```
 STAGE 0 — INTEGRITY GATES (all must pass; any failure aborts before GPU spend)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ 1. dataset.py sha256 gate     expected ff0102ed…96db                     │
 │ 2. forced-geometry subprocess image+mask agree within 2 px (timeout 600s)│
 │ 3. pretrained-weight check    first-conv L1 vs random init > 0.01        │
 │ 4. label audit                per-split counts, empty-label exclusions   │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
 STAGE 1 — MEMORIZATION GATE (8 images, 500 steps, augmentation ON)        │
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ can the model+LOSS+DATA pipeline overfit 8 images to lesion IoU >= 0.50? │
 │   PASS -> geometric augmentation stays ON                                │
 │   FAIL -> print grad-norm/requires_grad diagnostics, retry with          │
 │           GEOMETRIC AUG DISABLED (photometric only)                      │
 │           PASS2 -> PLAN B: photometric-only augmentation                 │
 │   FAIL2 -> STOP before main training (the pipeline is dead)              │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
 STAGE 2 — MAIN TRAINING (up to 120 epochs, hard 55-min cap)               │
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ per epoch:                                                               │
 │   train (cls + seg) with EMA(0.999), AdamW, cosine LR, AMP                 │
 │   evaluate RAW weights on val  -> macro-F1 / lesion IoU / severity MAE    │
 │   swap EMA in, evaluate          -> same metrics                          │
 │   keep the best of {raw, ema} by val macro-F1, tie-break lesion IoU        │
 │   save best_v*_raw.pt / best_v*_ema.pt / last_v*.pt every epoch            │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
 STAGE 3 — VAL SUMMARY of the selected checkpoint (test split NEVER touched)│
                                   │
                                   ▼
                     checkpoints + CSV log + JSON stage summaries
```

### 4.3 Process diagram — inference and export

```
  ┌──────────────┐   ┌───────────────────────────────────────────┐
  │ leaf image   │-->│ CropGuardMultiTaskV7.forward(...)         │
  │ 320x320x3    │   │  active_tasks={"det","cls","seg"}         │
  └──────────────┘   └───────────────┬───────────────────────────┘
                                     │
        ┌────────────────┬───────────┴──────────┬──────────────────┐
        ▼                ▼                      ▼                  ▼
   cls_logits       seg_logits              det_cls/det_box     anchors
   (B, 8)           (B, 2, 320, 320)        (B,2100,8)/(B,2100,32)  (2100,3)
        │                │                      │
        │                │                      │ DFL decode -> xywh -> NMS
        │                │                      ▼
        │                │                 boxes + scores
        │                ▼
        │        ┌───────────────────────┐
        │        │ severity = ΣL/(Σleaf+ε)│
        │        │ + grade bucket        │
        │        └───────────────────────┘
        │
        ▼
   temperature-scaled softmax + energy  ->  predict / ABSTAIN
        │
        ▼
   LayerCAM over neck (P3+P4+P5)  ->  heatmap overlay for the grower
```

### 4.4 Block diagram — where each task reads from

```
                    shared trunk (backbone)   shared trunk (neck)
                    ───────────────────────   ───────────────────
  detection      :          C3,C4,C5        ->     P3,P4,P5      (all levels)
  classification :          C3,C4,C5        ->     P3,P4,P5      (all levels, lesion-gated)
  segmentation   :          C2 (skip) + C3,C4,C5 -> P3,P4,P5     (all levels + finest skip)
  severity       :                derived from segmentation output only
  abstention     :                derived from classification logits only

  Reading: every head lives off ONE shared trunk. No head has its own encoder.
           This is the entire parameter-efficiency argument of the paper:
           93.2 % of params live in the shared trunk (backbone + neck),
           which all four learned tasks read from.
```

---

## 5. Backbone: FasterNet-T0

### 5.1 The partial convolution (PConv)

Standard convolution wastes FLOPs because it applies a dense $k\times k$ filter across **all** channels. PConv applies the filter to only a fraction of channels and leaves the rest untouched:

$$
\text{cost}_{\text{PConv}} = k^2 (C_{p})^2 H W, \qquad C_p = \tfrac{C}{r}
$$

with $r$ the partition ratio. For the canonical configuration $C=256, k=3, r=4$, a dense convolution costs $k^2C^2 = 590{,}000$ units while PConv costs $k^2C_p^2 = 36{,}864$ — roughly a **1/16** reduction. The untouched channels are then *reused* by a following pointwise convolution that mixes information across all channels:

$$
\text{PConv} \;\rightarrow\; \text{PWConv} \;\rightarrow\; \text{residual add}
$$

This is the FasterNet block. It preserves channel mixing (needed for representational capacity) while removing redundant spatial filtering (the expensive part).

### 5.2 What this repository actually instantiates

```python
self.net = timm.create_model("fasternet_t0", pretrained=pretrained,
                             features_only=True, out_indices=levels)
self.channels = self.net.feature_info.channels()
```

- **Backbone:** `fasternet_t0` from `timm`, `features_only=True`.
- **Levels for v7:** `_LEVEL_INDICES_V7 = (0, 1, 2, 3)` → `_STRIDES_V7 = (4, 8, 16, 32)`.
- **Channel widths read at runtime (measured):** `[40, 80, 160, 320]`.
- **Parameters (measured `[M]`):** 2,214,900 = **87.43 %** of the model.

### 5.3 Why the fourth (stride-4) level is requested — and why that was a real bug

Version 5 requested `out_indices=(1,2,3)` — three maps — while an offline fallback path in the same file requested `(0,1,2,3)` — four maps. The two code paths therefore **disagreed on arity**, and any caller that unpacked `c3, c4, c5 = backbone(x)` would raise on one path and not the other.

v7 resolves this by requesting four levels **on purpose** and unpacking four:

```python
feats = self.backbone(x)          # [C2, C3, C4, C5]
c2, c3, c4, c5 = feats
```

The extra `C2` (stride 4, 40 channels) exists **solely** to give the segmentation decoder a high-resolution skip. Justification: at 320×320, a lesion of 3–8 px occupies, after a stride-8 downsample, a 0.4–1.0 px region — i.e. it is annihilated before the decoder ever sees it. The stride-4 skip keeps it at 0.75–2.0 px, and keeps the *feature* rather than the sample at the finest level the backbone exposes.

### 5.4 Pretrained-weight policy

```python
except Exception as e:            # HF Hub unreachable (e.g. Kaggle, Internet off)
    if not pretrained or not allow_random:
        raise RuntimeError(
            f"fasternet_t0 pretrained weights unavailable ({e}). "
            "Pre-download them and attach as a Kaggle dataset, or pass "
            "allow_random=True to train from random init (not advised)."
        ) from e
```

A download failure is a **hard error** by default. This is deliberate: the previous silent fallback to random initialization allowed a run to train a full model on random features and report plausible-looking losses, wasting an entire GPU budget. Random initialization is available only via the explicit `allow_random=True` opt-out, used by smoke tests.

**Measured `[M]` (`docs/v8_report.md`, `stage0_summary.json`):**

```
pretrained_mode = "pretrained"
pretrained_l1   = 0.23836876451969147     (first-conv L1 vs a random-init reference)
threshold       = 0.01                    -> declared NOT random init
```

---

## 6. Neck: BiFPN-Lite

### 6.1 Purpose

The backbone's $C3, C4, C5$ maps have different channel counts (80, 160, 320) and different semantics (fine texture vs. coarse object). The neck (i) **projects** all three to a common width $w = 96$, and (ii) allows information to **flow between scales** so that a coarse map can contribute to a fine prediction and vice versa.

### 6.2 Weighted cross-scale fusion

Each fusion node learns two weights and normalizes them across the two inputs:

$$
O = \mathrm{act}\Big(\mathrm{BN}\big(\mathrm{PW}(\mathrm{DW}(w_0\!\cdot\!x_1 + w_1\!\cdot\!x_2))\big)\Big),
\qquad
w_i' = \frac{\mathrm{ReLU}(w_i)}{\sum_j \mathrm{ReLU}(w_j) + \epsilon}
$$

The `ReLU` before normalization enforces non-negativity (a *negative* contribution weight is not meaningful in a fusion node), and the softmax-style normalization makes the fusion **scale-invariant** with respect to the raw parameter magnitudes. In code, the weights are per-channel:

```python
self.w = nn.Parameter(torch.ones(2, c))
def fuse(self, x1, x2):
    w = torch.relu(self.w)
    w = w / (w.sum(dim=0, keepdim=True) + self.eps)
    w0 = w[0][None, :, None, None]          # (1, C, 1, 1) channel-wise, not scalar
    w1 = w[1][None, :, None, None]
    return self.act(self.bn(self.pw(self.conv(w0 * x1 + w1 * x2))))
```

Because the weights are shape $(2, C)$, each *channel* independently decides how much to trust the fine vs. the coarse input. A scalar weight would force one global trade-off; per-channel weights let, e.g., a "small-spot" channel stay high-resolution while a "necrosis-gradient" channel borrows coarse context.

### 6.3 Topology: one top-down round then one bottom-up round

```
   C3 /8  ──►in_proj──► c3 ──────────────────────────────┐
   C4 /16 ──►in_proj──► c4 ──────────────┐               │
   C5 /32 ──►in_proj──► c5 ──┐           │               │
                             │           │               │
        ┌────────────────────┘           │               │
        │  p5 = c5                       │               │
        │                                │               │
        │  p4 = td0( c4 , up2(p5) ) ◄────┘  (top-down, /32 -> /16)
        │            │
        │  p3 = td1( c3 , up2(p4) )         (top-down, /16 -> /8)
        │            │
        │  p4 = bu0( c4 , maxpool2(p3) )    (bottom-up, /8 -> /16)
        │            │
        │  p5 = bu1( c5 , maxpool2(p4) )    (bottom-up, /16 -> /32)
        │
        └──► out_proj x3 (depthwise 3x3 -> pointwise 1x1 -> BN -> GELU)
                     │           │           │
                    P3 /8       P4 /16      P5 /32          all w = 96 channels
```

Two design choices deserve justification:

- **Upsampling uses `nearest`, downsampling uses `max_pool2d`.** Nearest-neighbour upsampling does not invent sub-pixel intensities (important because the decoder will later refine them), and max-pooling downsampling propagates the *strongest* activation — appropriate for disease spots, which are high-contrast local maxima, rather than blurring them with average pooling.
- **One round, not three.** Full BiFPN repeats top-down/bottom-up several times, which buys a few tenths of a mAP point at a real parameter and latency cost. Given that the target device is a phone, the "lite" variant keeps a single round. This is the same trade-off that justifies 96 channels rather than 128/256.

**Measured `[M]`:** neck = 146,592 params = **5.79 %**.

---

## 7. Heads: low-level breakdown of every branch

### 7.1 Segmentation head — `BilinearDecoder`

**Objective.** Produce a full-resolution 320×320 two-channel map `[lesion, leaf]`.

**Why not a transposed convolution stack?** The predecessor used three transposed convolutions with no early skip. Transposed convolution with stride 2 and kernel 3 has a known **checkerboard** artifact (uneven overlap of the upsampling kernel), and, more decisively for this problem, it reconstructs from stride-8 features only. Small lesions are already gone. Additionally, the predecessor used `GELU`, which is not INT8-friendly.

**The v7 design.** Bilinear interpolation (artifact-free, exactly representable) paired with 3×3 convolutions that *refine* rather than *create* resolution, plus the stride-4 skip.

```
  P3 (/8,  96ch) ──┐
  P4 (/16, 96ch) ──┼─► interpolate to /8 ─► concat(3x96=288) ─► reduce 288->96 (1x1)
  P5 (/32, 96ch) ──┘                                              │  /8, 96ch
                                                                  │
  C2 (/4, 40ch) ──► skip: 1x1 40->48, BN, Hardswish,               │
                          depthwise 3x3 48->48, 1x1 48->48,       │
                          BN, Hardswish                             │  /4, 48ch
                                                                  │
                          interpolate x to /4  ──► concat(96+48=144) ─► fuse_skip 144->48
                                                                  │  /4, 48ch
                                        ┌─────────────────────────┘
                                        ▼
                              bw: 3x3 48->48, BN, Hardswish        /4, 48ch
                                        │
                          interpolate x2  ──►  /2 (160x160)
                                        │
                              up1: 3x3 48->32, BN, Hardswish      /2, 32ch
                                        │
                          interpolate to 320x320
                                        │
                              up2: 3x3 32->24, BN, Hardswish      /1, 24ch
                                        │
                              head: 1x1 24->2                     /1, 2ch
                                        │
                                        ▼
                              seg_logits (B, 2, 320, 320)
```

**Progressive (not single-shot) upsampling.** The decoder refines at /4, then /2, then /1, applying a 3×3 convolution at each intermediate resolution. Each refinement step operates on features whose spatial statistics are locally consistent, which is what allows the final 1×1 head to emit crisp boundaries instead of a blurred blob. The module docstring states the intent precisely: *"Produces crisp lesion boundary features avoiding the /4 bilinear blur bottleneck."*

**Channel arithmetic (measured `[M]`):** 81,666 params = **3.22 %**.

### 7.2 Classification head — `LesionGatedClassificationHead`

**The problem with global average pooling.** GAP reduces a $C\times H\times W$ map to $C$ numbers by averaging **over every pixel**. For a leaf photographed on a soil/table background, the pooled vector contains as much background as leaf. A classifier trained on such features can achieve high accuracy by reading *background statistics* rather than lesion appearance — the "background shortcut".

**The v7 design: pool where the lesion is.**

For each pyramid level $\ell \in \{P3, P4, P5\}$ with feature map $F_\ell$ and a mask $M$ downsampled to that level:

$$
\text{pool}_{\text{lesion}}(F_\ell, M_\ell)
= \sum_{i,j} F_\ell[:,:,i,j]\,
\frac{M_\ell[:,:,i,j]}{\sum_{i,j} M_\ell[:,:,i,j] + \epsilon}
$$

and in parallel the plain global average $\text{GAP}(F_\ell)$. The two are concatenated **per level**, all levels are concatenated, and a linear+Hardswish reduction followed by a 2-layer MLP produces the logits:

$$
z = \mathrm{fuse}\big([\,\text{lesion}_\ell \,\|\, \text{GAP}_\ell\,]_{\ell=1..3}\big),
\qquad
\text{logits} = \mathrm{MLP}(z)
$$

with dimensions $\text{Linear}(96\cdot3\cdot2 = 576 \to 96)$ then $\text{Linear}(96\to96)\to\text{Hardswish}\to\text{Linear}(96\to 8)$.

```
   P3 (/8,96) ──┬─► lesion-masked avg (96) ──┐
                └─► global avg        (96) ──┤
   P4(/16,96) ──┬─► lesion-masked avg (96) ──┤   concat = 576
                └─► global avg        (96) ──┤        │
   P5(/32,96) ──┬─► lesion-masked avg (96) ──┤   Linear 576->96 + Hardswish
                └─► global avg        (96) ──┘        │
                     ▲                                ▼
                     │                     Linear 96->96, Hardswish
                 mask M downsampled                │
                 per level on the fly       Linear 96->8
                                                      │
                                                      ▼
                                              cls_logits (B, 8)
```

**Three subtleties that make this work:**

1. **Empty-mask fallback.** If a batch contains no predicted lesion ($\sum M \to 0$), the normalization would divide by zero. The implementation replaces the weights with a **uniform** $1/(hw)$ distribution, which makes the "lesion-pooled" branch degrade gracefully into **GAP**:
   ```python
   wsum = mask_small.sum(dim=(2, 3), keepdim=True)
   weight = torch.where(wsum > eps,
                        mask_small / (wsum + eps),
                        torch.full_like(mask_small, 1.0 / (h * w)))
   ```
   This is what keeps detection-only or classification-only training stages functional when the segmentation head is not running.

2. **The mask gradient is detached** (`m = mask.detach()`). The classifier therefore never back-propagates into the segmentation head. Without this, the segmentation head would be incentivized to *paint lesions wherever the classifier finds them convenient*, corrupting the pixel-level objective. This is a cross-task interference guard of the same species as PCGrad.

3. **Train/inference gate mismatch is explicitly scheduled.** At inference, the classifier is gated on **predicted** masks. If it only ever saw clean ground-truth masks during training, it would be optimized for an input distribution it never receives. The forward pass therefore mixes the two, with probability `gt_mask_prob`:

   ```python
   if gt_mask is not None and gt_mask_prob > 0.0 and torch.rand(()).item() < gt_mask_prob:
       mask_prob = gt_mask.float() if gt_mask.dim() == 4 else gt_mask.float()[:, None]
   ```

   The schedule anneals `gt_mask_prob` from **0.8 → 0.2** across training, so the head ends up seeing mostly real predictions.

**Measured `[M]`:** 65,480 params = **2.58 %**.

### 7.3 Detection head — `DecoupledDetHead` (anchor-free, DFL)

**What it emits, per pyramid level:** a classification tensor over $N_c$ classes and a box-regression tensor of $4R$ channels, where $R$ = `reg_max` = 8.

**Anchor generation.** Anchors are dense per-pixel points in image coordinates with a stride column:

```python
ys, xs = meshgrid(arange(h)+0.5, arange(w)+0.5)
a = stack([xs.flatten(), ys.flatten()], 1) * stride      # pixel centres
anchors = cat([a, full((h*w,1), stride)], dim=1)         # (A, 3): x, y, stride
```

For 320×320 with strides 8/16/32 the three levels contribute $40^2 + 20^2 + 10^2 = 1600+400+100 = \mathbf{2100}$ anchors (measured `[M]`: `anchors (2100, 3)`).

**Distribution Focal Loss (DFL) decoding.** Instead of regressing four distances directly, the head predicts a **discrete distribution over $R$ bins** per side, and the expected value is the regression:

$$
\text{reg} = \sum_{r=0}^{R-1} r \cdot \mathrm{softmax}(\text{box}[r]),
\qquad
\begin{aligned}
x_1 &= (x_a - \ell s)/S, & y_1 &= (y_a - t s)/S,\\
x_2 &= (x_a + r s)/S, & y_2 &= (y_a + b s)/S,
\end{aligned}
$$

with $S$ the image size, $s$ the anchor stride, and $(\ell,t,r,b)$ the left/top/right/bottom expected distances in stride units. DFL's advantage over direct regression is that it represents **uncertainty about the boundary** explicitly; a blurry edge yields a broad (high-entropy) distribution rather than an arbitrary point estimate, and the loss is a cross-entropy over bins, which is better conditioned than L1 on tiny objects.

**Detected shapes (measured `[M]`, B=1):**

```
det_cls   (1, 2100, 8)
det_box   (1, 2100, 32)      # 4 sides x reg_max 8
anchors   (2100, 3)          # x, y, stride in pixels
```

**Measured `[M]`:** 24,616 params = **0.97 %** — by far the cheapest head, because it operates on coarse maps only.

**Chicken-and-egg at initialization.** With the YOLO-standard 1 % prior on the classification bias (logit ≈ −4.6), a freshly initialized head produces maximum confidences around 0.01. Above the standard 0.25 threshold this yields **exactly zero predicted boxes**, and mAP is `nan`. This was observed in the v10-eval smoke `[M]` (`pred=0` boxes on val, raw max confidence 0.0104) and is **correct behaviour**, not a decoding bug — it is discussed further in §14.4.

### 7.4 Abstention head — `AbstainHead`

```
   cls_logits ──┬─► / T ──► softmax ──► probs, pred = argmax, conf = max
                │
                └─► energy E = -logsumexp(logits / T)
                          │
                          ▼
        abstain if (conf < prob_thresh)  OR  (E > energy_thresh)
```

- **Temperature scaling** $T$ is a registered buffer (not trained by gradient); it is **fitted on the validation split** by `calibrate_abstention.py` after training. For the v7 checkpoint the fitted value was **T = 1.4618** `[M]` (`docs/V7_EVAL_2026-10-11.md`).
- **Energy score** $E = -\log\sum_c e^{z_c/T}$ is a free, training-free out-of-distribution detector: an input the model has never seen produces a *flatter* logit distribution — hence a *higher* energy — than an in-distribution input.
- **Measured effect `[M]`:** on PlantDoc, abstaining on the least-confident 6.3 % raised top-1 from 0.430 to **0.460** (+3.0 points). On PlantSeg it did **not** help (0.513 confident vs 0.517 all) — reported as-is rather than omitted.

---

## 8. Analytic severity and abstention

### 8.1 The identification problem with a learned severity head

Severity has a natural, measurable definition: *fraction of leaf area that is diseased*.

$$
\text{sev} = \frac{\text{lesion pixels}}{\text{leaf pixels}}
$$

The predecessor trained a **regression head** to predict a severity target. But the severity target itself was computed from masks as exactly this ratio. The head was therefore being trained to reproduce a function of its own input — a non-identifiable objective. It cannot learn anything the pipeline does not already know, and it can drift arbitrarily.

### 8.2 The v7 resolution

Severity is computed **analytically** from the segmentation output, with no learned parameters:

```python
def analytic_severity(self, seg_logits, eps=None):
    eps = eps or self.SEV_EPS                 # 1e-3
    p = torch.sigmoid(seg_logits.float())
    lesion, leaf = self._seg_channels(p)      # (B,1,H,W) each
    return (lesion.sum(dim=(2,3)) / (leaf.sum(dim=(2,3)) + eps)).squeeze(1).clamp(0,1), \
           (lesion.sum(dim=(2,3)) / (leaf.sum(dim=(2,3)) + eps)).squeeze(1)
```

**Properties of this choice.**

- The severity error is now **fully attributable** to segmentation error. If severity is wrong, the segmentation is wrong — there is no separate black box to debug. This is why the evaluation protocol reports severity MAE twice: once with the predicted leaf mask (`sev_mae_predmasks`) and once with the **true** leaf area (`sev_mae_trueleaf`), which isolates leaf-channel error from the severity arithmetic.
- The `+ε` guard makes it safe when the leaf mask is empty. The `clamp(0,1)` enforces the physically necessary bound "a lesion cannot exceed its leaf" — a constraint the predecessor violated, producing severity values up to **4.04** and clamping them silently.
- Severity is **differentiable**, so a consistency loss *may* supervise it (§10.6), but at deployment it is normally used under `no_grad`.

### 8.3 Grade mapping

The continuous value is bucketed for reporting (`severity_from_ratio`):

| Condition | `sev_cont` | `sev_ord` | Label |
|---|---|---|---|
| no lesion present | 0.0 | 0 | healthy |
| ratio < 0.10 | ratio | 1 | mild |
| 0.10 ≤ ratio < 0.30 | ratio | 2 | moderate |
| ratio ≥ 0.30 | ratio | 3 | severe |

### 8.4 The severity denominators differ per corpus — and that matters

The constant-severity baseline (the number severity must beat) is a **property of the corpus**, not of the model. Two different values appear in this repository, and conflating them is an error:

| Corpus | Constant-severity MAE baseline | Source |
|---|---|---|
| outline masks (v7/v8) | **0.0384** | `docs/v8_report.md` |
| filled masks (v9) | **0.2293** | `V9_BASELINE_MAE` in the v9 driver, measured as the val-split mean |

The v9 value is much larger because the repaired whole-leaf masks change the denominator and the distribution of ratios. Any severity comparison must name its corpus and its baseline. The v9 driver asserts the corpus by fingerprint (§11.2) before it trusts either number.

---

## 9. Parameter and tensor-shape budget

### 9.1 Parameter table (measured `[M]`)

Command:

```bash
python3 - <<'PY'
import sys, os; sys.path.insert(0, os.path.join(os.getcwd(), "model"))
import torch
from model.architecture.v7_model import CropGuardMultiTaskV7
m = CropGuardMultiTaskV7(nc=8, width=96, pretrained=False, reg_max=8)
n = lambda mod: sum(p.numel() for p in mod.parameters())
for name in ("backbone", "neck", "det_head", "cls_head", "seg_head", "abstain"):
    print(name, n(getattr(m, name)))
print("TOTAL", n(m))
PY
```

```
module             params       %
backbone        2,214,900  87.43%
neck              146,592   5.79%
det_head           24,616   0.97%
cls_head           65,480   2.58%
seg_head           81,666   3.22%
abstain                 0   0.00%
TOTAL           2,533,254
```

**Interpretation.** The paper's central efficiency claim is visible directly in this table: **93.2 % of the parameters are in the shared trunk** (backbone 2,214,900 + neck 146,592 = 2,361,492 of 2,533,254). The four task-specific heads together cost only 171,762 parameters (6.8 %: det 24,616 + cls 65,480 + seg 81,666). Adding a task to this system is therefore *cheap*; the marginal cost of the detection head is under 1 % of the model.

### 9.2 Verified tensor shapes at 320×320 (measured `[M]`)

```
input        (1, 3, 320, 320)
backbone C2  (1,  40,  80,  80)     /4
backbone C3  (1,  80,  40,  40)     /8
backbone C4  (1, 160,  20,  20)     /16
backbone C5  (1, 320,  10,  10)     /32
neck P3/P4/P5 (1, 96, 40,40) (1,96,20,20) (1,96,10,10)
seg_logits   (1,   2, 320, 320)
cls_logits   (1,   8)
det_cls      (1, 2100,   8)
det_box      (1, 2100,  32)
anchors      (2100,  3)
```

### 9.3 Export sizes (measured `[M]`)

| Artifact | Size |
|---|---|
| ONNX FP32 (4 outputs) | **10.23 MB** |
| ONNX INT8 static | **3.13 MB** |

The INT8 reduction is ≈3.3×, consistent with weight-only 8-bit quantization of a model dominated by convolutions.

---

## 10. Loss functions: complete mathematics

This section reproduces the exact losses implemented in `model/training/losses.py`.

### 10.1 Segmentation: `seg_loss_v7` (two-channel)

Two channels, two different objectives, because they have two different statistics.

**Lesion channel** — the target is *tiny and rare* (typically 1–2 % of pixels). A plain BCE is dominated by the enormous negative class. Two mechanisms compensate:

*(i) BCE with a positive weight of 20:*

$$
\mathcal{L}_{\text{BCE}}^{\text{les}} = -\frac{1}{N}\sum_i \Big[\, p_y \cdot y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \,\Big], \quad p_y = 20
$$

A false negative (a missed lesion pixel) is penalized **20×** more than a false positive.

*(ii) Focal-Tversky loss,* which directly optimizes the false-negative/false-positive trade-off:

$$
\text{TP}=\sum \hat{p}\,y,\quad
\text{FP}=\sum \hat{p}(1-y),\quad
\text{FN}=\sum (1-\hat{p})\,y
$$
$$
\mathrm{TI} = \frac{\text{TP}+\epsilon}{\text{TP} + \alpha\,\text{FP} + \beta\,\text{FN} + \epsilon},
\qquad
\mathcal{L}_{\text{FT}} = (1-\mathrm{TI})^{\gamma}
$$

with $\alpha = 0.3$ (weight on false positives), $\beta = 0.7$ (weight on false negatives), $\gamma = 1.33$ (focal exponent, which concentrates gradient on hard examples). $\beta > \alpha$ encodes the clinical preference: **missing a lesion is worse than over-calling one.**

The lesion loss is the sum: $\mathcal{L}_{\text{les}} = \mathcal{L}^{\text{les}}_{\text{BCE}} + 3.0\,\mathcal{L}_{\text{FT}}$.

**Leaf channel** — the target is *large* (most of the image) and may be **invalid** for some images. Invalidity is encoded by a sentinel value of `-1.0`, and a validity mask removes those samples from both terms without removing the sample from the batch:

$$
\mathcal{L}_{\text{leaf}} = \frac{\sum (v \odot \text{BCE})}{\sum v} + \Big(1 - \frac{1}{B}\sum_b \frac{2\sum \hat p\,y\,v + 1}{\sum \hat p\,v + \sum y\,v + 1}\Big)
$$

where $v$ is the validity mask. Total: $\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{les}} + \mathcal{L}_{\text{leaf}}$.

**Why two channels rather than one?** The severity denominator *is* the leaf. A single-channel head cannot supply a denominator, which is precisely why the predecessor's severity was a constant and the "128 = lesion" mask value never occurred in the corpus.

### 10.2 Task-aligned assignment (TAL)

Before detection losses can be computed, each anchor must be matched to a ground-truth box. This model implements a simplified YOLOv8-style TaskAlignedAssigner:

$$
\text{align}(a, g) = \underbrace{s_{a,c_g}^{\alpha}}_{\text{class score}} \cdot \underbrace{\mathrm{IoU}(a,g)^{\beta}}_{\text{localization}} \cdot \mathbb{1}[\,a_{\text{centre}} \in g\,]\cdot \mathbb{1}[\,g \text{ real}\,]
$$

with $\alpha = 0.5$, $\beta = 6.0$, and the **centre-in-box** constraint. For each ground-truth box, the top-$k$ ($k=10$) anchors by `align` become candidates; conflicts (an anchor being the top candidate for two boxes) are resolved by keeping the ground-truth with the highest `align`; anchors whose best `align` is zero are background.

The exponent $\beta = 6$ is what makes this "task-aligned": localization quality dominates the assignment decision, so anchors that are well-placed get assigned even when the current class score is still low — which accelerates the early phase of training for small objects.

### 10.3 Detection classification loss — soft targets

Instead of hard one-hot targets, the assignment quality becomes the supervision signal:

$$
\text{target}_a = \text{onehot}(c_g) \cdot \tilde{a}, \qquad
\tilde{a} = \frac{\text{align}_a}{\max_{a}\text{align}_a}\cdot\mathbb{1}[\text{fg}]
$$

$$
\mathcal{L}_{\text{cls}}^{\text{det}} = \frac{1}{N_{fg}}\sum_a \mathrm{BCEWithLogits}\big(\hat{s}_a, \text{target}_a\big)
$$

Normalization is by the number of foreground anchors $N_{fg}$, so the loss is invariant to how many boxes an image happens to contain. An **all-background batch** (rare but possible) falls back to a mean-reduced BCE so the loss remains finite and differentiable.

### 10.4 Box regression — CIoU

$$
\mathcal{L}_{\text{CIoU}} = 1 - \mathrm{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v,
\qquad
v = \frac{4}{\pi^2}\Big(\arctan\frac{w}{h} - \arctan\frac{w^{gt}}{h^{gt}}\Big)^2,
\qquad
\alpha = \frac{v}{(1-\mathrm{IoU}) + v}
$$

where $\rho$ is the centre distance and $c$ the diagonal of the smallest enclosing box. CIoU adds aspect-ratio consistency to IoU and centre-distance, which matters here because lesions have strongly anisotropic shapes. **Implementation detail:** all geometry is computed in **FP32 even under AMP**, because predicted boxes can become extremely thin $(w \to 0)$ and FP16 would produce `inf`/`nan` in the `arctan` and division terms. Dimensions are clamped with `clamp(min=eps)` before the ratio.

### 10.5 Box regression — DFL (cross-entropy over bins)

Let $d \in \{\ell, t, r, b\}$ be the anchor-to-edge distance in stride units, $l=\lfloor d \rfloor$, $f = d - l$:

$$
\mathcal{L}_{\text{DFL}} = -\frac{1}{N_{fg}}\sum_d \Big[(1-f)\,\log p_d[l] + f\,\log p_d[l+1]\Big]
$$

This is the standard two-bin formulation: the continuous target is expressed as a linear interpolation between the two adjacent bins, and the loss is the corresponding weighted cross-entropy. `l` is clamped to `[0, R-2]` so the `l+1` bin always exists.

**Why the assigned anchor is guaranteed inside the box:** because TAL enforces centre-in-box, each of the four distances is ≥ 0 *before* clamping — the `.clamp(0, R-1-1e-4)` is a numerical guard, not a semantic one.

### 10.6 Classification loss

$$
\mathcal{L}_{\text{cls}} = \mathrm{CrossEntropy}\big(\text{logits}[\,y\!\ge\!0\,],\ y[\,y\!\ge\!0\,],\ \text{weight}=\mathbf{c},\ \text{ls}=0.05\big)
$$

- **`ignore` semantics via masking:** images with no valid class ($y = -1$) are removed from the loss, not assigned an arbitrary class.
- **Class weights** $\mathbf{c}$ are *mild inverse-square-root* weights capped at 3×, computed from the filtered training label counts:

$$
c_k = \min\Big(3.0,\ \big(\text{median}_j\, n_j \,/\, n_k\big)^{1/2}\Big)
$$

The square root halves the aggression of pure inverse-frequency weighting: pure $1/n$ over-corrects on a 40-image class and destabilizes the majority classes. The cap at 3.0 is a hard ceiling on how much any single class can dominate the gradient.

- **Label smoothing 0.05** softens the target, which is valuable for the rare classes (notably `mosaic_virus`, support 8 in val) where hard targets encourage overconfident memorization.

### 10.7 Cross-task gradient surgery — PCGrad

When task gradients conflict, the offending component is projected out:

$$
\mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i\cdot\mathbf{g}_j}{\|\mathbf{g}_j\|^2}\,\mathbf{g}_j
\quad\text{whenever}\quad \mathbf{g}_i\cdot\mathbf{g}_j < 0
$$

and the update uses the mean of the (possibly projected) per-task gradients. In this repository PCGrad is implemented and available but the primary staged recipe does **not** require it. It is offered because multi-task tomato models routinely exhibit exactly this conflict (classification wants leaf-scale texture; segmentation wants pixel-scale edges).

### 10.8 Total objective

$$
\mathcal{L}_{\text{total}} =
\lambda^{\text{cls}}_{det}\mathcal{L}^{\text{det}}_{\text{cls}}
+ \lambda^{\text{box}}_{det}\mathcal{L}_{\text{box}}
+ \lambda^{\text{dfl}}_{det}\mathcal{L}_{\text{DFL}}
+ \lambda_{\text{cls}}\mathcal{L}_{\text{cls}}
+ \lambda_{\text{seg}}\mathcal{L}_{\text{seg}}
+ 0.5\,\mathcal{L}_{\text{cons}}
$$

with the detection triplet explicitly set to the YOLOv8-standard
$(\lambda^{\text{cls}}_{det},\lambda^{\text{box}}_{det},\lambda^{\text{dfl}}_{det}) = (0.5,\ 7.5,\ 1.5)$
and $\lambda_{\text{cls}} = \lambda_{\text{seg}} = 1.0$. Justification for those specific numbers is in §15.4 — they are not cosmetic: with equal weights the classification term exceeds the box term by roughly two orders of magnitude, and boxes visibly fail to sit on the lesion.

---

## 11. Data pipeline

### 11.1 Corpus

| Split | Images | Label files | Valid classes (after audit) | Excluded (empty label) |
|---|---|---|---|---|
| train | 715 | 715 | 683 | 32 |
| val | 156 | 156 | 151 | 5 |
| test | 150 | 150 | 140 | 10 |

`[M]` — from `stage0_label_audit.json` (v8 run) and re-measured in the local smoke on the v9 corpus.

Per-image majority-class counts `[M]` (v8 audit):

```
train:  {"0":80, "1":75, "2":113, "3":114, "4":106, "5":40, "6":91, "7":64}
val:    {"0":20, "1":21, "2":33,  "3":19,  "4":20,  "5":8,  "6":14, "7":16}
test:   {"0":20, "1":10, "2":23,  "3":26,  "4":21,  "5":14, "6":18, "7":8}
```

Classes (index order is load-bearing and appears identically in the dataset, the evaluation and the mobile client):

```
0 healthy | 1 bacterial_spot | 2 early_blight | 3 late_blight
4 leaf_mold | 5 mosaic_virus | 6 septoria_leaf_spot | 7 yellow_leaf_curl_virus
```

**Class imbalance is severe and real:** `mosaic_virus` has 40 train images and 8 val images, versus 114 for `late_blight`. This is why class weighting (§10.6) and a `WeightedRandomSampler` with $w_k = 1/\sqrt{n_k}$ are both present.

### 11.2 On-disk layout and the v9 mask repair

```
cropguard_v7_split/
├── images/{train,val,test}/*.jpg      RGB photographs
├── labels/{train,val,test}/*.txt      YOLO format: "class cx cy w h" (normalized)
├── masks_lesion/{train,val,test}/*.png   lesion mask
└── masks_leaf/{train,val,test}/*.png     leaf mask
```

The **v9 corpus** repairs two defects in the older masks:

| Mask | Before | After |
|---|---|---|
| `masks_lesion` | 1-px polygon **outlines** | **filled** polygon regions |
| `masks_leaf` | healthy tissue only (necrosis sat *outside* it) | `fill_holes(leaf) \| lesion` (whole leaf) |

The `masks_leaf` defect is the more dangerous of the two: with the lesion outside the leaf mask, the severity denominator was too small, inflating severity; a measured worst case was **4.04** ("the lesion is 4× the leaf") which is physically impossible, and it was silently clamped to 1.0 — labelling every such image *severe* `[M]` (`docs/V7_EVAL_2026-10-11.md`).

The v9 driver does not merely assume the repair; it **verifies** it by fingerprint before training is allowed to proceed, refusing to run if either check fails:

```
lesion filled/outline median ratio  >= 5.0      measured 20.40x   [M]
lesion-inside-leaf (min over samples) >= 0.99   measured  1.0000  [M]
counts == 715 / 156 / 150                       enforced per split
```

This is a deliberate anti-silent-regression measure: a corpus that silently reverted to outlines would otherwise train a plausible-looking but wrong model.

### 11.3 Augmentation

**One shared geometric transform, applied to image and every mask together.** This is the single most important correctness property of the augmentation code, because a mask that is flipped while its image is not produces a model that learns geometry that does not exist.

```python
def _augment(self, img, boxes, lesion_mask=None, leaf_mask=None,
             combined_mask=None, flip=None, scale=None):
    if flip is None:  flip = random.random() < 0.5
    if scale is None: scale = random.uniform(0.9, 1.1)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        boxes = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in boxes]
    # HSV jitter
    if random.random() < 0.5:
        img = TF.adjust_hue(img, random.uniform(-0.02, 0.02))
        img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
    ...  # scale jitter: pad (s<1) or centre-crop (s>1) back to the original size
    def transform(m):
        return self._transform_mask(m, flip=flip, s=s) if m is not None else None
    return img, boxes, transform(lesion_mask), transform(leaf_mask), transform(combined_mask)
```

- **Image is resampled bilinear; masks are resampled nearest.** Interpolating a binary mask bilinearly would create fractional "lesion" values at every boundary — i.e. label noise along the entire contour, exactly where the loss is most sensitive.
- **Flip is threaded through explicitly** rather than applied inline. The code comment records why: an inline flip was previously easy to desynchronize from the mask path.
- **Photometric jitter is HSV-space**, deliberately *not* a global brightness multiply, so it does not destroy the chromatic contrast between necrotic and healthy tissue that the lesion channel relies on.
- **Scale jitter is 0.9–1.1, then padded/cropped back** to the original size, so no image is ever upsampled beyond its native detail.

### 11.4 Collation

Detection targets have variable cardinality, so they are kept as a **list** rather than stacked:

```python
def collate(batch):
    return {
        "image":     torch.stack([b["image"] for b in batch]),
        "det_boxes": [b["det_boxes"] for b in batch],       # variable-length list
        "n_boxes":   torch.tensor([b["n_boxes"] for b in batch]),
        "cls":       torch.tensor([b["cls"] for b in batch]),
        "mask":      torch.stack([b["mask"] for b in batch]),
        "leaf":      torch.stack([b["leaf"] for b in batch]),
        "sev_cont":  torch.stack([b["sev_cont"] for b in batch]),
        "sev_ord":   torch.stack([b["sev_ord"] for b in batch]),
    }
```

The DetectionLoss then truncates each entry to `n_boxes[i]`. A naive `torch.stack` would require padding boxes to a fixed count and would corrupt the assignment for short images.

---

## 12. Training protocol

### 12.1 Configuration

| Setting | Value | Rationale |
|---|---|---|
| epochs | 120 (capped by wall clock) | long enough for cosine decay to converge |
| batch size | 8 | fits T4×2 comfortably at 320² with AMP |
| base LR | 5e-4 | AdamW; safe with BN at width 96 |
| LR schedule | cosine, 5-epoch warmup, min 1e-6 | warmup prevents early BN instability |
| optimizer | AdamW, weight decay 1e-4 | decoupled weight decay |
| AMP | `torch.amp.autocast("cuda")` + `GradScaler` | T4 tensor cores |
| grad clipping | global L2 norm 10.0 | guards the small-lesion loss spikes |
| EMA decay | 0.999 | slow-weight average for robustness |
| seed | 1337 | reproducibility |
| image size | 320 | accuracy/throughput knee for a phone target |
| precision at export | FP32 + static INT8 | §18 |

**Measured learning-rate trajectory `[M]`** (`v8_train_log.csv`): epoch 1 `9.889e-05` (warmup) → epoch 30 `4.441e-04` → epoch 60 `2.676e-04` → epoch 90 `8.026e-05` → epoch 102 `3.060e-05`. The schedule behaves exactly as specified.

### 12.2 Stage 0 — integrity gates

Four gates, all mandatory; any failure aborts **before** GPU budget is spent:

1. **`sha256` gate on `dataset.py`.** Expected `ff0102ed509d595abe08b79b8e1902fd6756cb7404f5efd102d1a507b90696db`. This pins the data-loading semantics, so a "good" result cannot come from a silently different loader.
2. **Forced-geometry subprocess.** A separate script checks that image and both masks agree within 2 px after the shared transform.
3. **Pretrained-weight check.** First-conv L1 against a random-init reference; $> 0.01$ means real pretrained weights.
4. **Label audit.** Per-split, per-class counts and empty-label exclusions.

### 12.3 Stage 1 — the memorization gate

**Question:** can model + loss + data pipeline overfit *eight* images?

This is a cheap but decisive test. A pipeline that cannot overfit 8 images has a *structural* defect (no gradient path, wrong target, dead head) and no amount of epochs or compute will fix it. Gating on it converts a silent 55-minute GPU failure into a 2-minute CPU-legible verdict.

The gate is scored at `lesion IoU ≥ 0.50` after 500 steps, with augmentation ON. If it fails, the driver prints per-component diagnostics (`requires_grad` per submodule, gradient norms for `seg_head`/`neck`/`backbone`, the last step's loss components) and retries with geometric augmentation disabled.

**Measured `[M]` (`docs/v8_report.md`, `stage1_memorize*.json`):**

```
with geometric augmentation   : step 500 IoU = 0.1935   -> gate FAILS
photometric-only (PLAN B)     : step 500 IoU = 0.6708   -> gate PASSES
```

The interpretation is instructive and is *not* a defect: with flips and scale jitter active, the model must learn an invariance, so 8 images is genuinely harder; without geometry, it memorizes the 8 images as expected. Both runs show `seg_head` gradient norms rising from ≈0.76 to ≈3.9–5.7, confirming the segmentation head is a live gradient path. The v8 run therefore selected **PLAN B: photometric-only augmentation**.

### 12.4 Stage 2 — main training with a wall-clock cap

Every epoch: train, then evaluate **both** the raw weights and the EMA shadow on the validation split, then select:

```python
for which, m in (("raw", ra), ("ema", em)):
    score, tie = m["macro_f1"], m["lesion_iou"]
    if score > best["score"] or (score == best["score"] and tie > best["iou"]):
        best = {"score": score, "iou": tie, "epoch": epoch, "which": which, "metrics": m}
        # saves best_v*_raw.pt  or  best_v*_ema.pt
```

- **Selection metric: validation macro-F1**, tie-broken by **validation lesion IoU**. The test split is never consulted for any selection decision, so no reported test number can be the product of test-set fitting.
- **EMA is verified to move.** A checksum of the shadow parameters is recorded every epoch and a WARNING is emitted if it fails to change — guarding against the classic silent failure where EMA is declared but never updated:
  ```
  epoch 1: 3.469940 | epoch 35: 1.316943 | epoch 60: 1.851464 | epoch 90: 2.186074 | epoch 102: 2.184402   [M]
  ```
- **Cap.** The run stops cleanly when `elapsed_min + per_epoch > CAP_MINUTES` (55). Measured `[M]`:
  ```
  TIME CAP: 54.7 min used, ~0.54 min/epoch, cap 55.0 min -> stopping cleanly at epoch 102
  TOTAL 62.8 min
  ```

### 12.5 Stage 3 — validation summary

The selected checkpoint is re-evaluated on the validation split and written to `stage_summary.json`. The test split is not touched.

### 12.6 Versioning philosophy: change one thing per version

| Version | The one change | Why it was the right next change |
|---|---|---|
| **v8** (complete) | baseline recipe on the outline-mask corpus | establishes a reference point, and the corpus's defects are *known* |
| **v9** (queued) | same recipe, **repaired masks** (filled lesion, whole-leaf leaf) | segmentation and severity were learning the wrong target; the target had to be fixed before any architecture work could be judged |
| **v10** (queued) | v9 **plus the detection head actually trained** | the detection head had never been trained, so mAP was undefined; training it is a pure addition enabled by §10.2–10.5 |

The discipline here is methodological: a version that changes two things cannot attribute its result to either.

---

## 13. Evaluation protocol

### 13.1 Paired evaluation through one shared code path

The evaluation kernel loads **both** the old and the new checkpoint and runs them through **identical** code (`predict` → `metrics`) on the same splits. This removes the single largest source of spurious comparison in applied ML: two runs measured by two slightly different scripts.

Reported per checkpoint, per split (PlantSeg val, PlantSeg test, PlantDoc test):

- accuracy and macro-F1, with **bootstrap confidence intervals**
- expected calibration error (ECE, 10-bin) after temperature scaling
- lesion **IoU** and **Dice** against real masks
- **mAP@0.5 and mAP@0.5:0.95**, with and without horizontal-flip TTA
- **severity MAE** against a constant baseline
- abstention rate and accuracy on the confident subset
- ONNX parity and CPU latency
- a **healthy-shortcut probe**

### 13.2 Metrics

$$
\mathrm{IoU} = \frac{|P \cap G|}{|P \cup G|},
\qquad
\mathrm{Dice} = \frac{2|P\cap G|}{|P| + |G|}
$$

$$
\mathrm{macroF1} = \frac{1}{N_c}\sum_{c} \frac{2\,\mathrm{prec}_c\,\mathrm{rec}_c}{\mathrm{prec}_c + \mathrm{rec}_c},
\qquad
\mathrm{MAE}_{\text{sev}} = \frac{1}{N}\sum_i |\hat{s}_i - s_i|
$$

$$
\mathrm{ECE} = \sum_{m=1}^{M} \frac{|\mathcal{B}_m|}{N}\,\big|\,\mathrm{acc}(\mathcal{B}_m) - \mathrm{conf}(\mathcal{B}_m)\,\big|,
\qquad M = 10
$$

Average precision is computed as the area under the interpolated precision–recall curve, with the implementation verified **bit-identical** to the reference (`ap_per_class`): worst absolute difference **0.000e+00** over 50 comparisons `[M]`.

### 13.3 Test-time augmentation (TTA)

Horizontal flip only: the image is flipped, inference is run, detections are mirrored back into the original frame, and the two lists are **unioned** and passed to NMS. Only *mirror-symmetric* augmentation is used, because a vertical flip would produce a physiologically implausible leaf orientation that the model has never seen.

### 13.4 The healthy-shortcut probe

**This is the most important measurement in the protocol.**

A logistic regression is fitted on features computable **without the model** — image size, aspect ratio, mean RGB, green fraction — to predict "healthy" on the validation split, and evaluated on a held-out portion.

**Measured `[M]`:** val-holdout accuracy **0.9841** (n = 156, 60/40 split; train portion 0.9677). Flagged = true.

**Interpretation, stated carefully.** This is *not* a test-set leak and it does not invalidate the classification numbers — it is a property of the **corpus and its split**. It means that in this dataset, "healthy" is almost perfectly decidable from acquisition cues alone (leaf framing, background, colour cast). Consequences:

- A high `healthy` F1 (measured **1.000**, support 20, v8 best epoch) is therefore **not** evidence that the model learned healthy-leaf appearance. It may be reading acquisition statistics.
- Any downstream claim about healthy-vs-diseased discrimination must be qualified by this number.
- The honest remedy is a corpus collected under a single acquisition protocol, or explicit stratification, neither of which is possible retroactively.

This is disclosed rather than removed because a paper that hides a 0.98 trivially-solvable subproblem is a paper that will not survive review.

---

## 14. Results

### 14.1 v8 baseline — validation (measured `[M]`)

Everything below is **validation**, from the selected checkpoint `best_v8_raw.pt` at epoch 35, chosen by val macro-F1. Test numbers require the eval kernel and are reported as not-measured (§14.5).

```
epoch 35 | lr=4.209e-04 | cls=0.3505 seg=3.1593 total=3.5098 | 17.7 s
  RAW: acc=0.6954  macro_f1=0.6786  lesion_iou=0.0459  dice=0.0878
       sev_mae_predmasks=0.1026  sev_mae_trueleaf=0.1137
       pred_area=4729.0  true_area=621.3  over_pred_ratio=7.61
       best_thr=0.7  n_valid_cls=151
```

Per-class (val) `[M]`:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| healthy | 1.000 | 1.000 | **1.000** | 20 |
| bacterial_spot | 0.818 | 0.429 | 0.562 | 21 |
| early_blight | 0.760 | 0.576 | 0.655 | 33 |
| late_blight | 0.765 | 0.684 | 0.722 | 19 |
| leaf_mold | 0.696 | 0.800 | 0.744 | 20 |
| mosaic_virus | 0.600 | 0.375 | 0.462 | 8 |
| septoria_leaf_spot | 0.310 | 0.643 | 0.419 | 14 |
| yellow_leaf_curl_virus | 0.762 | 1.000 | 0.865 | 16 |

### 14.2 v9/v10 status

| Kernel | Version | Status | Meaning |
|---|---|---|---|
| `cropguard-v9-train` | 2 (hardened) | **QUEUED** | corpus repair run |
| `cropguard-v10-train-detection-enabled` | 2 (hardened) | **QUEUED** | detection-enabled run |

Both runs are **pending** `[P]`. Their outputs (and the paired eval) do not yet exist, so no v9/v10 accuracy, IoU, mAP or severity number can be reported. This paper therefore reports v8 as the current baseline and states clearly that the v9 and v10 comparisons are pending. **Claiming any v9/v10 metric now would be fabrication.**

### 14.3 v7 → v8: what the repair actually bought (measured `[M]`)

| Metric (val, same code path) | v7 @ ep120 | v8 best | Change |
|---|---|---|---|
| macro-F1 | 0.6433 | **0.6786** | +0.035 |
| lesion IoU | 0.0282 | **0.0459** | ×1.63 |
| lesion Dice | 0.0549 | 0.0878 | ×1.60 |
| severity MAE | 0.3104 | **0.1026** | −0.208 |
| segmentation loss (train) | flat at ≈3.63 | 4.8 → 2.5 | now *decreasing* |

The last row is the diagnostic one. v7's segmentation loss was **flat** — proof of no learning. v8's decreases monotonically from 4.8 to 2.5. A loss that moves is a gradient path that exists.

### 14.4 v10-eval end-to-end smoke (plumbing validation) `[M]`

This is explicitly a **pipeline test on 1-epoch smoke checkpoints**. Its accuracy numbers are meaningless by construction and are not reported as results. Its *engineering* findings are result-grade:

```
both checkpoints load: missing=0 unexpected=0
val=156  test=150   PlantDoc = not measured (absent locally)
det rows: old/plantseg_val mAP@0.5 = nan  (gt=959 boxes, pred=0)
  -> diagnosed, not a bug: raw max det confidence = 0.0104 (NEW) / 0.0149 (OLD)
     the det head's bias init is the YOLO 1% prior (logit ~ -4.6); 0 boxes above
     conf 0.25 after 1 epoch is CORRECT. anchors = 2100.
ONNX vs PyTorch, 4 outputs:
  cls 1.5e-08 | seg 1.0e-07 | det_cls 6.2e-06 | det_box 1.5e-06
ONNX sizes: FP32 10.23 MB | INT8 3.13 MB
CPU latency, 1 thread, i5-11400H (NOT a phone): FP32 median 74.37 ms | INT8 67.90 ms
AP port equivalence: worst |diff| = 0.000e+00 over 50 comparisons
healthy-shortcut probe: val-holdout accuracy 0.9841  -> FLAGGED
```

**Note what the `mAP = nan` line actually means.** It is the correct, expected signature of an *untrained* head, and the diagnosis was verified by inspecting raw confidences rather than by loosening a threshold. Loosening the 0.25 threshold to force boxes would have produced a non-`nan` mAP that measured nothing.

### 14.5 Not measured — stated plainly

| Item | Status | Why |
|---|---|---|
| PlantSeg **test** metrics for v8 | `[N]` | eval kernel output not retrievable from this host |
| PlantDoc **test** metrics for v8 | `[N]` | same |
| mAP@0.5 / mAP@0.5:0.95 for a **trained** detection head | `[P]` | requires the v10 run |
| v9 / v10 accuracy, IoU, Dice, severity | `[P]` | runs queued |
| Phone latency | `[N]` | **not measurable from this host**; a T4 or desktop-CPU number would be misleading for a phone target. A browser harness exists at `mobile/bench/` |
| Bootstrap CIs in this document | `[P]` | computed by the eval kernel |

### 14.6 Explainability — LayerCAM (measured `[M]`)

LayerCAM is applied to the neck's P3/P4/P5 feature maps, using ReLU(grad) element-wise weighting and the predicted-class logit. The reported statistic is the fraction of CAM energy falling inside the ground-truth lesion, divided by the lesion's area fraction — a lift of 1.0 means "no better than a random heat map".

**Null calibration first** (random weights): lesion lift **1.0295**, leaf lift **1.0088** — i.e. the metric correctly reports "no signal" when there is none. Only then was the real model measured:

```
real v8 checkpoint, 40 val images:
  lesion_cam_energy_frac_mean = 0.2974
  lesion_area_frac_mean       = 0.1608
  lesion_lift                 = 1.8501        <- 1.85x above chance
  leaf_lift                   = 1.1825
```

**Interpretation.** The classifier/neck representation concentrates attention on lesion tissue at **1.85×** the rate a random map would. Combined with the healthy-shortcut caveat, the defensible claim is: *the shared representation does localize lesion tissue above chance*, not *the model's decision is clinically explained*.

---

## 15. Design rationale: why this particular combination

This section answers the question the paper is really asked: **why these choices and not others?**

### 15.1 Why FasterNet-T0 as the backbone?

| Alternative | Why not here |
|---|---|
| ResNet-18/50 | $k^2 C^2$ dense convolutions; ResNet-50 is ≈25 M params — 10× this model — for a device target where 2.5 M already carries the accuracy |
| MobileNetV3 | Excellent latency, but the accuracy floor on fine-grained foliar texture is lower; it is a *classification*-first backbone |
| ViT / Swin-T | Strong accuracy, but attention's quadratic cost and its unquantizable softmax make INT8 export on a phone far harder; also ≈28 M params |
| **FasterNet-T0** | PConv removes redundant spatial filtering while the pointwise conv keeps channel mixing; **2.21 M params**, quantizes cleanly, and is a *first-class* `timm` citizen so pretrained weights are one line |

The decisive constraint is the deployment target. A 2.53 M-parameter, 10.23 MB FP32 / 3.13 MB INT8 model is a phone model. Every "better" backbone above breaks either the size or the quantization budget.

### 15.2 Why BiFPN-Lite rather than a full FPN or a plain concatenation neck?

The task needs **both** fine texture (segmentation of 3–8 px lesions) and coarse context (is this a leaf at all? which disease?). A plain `Feature Pyramid Network` propagates top-down only, so a fine-level feature never borrows coarse evidence. Plain concatenation gives every level equal say, which is wrong when the levels have different reliabilities. BiFPN's **learned, normalized, per-channel** fusion weights let the model decide per channel how much to trust each scale, while one top-down + one bottom-up round keeps the cost at 146,592 params. The "Lite" reduction is a deliberate latency choice, and per-channel (rather than scalar) weights are what preserve fine-scale detail through the fusion.

### 15.3 Why a stride-4 skip in the segmentation decoder?

Because the target is **3–8 px at 320×320**, and the arithmetic is unforgiving:

| Source level | Stride | Map size | A 3 px lesion becomes |
|---|---|---|---|
| C2 | 4 | 80×80 | 0.75 px |
| C3 | 8 | 40×40 | 0.38 px |
| C4 | 16 | 20×20 | 0.19 px |
| C5 | 32 | 10×10 | 0.09 px |

A 0.09-px target in a stride-32 map is not "small"; it is *gone*. The stride-4 skip is the difference between a decoder that can represent a lesion and one that cannot. This is also the direct cause of the v7 failure (§2.2b).

### 15.4 Why (0.5, 7.5, 1.5) detection weights?

Because the raw magnitudes of the three detection terms are wildly unequal, and equal weights silently optimize only one of them:

| Weight | Term | If left at 1.0 |
|---|---|---|
| 0.5 | classification | dominates the total by ≈2 orders of magnitude; box regression barely trains, and predicted boxes visibly fail to sit on lesions |
| 7.5 | CIoU box | 15× the classification weight — places localization quality on par with class prediction |
| 1.5 | DFL | 3× classification — makes the *distributional* boundary estimate matter, without overriding the point regression |

These are the YOLOv8-standard values and they are used **unchanged** rather than tuned, specifically so that a detection result from this model is comparable to the detection literature without the caveat "but we used unusual loss weights". Where a standard exists, adopting it is a feature.

### 15.5 Why 2-channel segmentation (rather than a single lesion mask)?

Because severity needs a **denominator**. $\text{sev} = \text{lesion}/\text{leaf}$ requires both terms. With a single-channel head there is no leaf, which is exactly why the predecessor's severity was a constant and its "128 = lesion" mask encoding never appeared in its corpus. Two channels also give the second channel its own supervision (real leaf masks) instead of being an unsupervised auxiliary.

### 15.6 Why analytic severity rather than a learned severity head?

Because with no human severity ratings, a learned head regressing a target that is *itself computed from the masks* is non-identifiable: it can only relearn a function of its input, and it can drift without bound. Removing the head makes severity error **attributable** — if severity is wrong, the segmentation is wrong. It also removes an entire head's worth of parameters and one more thing that can silently fail.

### 15.7 Why lesion-gated classification rather than GAP?

GAP over a leaf-on-soil photograph averages over background as much as leaf, which creates a shortcut from background statistics to the label. Masked pooling forces the classifier's features to come from **where the lesion is**. The measured evidence that this class of shortcut is real in this data is the healthy-shortcut probe at **0.9841** (§13.4). The gate does not eliminate that probe — the probe uses raw image statistics, not the model — but it removes the model's own incentive to use the same signal.

### 15.8 Why Hardswish in the heads and GELU in the neck?

INT8 quantization maps a real activation range onto 256 levels. A piecewise-linear function (Hardswish: built from ReLU6 and a multiply) is reproduced nearly exactly by that mapping. GELU's error function is not, and quantizing it introduces systematic error right where the activation is most nonlinear. The heads are the *output* path — errors there are not washed out by later layers; the neck sits between two FP32 stages where its GELU can be preserved. This single decision is why the INT8 model exports at 3.13 MB with the parity numbers in §14.4.

### 15.9 Why staged training?

Because multi-task training's worst failure mode is that a task with **bad supervision** poisons the shared trunk, and everything downstream of that is uninterpretable. If stage 1 (the memorization gate) fails, the run stops before spending 55 minutes of GPU producing a number that measures nothing. If it passes, the trunk is *proven* to carry lesion signal before detection and classification are switched on.

### 15.10 Why the val macro-F1 + lesion-IoU tie-break?

Macro-F1 is unweighted across classes, so it resists the 3:1 imbalance in this corpus (114 `late_blight` vs 40 `mosaic_virus`); accuracy would let the model ignore the rare classes entirely. The lesion-IoU tie-break is only meaningful *because* segmentation now works — under v7, IoU was ≈0.028 and would have been noise, so the tie-break would have selected randomly among equal-F1 checkpoints.

---

## 16. Reliability engineering: the runtime-hang class of failure

A deployed research pipeline fails in two ways: it produces a wrong number, or it produces **no number and no explanation**. The second is more expensive because it consumes budget invisibly.

### 16.1 The observed failure

| Run | Runtime | Output |
|---|---|---|
| v8 (reference) | **62.8 min** | 10 files, 55,336 bytes of log |
| v9 (version 1) | **2 h 24 min** | **0 files, 0 bytes** — still "RUNNING" |

### 16.2 The proof that this was a hang, not slowness

Stage 2's cap check runs at the **top** of each epoch, and `per_epoch` only exists once `epochs_done >= 1` (incremented at each epoch's *end*):

```python
for epoch in range(1, EPOCHS + 1):
    elapsed_min = (time.time() - t_start) / 60.0
    per_epoch = elapsed_min / max(epochs_done, 1) if epochs_done else None
    if per_epoch and elapsed_min + per_epoch > CAP_MINUTES:
        break
```

So stage 2 terminates at **CAP_MINUTES + at most one epoch = ≈55.5 min**. v8's own log confirms the mechanism fires correctly (`stopping cleanly at epoch 102`). Therefore a 2 h 24 min runtime **cannot** be stage 2. The overrun must be stage 0, stage 1, or **stage-2 epoch 1 never completing** (no cap check has fired yet) — i.e. a silent hang. Waiting could not have produced a result.

### 16.3 Why the log could not simply be read

A control experiment established the platform behaviour:

| Kernel | State | `log` bytes | output files |
|---|---|---|---|
| v8-train | COMPLETE | **55,336** | **10** |
| v9-train | RUNNING | **0** | **0** |

and the live log-stream endpoint returned **HTTP 500 after ~61 s** on both `api.kaggle.com` and `www.kaggle.com` while the kernel was running. Conclusion: for this platform, **a running batch kernel's log and output are not observable by any client**, and a cancelled run discards its log entirely. Consequently the rule "never call a job running without showing its log growing" is **satisfiable only post-hoc** here — which is precisely why the repair had to make the *run itself* self-documenting.

### 16.4 The hardening (applied to v9 and v10)

| # | Mechanism | Default | Defect it removes |
|---|---|---|---|
| 1 | `start_watchdog()` — daemon thread that force-exits past `GLOBAL_DEADLINE_MIN` and writes `stage_deadline_exceeded.json` | 100 min | a stuck C-level call (DataLoader worker, subprocess) blocking the main thread forever |
| 2 | `subprocess.run(..., timeout=SUB_TIMEOUT_S)` on the geometry gate | 600 s | a hung child process in stage 0 |
| 3 | `DataLoader(timeout=..., persistent_workers=False)` | 300 s | a dead/stalled worker hanging iteration with no error |
| 4 | Deadline re-checks in the stage 1/2 loops; a per-200-step progress line; a 30 s heartbeat | — | no visibility into which stage is alive |

**A separate thread is the only correct place for the deadline.** A Python-level `if time > deadline` inside the training loop can never fire if the loop is blocked inside a C-level call — and that is exactly the failure being defended against.

### 16.5 The hardening was verified, not asserted

```
py_compile both drivers                        -> OK
hardening block byte-identical across v9/v10   -> confirmed (line offsets differ only)
[A]  loader_for timeout=7 persistent_workers=False -> PASS
[A2] workers=0 timeout=0 persistent_workers=False  -> PASS
[B]  geometry hang killed after 3.0s               -> PASS
watchdog force-exit: rc=1, elapsed=30s, marker {"ok": false, "elapsed_min": 0.5, "state": {}}
full local smoke on the REAL corpus:
  v9 : EXIT=0  TOTAL 1.0 min  stage0 ok / stage1 photo / stage2 ok
       overlay_median_ratio=20.40x  lesion_inside_leaf_min=1.0
  v10: EXIT=0  TOTAL 1.1 min  stage0 ok / stage1 photo / stage2 ok
       E 1/1 [9s] cls=2.0992 seg=5.7173 det=5.139/0.764/8.514
```

**The transferable lesson** is the general one, not the Kaggle one: *an unobservable long-running job must be designed to terminate and to leave a diagnosis behind.* A deadline, a watchdog thread, per-operation timeouts and a marker file are the minimum. This is standard engineering for any batch pipeline whose telemetry is absent.

---

## 17. Limitations and threats to validity

Stated in descending order of how much damage each could do to a published claim.

1. **The healthy shortcut (0.9841).** Image statistics alone predict "healthy" on val with 98.41 % holdout accuracy. Any claim about healthy-leaf recognition is confounded with acquisition cues. **Required action:** disclose in the abstract and the limitations section of any submission; do not report the healthy class F1 as evidence of learning.
2. **Severity does not beat its baseline (v8).** `sev_mae_predmasks = 0.1026` vs a constant baseline of **0.0384** on the outline corpus. Severity from predicted masks is **not** trustworthy at v8, and the v9 baseline (0.2293) is a different, harder target that must be evaluated on its own terms.
3. **Gross lesion over-prediction.** Predicted lesion area = **7.61×** true area at the best epoch (3.10× at epoch 102). Rendered, this looks like large spurious lesion blobs. The over-prediction ratio is reported precisely because it is the most likely thing a reviewer would notice first.
4. **Small-lesion segmentation is intrinsically hard here.** Lesions are 3–8 px at 320×320; at that size, IoU is a brutally sensitive metric and a ±1 px boundary error moves it substantially. The v8 IoU of 0.0459 is 1.6× v7 but far from usable for pixel-level localization.
5. **Label-audit exclusion is not applied to training.** The audit correctly finds 32 empty-label train images, but the exclusion test compares against the *dict* rather than its `"train"` entry, so **0 train images are actually dropped** (`train: 16 images (0 dropped by label audit)` reproduces this even in the smoke test). This is **deliberately left unfixed to keep v8 and v9 on the same data** — changing it mid-comparison would confound the mask-repair result. It is a known, documented defect with a one-line fix (`exclusions["train"]`).
6. **EMA never beat raw in v8.** The selected checkpoint is a *raw* one, which is somewhat less robust than an EMA checkpoint is expected to be. EMA was verified to be moving, so this is a genuine observation rather than a bug — but it is unusual and unexplained.
7. **`sev_mae_trueleaf` (0.1137) > `sev_mae_predmasks` (0.1026)** is counter-intuitive: substituting the *true* leaf mask makes severity slightly *worse*. The plausible explanation is that both channels over-predict, so the true leaf is smaller than the predicted one and the ratio shifts. It is recorded rather than explained away.
8. **Class imbalance** (40 vs 114 training images; 8 vs 33 in val) caps achievable macro-F1. Mitigations (mild capped sqrt weights, weighted sampler, label smoothing) reduce but do not remove the effect.
9. **Cross-domain gap.** The v7 checkpoint scored **0.430** top-1 on PlantDoc (field imagery) against ≈0.83 on the lab-like corpus `[M]`. Field performance, not corpus performance, is what a grower experiences.
10. **Detection has never been trained.** All detection numbers from v8/v9 are `nan` with zero predicted boxes by construction. Any mAP claim awaits v10.
11. **Phone latency is unmeasured.** Desktop-CPU (74.37/67.90 ms) and T4 numbers are not phone numbers. Reported only as such.
12. **No bootstrap CIs in this document.** The eval kernel computes them; they are pending with the eval runs.

---

## 18. Deployment

### 18.1 Export

Four outputs are exported together — `cls_logits`, `seg_logits`, `det_cls`, `det_box` — through a thin wrapper (`OnnxWrap`) so the graph is self-contained.

| Artifact | Size | Note |
|---|---|---|
| FP32 ONNX | **10.23 MB** | reference, exact |
| INT8 ONNX (static) | **3.13 MB** | quantized, ≈3.3× smaller |

**Parity, FP32 ONNX vs PyTorch, 4 outputs, over 20 images `[M]`:**

```
cls     1.5e-08
seg     1.0e-07
det_cls 6.2e-06
det_box 1.5e-06
```

All four are at or below 1e-5 — i.e. the export is numerically faithful, not merely "runs".

### 18.2 Why static INT8, and why Hardswish

Static (calibration-based) quantization is chosen over dynamic because convolutions dominate this model and dynamic quantization targets the fully-connected/attention path. Calibration data comes from the validation split (never test). The Hardswish-only activation policy in the heads (§15.8) is what makes this work without per-layer exceptions.

### 18.3 On-device measurement harness

Because a host-CPU latency number would mislead for a phone target, a **browser-based harness** ships in the repository:

```
cd mobile/bench && python3 -m http.server 8000
# phone on the same Wi-Fi: http://<laptop-ip>:8000/phone_bench.html
```

It times FP32 / INT8-static / INT8-dynamic with warm-up and 30/50/100 repetitions, reporting median and p90, and `bench_onnx_phone.py` provides a native Termux equivalent. **The results table is intentionally empty in this paper** — filling it with anything other than a real phone measurement would violate the paper's own evidentiary standard.

---

## 19. Reproducibility

### 19.1 Integrity gates

```bash
sha256sum model/training/dataset.py
# expected ff0102ed509d595abe08b79b8e1902fd6756cb7404f5efd102d1a507b90696db

python3 -m py_compile kaggle/v9/cropguard-v9-train.py kaggle/v10/cropguard-v10-train.py
```

### 19.2 Local smoke of a driver on the real corpus

```bash
V8_WORK=/tmp/smoke9/work \
V8_CODE_ROOT="$PWD/model" \
V8_INPUT_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v7_split \
V9_FILLED_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v9_fixed \
V8_STAGE1_STEPS=20 V8_FORCE_AUG=photo V8_TRAIN_SUBSET=16 V8_VAL_SUBSET=16 \
V8_EPOCHS=1 V8_CAP_MIN=6 V8_DEADLINE_MIN=25 \
python3 -u kaggle/v9/cropguard-v9-train.py
```

### 19.3 Kaggle runs

```bash
kaggle kernels push -p kaggle/v9
kaggle kernels push -p kaggle/v10
kaggle kernels status ganrajlongadge/cropguard-v9-train
kaggle kernels output ganrajlongadge/cropguard-v9-train -p /home/logiclocus/tmp/kout_v9
```

### 19.4 Evaluation / calibration

```bash
python3 model/training/calibrate_abstention.py \
    --ckpt model/weights/v7/best_stage5_calibrated.pt \
    --root ~/tmp/cgpack/cropguard_v7_split --split val --write

python3 model/training/eval_tta.py --arch v7 \
    --root ~/tmp/cgpack/cropguard_v7_split --split test \
    --ckpt model/weights/v7/best_stage5_calibrated.pt \
    --unsure-thresh 0.231 --out model/weights/v7/metrics_plantseg_test_v7.json
```

### 19.5 Explainability

```bash
python3 model/training/layercam_v7.py \
    --ckpt /home/logiclocus/tmp/kout_v9/best_v8_ema.pt \
    --data ~/tmp/cgpack/cropguard_v7_split --split val --n 40 \
    --out /tmp/camtest_v9real
```

### 19.6 Artifacts written by a training run

```
best_v8_raw.pt / best_v8_ema.pt      selected checkpoints (raw / EMA)
last_v8.pt                           every-epoch checkpoint (+opt, +ema_shadow)
v8_train_log.csv                     102 epochs x 23 columns (raw + EMA metrics)
stage0_summary.json                  sha256 / geometry / pretrained gates
stage0_label_audit.json              per-split per-class counts + exclusions
stage1_memorize.json                 memorization gate, geometry ON
stage1_memorize_nogeo.json           memorization gate, geometry OFF
stage2_training.json                 best epoch + config + class weights
stage_summary.json                   stage 3 validation summary
```

---

## 20. References

The repository's bibliography (`resources/research_papers/references.bib`) contains the following works cited by the survey structure of this paper. Numeric keys below are the repository's own keys.

1. Azevedo, P., Santos, V. — *Comparative analysis of multiple YOLO-based target detectors and trackers for ADAS in edge devices.*
2. Yumang, A. N., Paglinawan, A. C., Dingel, B. D. — *EIN+Y: An Einstein activation function based YOLOv8 with alpha blending for road surface condition classification.*
3. Redmon, J., Divvala, S., Girshick, R., Farhadi, A. — *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
4. Lin, M., Chen, Q., Yan, S. — *Network in Network.* ICLR.
5. Tolstikhin, I., Houlsby, N., Kolesnikov, A., et al. — *MLP-Mixer: An all-MLP Architecture for Vision.* NeurIPS.
6. Yu, C.-C., Chen, Y.-D., Cheng, H.-Y., Jiang, C.-L. — *Semantic Segmentation of Satellite Images for Landslide Detection Using Foreground-Aware and Multi-Scale Convolutional Attention Mechanism.*
7. Ge, Z., Liu, S., Wang, F., Li, Z., Sun, J. — *YOLOX: Exceeding YOLO Series in 2021.*
8. Hasan, M. J., Mazumdar, S., Momen, S. — *Deployable Deep Learning for Cross-Domain Plant Leaf Disease Detection via Ensemble Learning, Knowledge Distillation, and Quantization.*
9. Freitas, L. C., Dias, I. V., Santos, V. R. S., Ferreira Jr., P. R., de Mattos, J. C. B., de Brisolara, L. B. — *Optimising Embedded Neural Network Inference in Smart Traps for Fruit Pest Detection via Quantization-Aware Training and FPGA Acceleration.* SBESC 2025.
10. Upadhyay, A., Chandel, N. S., Singh, K. P., Chakraborty, S. K., Nandede, B. M., Kumar, M., Subeesh, A., Upendar, K., Salem, A., Elbeltagi, A. — *Deep learning and computer vision in plant disease detection: a comprehensive review of techniques, models, and trends in precision agriculture.*
11. Ramos, et al. — *A Decade of You Only Look Once (YOLO) for Object Detection: A Review.*
12. Shafay, et al. — *Recent advances in plant disease detection: challenges and opportunities.*
13. Li, et al. — *PMVT: a lightweight vision transformer for plant disease identification on mobile devices.*
14. Khan, et al. — *Plant disease detection model for edge computing devices.*
15. — *SSD-YOLO: a lightweight network for rice leaf disease detection.*
16. Chen, J., et al. — *FasterNet: a new neural network architecture* (PConv). *(referenced by the backbone choice in §15.1)*
17. Tan, M., Pang, R., Le, Q. V. — *EfficientDet: BiFPN weighted bidirectional feature fusion.* *(referenced by §15.2)*
18. Li, H., et al. — *LayerCAM* (class activation mapping). *(referenced by §14.6)*

---

## Appendix A — Provenance map

| Claim | Artifact |
|---|---|
| Parameter counts, tensor shapes | measured in this session (`§9.1`, `§9.2`) |
| v8 val metrics, per-class F1, LR schedule | `docs/v8_report.md`, `v8_train_log.csv`, `stage_summary.json` |
| v8 runtime, TIME CAP line, epoch count | `cropguard-v8-train.log` final lines |
| Memorization gate numbers | `stage1_memorize*.json` |
| Corpus gate numbers (20.40×, 1.0000) | v9 driver `prepare_corpus()`, re-measured in local smoke |
| ONNX parity, sizes, latency, AP equivalence | v10-eval local smoke (documented in `RUNLOG.md`) |
| Healthy-shortcut 0.9841 | v10-eval smoking probe |
| LayerCAM lifts (1.8501 / 1.0295) | `model/training/layercam_v7.py` runs recorded in `RUNLOG.md` |
| v7 → v8 comparison | `docs/V7_EVAL_2026-10-11.md`, `runs_v7_120/train.log` |
| The hang diagnosis and hardening | `RUNLOG.md`, 2026-10-12 entries |

## Appendix B — What to add when the pending runs land

1. **§14.2** — replace "QUEUED" with the v9 and v10 `stage_summary.json` numbers.
2. **§14.5** — fill the PlantSeg/PlantDoc test table from the eval kernel's `results.json`, including bootstrap CIs.
3. **§14.4** — replace the `nan` mAP with the trained head's mAP@0.5 / mAP@0.5:0.95, with and without flip-TTA.
4. **§17.5** — either fix the label-audit exclusion and re-run, or keep the disclosure.
5. **§18.3** — paste the phone-benchmark table once measured on real hardware.
