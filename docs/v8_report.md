# CropGuard v8 — Results Report

**Source of truth:** Kaggle kernel `ganrajlongadge/cropguard-v8-train`, status **COMPLETE** (last run 2026-10-11 19:06 UTC). We downloaded the **train output** to `/tmp/kaggle_out_train` and are reporting from that. The **eval kernel** (`ganrajlongadge/cropguard-v8-eval`) output was **not available** from this host (download denied: `Permission 'kernels.get' was denied` — the eval kernel either does not exist under this account or is private). So **test metrics are `not_measured`** unless the eval kernel is re-run. We did **not** run the eval kernel here (no GPU; would need the Kaggle CLI + datasets mounted). We did **not** fit or select on any test split.

**What we have:**
- `best_v8_raw.pt` — best checkpoint by VAL macro-F1 (epoch 35, raw). 10.3 MB.
- `last_v8.pt` — saved every epoch (model + ema_shadow + optimizer). 40.8 MB.
- `v8_train_log.csv` — 102 epochs of train+val metrics (RAW + EMA). 16.5 KB.
- `stage_summary.json` — VAL metrics of the selected checkpoint (best_v8_raw.pt epoch 35).
- `stage2_training.json` — best-epoch record + training config.
- `stage0_summary.json` — sha256/geometry/pretrained checks.
- `stage0_label_audit.json` — per-split per-class counts + exclusions.
- `stage1_memorize.json`, `stage1_memorize_nogeo.json` — 8-image memorization gate.

**What we do NOT have (not_measured unless eval re-run):**
- PlantSeg **test** metrics (acc, macro-F1, ECE, confusion matrix, per-class F1, lesion IoU/Dice on test).
- PlantDoc **test** metrics (79 images).
- severity MAE (const baseline) on test, severity Spearman on test.
- over-prediction ratio on **test** (we have it on VAL only).
- contact sheet of 30 test images.
- per-image class counts per split (we have per-class **aggregate** counts from the label audit, not per-image).
- contact sheet of 24 healthy vs 24 diseased train images.
- logistic-regression healthy-shortcut accuracy (requires running the eval kernel's `healthy_shortcut` function on val).

---

## 1) What changed in v8

**v7 failure (cause found):** segmentation did not learn. In the v7 120-epoch log (`train_full.log` / `runs_v7_120/train.log`) the lesion seg loss stayed **~3.6** for the entire run (e.g. epoch 90 `seg=3.6495`, epoch 120 `seg=3.6311`) while classification settled (epoch 90 `cls=0.3081`, epoch 120 `cls=0.3072`). Result: epoch 120 val = `acc=0.6755 macro_f1=0.6433 lesion_iou=0.0282 (thr=0.5) dice=0.0549 sev_mae=0.3104 (vs base 0.0384)`. The seg loss was not just high — it was **flat**, i.e. the seg head was not receiving useful gradient signal.

Root cause (code evidence, this repo's actual files):

- **v7 seg head = single-channel `BilinearDecoder` driven by pseudo-masks** (bbox rectangles), **not real lesion masks**. v7 model: `model/architecture/v7_model.py` L49 L53 L118 — `self.seg_head = BilinearDecoder(width, skip_ch=self.backbone.channels[0])` and `seg_logits = self.seg_head(p3, p4, p5, c2, out_size=...)` with a **single seg channel**.
- v7 dataset `__getitem__` (`model/training/dataset.py` read in v7 context) built the seg target as `make_pseudo_mask(boxes, size)` = bbox rectangles — so the seg head was trained on rectangles, not on real lesion pixels. Real lesion masks did not exist for the corpus at v7 time (the corpus masks were filled **leaf** bounding boxes, binary {0,255}; the "128 = lesion" value never occurs — see `dataset.py` lines 311-328 comment and `HANDOFF.md` L84-85, `README.md` L24).
- v7 loss path: v7 training loop (`model/training/train_full_v7.py` L278-284) called `model(..., active_tasks={"cls","seg"}, gt_mask=batch.get("mask"), gt_mask_prob=...)` and the model's forward used `two_channel_seg` flag only in some paths; the seg loss was a single-channel BCE/dice on the pseudo-mask — so **the lesion channel had no real lesion supervision**.
- v7 architecture had **no stride-4 C2 skip** for seg (only C3/C4/C5 @ /8, /16, /32), so tiny lesions (3-8 px at 320) were already degraded before the decoder.

**What v8 changed (code files + line ranges), and why:**

| # | File | Change | Why |
|---|---|---|---|
| 1 | `model/architecture/v7_model.py` | Added a **2-channel** seg output: `BilinearDecoder(width, skip_ch=self.backbone.channels[0])` now produces 2 channels (lesion, leaf) — L53, L60-61 (`_seg_channels`), and the forward writes both `seg_logits[:, 0:1]` (lesion) and `[:, 1:2]` (leaf) — see L118-125, L137-144 (`analytic_severity` now uses both channels: `sev = sum(sig(lesion_logit)) / (sum(sig(leaf_logit)) + eps)`). | Two channels = lesion channel can be supervised by real lesion masks, leaf channel by real leaf masks; severity is analytic from the two predicted masks instead of a learned head. |
| 2 | `model/architecture/backbone.py` | v7 backbone levels changed to `_LEVEL_INDICES_V7 = (0,1,2,3)`, `_STRIDES_V7 = (4,8,16,32)` — adds the **stride-4 C2 (40ch)** skip for the seg decoder — L11-16, L21-22 (`Backbone(pretrained, levels=_LEVEL_INDICES_V7)`). | Higher-resolution skip lets the seg decoder see small lesions (3-8 px) — without it they are lost in the /8 stride. |
| 3 | `model/training/losses.py` | New `seg_loss_v7(seg_logits, lesion_mask, leaf_mask)` — L227-266: lesion channel = BCE(pos_weight=20) + 3×B Focal-Tversky(alpha=0.3, beta=0.7, gamma=1.33); leaf channel = BCE + Dice with -1.0 exclusion sentinel. | Focal-Tversky with beta=0.7 heavily penalizes false negatives on tiny lesion targets; BCE pos_weight=20 up-weights the rare lesion pixels; leaf channel gets its own Dice+BCE so the leaf mask is learned from real leaf masks; -1.0 exclusion skips bad leaf-mask images from the leaf loss only. |
| 4 | `model/training/dataset.py` | `__getitem__` now returns **real lesion mask** (`mask` key = `(raw_lesion > 127)` when `lesion_masks_dir` is present — L298-304) and **real leaf mask** (`leaf` key = `(raw_leaf > 127)` or from combined-mask leaf region — L284-294), plus `sev_cont`/`sev_ord` from `severity_from_ratio(lesion_px/leaf_px)` — L304, L366-367. The `mask` key is the **lesion** mask (not a pseudo bbox). | This is the actual fix: the seg head now trains on **real lesion + real leaf masks** derived from the separate `masks_lesion/` and `masks_leaf/` dirs produced by `derive_dataset.py`. Severity target is now the lesion/leaf area ratio from real masks, not a constant. |
| 5 | `kaggle/v8/cropguard-v8-train.py` | Training loop calls `seg_loss_v7(out["seg_logits"], batch["mask"], batch["leaf"])` — L558 (stage1), L746 (stage2); optimizer asserts backbone+neck+seg_head are fully in it — L697-699; EMA inline class with checksum — L155-184; cosine LR — L193-197; `build_model` uses `CropGuardMultiTaskV7(nc=NC, pretrained=PRETRAINED_OK)` — L316-319; class weights + WeightedRandomSampler — L667-678, L371-383. | Stitch it all together: real 2-channel seg loss on real masks, seg head trainable, EMA proven to move, LR schedule, class balance. |

Why I think it fixes the v7 failure: v7's seg loss was flat at 3.6 because the lesion channel was trained on bbox pseudo-masks (rectangles) with no real lesion signal and the decoder had no high-res skip — so gradient had nothing lesion-shaped to latch onto. v8 gives the lesion channel **real lesion masks** (2-channel head, Focal-Tversky+BCE), a **stride-4 C2 skip** for the tiny lesions, and a **leaf channel** trained on real leaf masks. The proof is in the v8 numbers: seg loss drops from 4.8 (epoch 1) to 2.5 (epoch 102), and lesion IoU goes from 0.0074 (epoch 1) to **0.0459** (epoch 35 best) — vs v7's flat 0.028. That is a real (if still weak) seg signal, which v7 never had.

**v8 checkpoint(s), log, config:**
- Checkpoint: `/tmp/kaggle_out_train/best_v8_raw.pt` (best by VAL macro-F1, epoch 35, raw). `last_v8.pt` (epoch 102, has ema_shadow). `best_v8_ema.pt` was **not** saved (EMA never beat raw on VAL).
- Training log: `/tmp/kaggle_out_train/v8_train_log.csv` (102 epochs) + `cropguard-v8-train.log` (full kernel log).
- Config: epochs=120 (ran 102, stopped by TIME CAP at 55 min), batch=8, LR=5e-4 base (cosine, 5-epoch warmup), SEED=1337, SIZE=320, NC=8, weight_decay=1e-4, EMA decay=0.999, image=320, seg_loss=seg_loss_v7 (lesion: BCE+3×FocalTversky; leaf: BCE+Dice).

---

## 2) Sanity checks

**Pretrained fasternet_t0 weights loaded?** YES — from `stage0_summary.json`:

```json
{
    "sha256_ok": true,
    "geometry_ok": true,
    "pretrained_mode": "pretrained",
    "pretrained_l1": 0.23836876451969147
}
```

Raw output of the check (from `kaggle/v8/cropguard-v8-train.py` L323-345 `check_pretrained()`):

```python
ref = timm.create_model("fasternet_t0", pretrained=False,
                        features_only=True, out_indices=(0, 1, 2, 3))
a = next(m.backbone.net.parameters()).detach().float().cpu()
b = next(ref.parameters()).detach().float().cpu()
l1 = float((a - b).abs().mean().item())
log(f"pretrained check: mode={mode} first-conv L1 vs random init = {l1:.6f} "
    f"({'>' if l1 > 0.01 else '<='} 0.01)")
```

Result: **L1 = 0.2384 > 0.01 threshold ⇒ NOT random init.** The pretrained fasternet_t0 weights were downloaded and loaded (mode=pretrained). If it had been random, L1 would be ≤ 0.01 and the script logs a LOUD WARNING.

**Exact code line + metric that selects the "best" checkpoint.** From `kaggle/v8/cropguard-v8-train.py` L797-812:

```python
for which, m in (("raw", ra), ("ema", em)):
    score, tie = m["macro_f1"], m["lesion_iou"]
    if score > best["score"] or (score == best["score"] and tie > best["iou"]):
        best = {"score": score, "iou": tie, "epoch": epoch,
                "which": which, "metrics": m}
        if which == "raw":
            torch.save({...}, WORK / "best_v8_raw.pt")
        else:
            torch.save({...}, WORK / "best_v8_ema.pt")
```

**Selection metric = val macro-F1**, tie-broken by **val lesion IoU**. This is val-only (no test data touched). The brief asked: "Selection must use val macro-F1 only, unless segmentation now works." Segmentation does now produce a non-zero IoU (0.0459), so the tie-break is active, but the **primary selector is still val macro-F1**. The best checkpoint is **epoch 35, raw** (`best_v8_raw.pt`), macro_f1=0.6786, lesion_iou=0.0459.

**Which weights produce the reported numbers: raw or EMA? Is EMA updating?** The reported VAL numbers come from the **raw** checkpoint (`best_v8_raw.pt`, epoch 35, `which="raw"`). EMA was **not** saved as a best checkpoint (best_v8_ema.pt absent). EMA **is** updating: the inline EMA class (`kaggle/v8/cropguard-v8-train.py` L155-184) tracks `ema_checksum` every epoch, and the checksum **moves every epoch** (epoch 1: 3.469940, epoch 35: 1.316943, epoch 60: 1.851464, epoch 90: 2.186074, epoch 102: 2.184402 — monotonically varying ⇒ EMA shadow actually changed; if it had stalled the script logs a WARNING at L773-776). So EMA is live and moving; it just didn't beat raw on val macro-F1, so best_v8_ema.pt was never written.

**Learning rate at epochs 1/30/60/90/final + seg head in optimizer.** From `v8_train_log.csv` (every epoch has LR in column 2):

```
epoch 1  lr=9.889e-05   (warmup step, ~0.2× base 5e-4)
epoch 30 lr=4.441e-04   (~0.89× base)
epoch 60 lr=2.676e-04   (~0.54× base)
epoch 90 lr=8.026e-05   (~0.16× base)
epoch 102 lr=3.060e-05  (final, ~0.06× base — cosine near zero)
```

LR schedule: cosine decay from 5e-4 with a 5-epoch (5×steps_per_epoch) warmup, min_lr=1e-6. `cosine_lr(step, total_steps, warmup, base_lr, min_lr)` at L193-197.

**Seg head in optimizer: YES.** `kaggle/v8/cropguard-v8-train.py` L690-699 asserts backbone, neck, and **seg_head** are fully represented in the optimizer param groups:

```python
for name, mod in (("backbone", model.backbone), ("neck", model.neck),
                  ("seg_head", model.seg_head), ("cls_head", model.cls_head),
                  ("det_head", model.det_head)):
    inside = sum(1 for p in mod.parameters() if id(p) in ids)
    n = sum(1 for _ in mod.parameters())
    if name in ("backbone", "neck", "seg_head") and inside != n:
        raise RuntimeError(f"{name} is NOT fully in the optimizer ({inside}/{n})")
log(f"optimizer coverage: {' '.join(opt_names)}  (asserted backbone/neck/seg_head)")
```

Stage1 also prints per-component gradient norms including `seg_head` every 100 steps (stage1_memorize.json shows `gnorms.seg_head` from 0.76 at step 100 up to 3.86 at step 500-withmasks), confirming the seg head receives gradient.

**DataLoader mask alignment check (8 training batches, all transforms, augmentation OFF, contact sheet).** The requested contact sheet of 8 training batches with overlays — **was not produced** in the train output and we did not run it here (it's an eval-side visual check, and we have no GPU to run inference here; the train output only has the numeric gates). What **is** verified in the code:

- `stage0 check_geometry` (`kaggle/v8/cropguard-v8-train.py` L290-317) runs the geometry gate on the Kaggle input: it calls a separate test script that checks image + both masks agree within 2 px after the shared transform. Result: `geometry_ok: true` in `stage0_summary.json`.
- The dataset's `_augment` (`model/training/dataset.py` L126-170) applies **one shared geometric transform** to `img`, `boxes`, `lesion_mask`, `leaf_mask`, `combined_mask` together (image bilinear, masks nearest) — so masks and image stay aligned by construction. L163-164 comment: "ONE geometry path for every mask: the SAME flip and scale the image just received (image bilinear, masks nearest)."
- Stage1 runs both with geometric aug (`stage1_memorize.json`) and without (`stage1_memorize_nogeo.json`) and reports IoU every 100 steps — a functional check that masks participate in training.

We did **not** render a contact sheet of 8 batches; the code-level alignment guarantees are what we have. To produce the visual contact sheet you would run the eval kernel (or a small inference script) on 8 train images with augmentation off and overlay predicted lesion/leaf on the image — that needs the eval kernel + datasets mounted, which we don't have here.

**8-image memorization test (500 steps, IoU/Dice every 100 steps at best threshold).** We **do** have this in the train output — `stage1_memorize.json` (with geometric aug) and `stage1_memorize_nogeo.json` (photometric only). Raw output:

```json
// stage1_memorize.json (WITH geometric augmentation, 8 images, 500 steps)
[
  {"step": 100, "iou": 0.0364, "cls": 0.2649, "seg": 3.8216, "total": 4.0864,
   "gnorms": {"seg_head": 0.7576, "neck": 0.1305, "backbone": 0.1464}},
  {"step": 200, "iou": 0.0762, "cls": 0.2650, "seg": 3.2127, "total": 3.4777,
   "gnorms": {"seg_head": 0.7299, "neck": 0.1882, "backbone": 0.1930}},
  {"step": 300, "iou": 0.1074, "cls": 0.2649, "seg": 2.7553, "total": 3.0201,
   "gnorms": {"seg_head": 0.8763, "neck": 0.3640, "backbone": 0.2889}},
  {"step": 400, "iou": 0.1629, "cls": 0.2650, "seg": 2.3184, "total": 2.5833,
   "gnorms": {"seg_head": 1.5883, "neck": 0.7709, "backbone": 0.3952}},
  {"step": 500, "iou": 0.1935, "cls": 0.2651, "seg": 2.0227, "total": 2.2878,
   "gnorms": {"seg_head": 1.3857, "neck": 0.7068, "backbone": 0.5419}}
]
```

```json
// stage1_memorize_nogeo.json (photometric-only, no geometric aug)
[
  {"step": 100, "iou": 0.0693, "cls": 0.2648, "seg": 3.6612, "total": 3.9260,
   "gnorms": {"seg_head": 0.8664, "neck": 0.0823, "backbone": 0.0944}},
  {"step": 200, "iou": 0.1556, "cls": 0.2648, "seg": 2.9065, "total": 3.1714,
   "gnorms": {"seg_head": 1.3491, "neck": 0.2020, "backbone": 0.1033}},
  {"step": 300, "iou": 0.3275, "cls": 0.2649, "seg": 1.9573, "total": 2.2222,
   "gnorms": {"seg_head": 3.5208, "neck": 0.5380, "backbone": 0.3834}},
  {"step": 400, "iou": 0.5832, "cls": 0.2649, "seg": 1.0939, "total": 1.3588,
   "gnorms": {"seg_head": 5.7019, "neck": 0.7534, "backbone": 0.4378}},
  {"step": 500, "iou": 0.6708, "cls": 0.2649, "seg": 0.4969, "total": 0.7618,
   "gnorms": {"seg_head": 3.8588, "neck": 0.4970, "backbone": 0.3256}}
]
```

Result: memorization gate **passes**. Without geometric augmentation (the easier case — the model can overfit 8 images), IoU reaches **0.671 at step 500** (> 0.5 threshold). With geometric augmentation (the realistic case — flips + scale jitter), IoU reaches **0.193 at step 500** — the model memorizes less because the augmentation forces invariance, but it still learns (IoU 0 → 0.19). The gate's purpose is to confirm the model + loss + data pipeline can actually overfit a tiny set (proving no dead pipeline), which it does. The "best threshold" for IoU in stage1 is the same `best_thr` grid search used in `evaluate()` (thresholds 0.3,0.4,0.5,0.6,0.7; the max-IoU threshold is recorded as `best_thr` in the metrics dict). Stage1 does not store the per-threshold grid in the JSON (it stores the max-IoU value and the best threshold is implicit), but the `evaluate()` function (L408-466) does the full grid and reports `best_thr=0.7` for the best checkpoint.

---

## 3) Training log

**Last 20 lines of the log.** The full kernel log is `cropguard-v8-train.log` in the train output; here are the **last 20 lines of `v8_train_log.csv`** (the per-epoch CSV, which is the structured log). The CSV is the authoritative per-epoch log (the kernel log repeats these lines plus stage messages).

Raw output (`v8_train_log.csv` lines 83-103, i.e. epochs 82-102):

```
epoch,lr,cls,seg,total,sec,raw_acc,raw_f1,raw_iou,raw_dice,raw_sevMAE_pred,raw_sevMAE_trueleaf,raw_pred_area,raw_true_area,ema_acc,ema_f1,ema_iou,ema_dice,ema_sevMAE_pred,ema_sevMAE_trueleaf,ema_pred_area,ema_true_area,ema_checksum
82,1.239e-04,0.2860,2.6093,2.8953,22.0,0.6291,0.5884,0.0432,0.0828,0.0448,0.0475,2137.3,621.3,0.6490,0.6078,0.0450,0.0860,0.0595,0.0605,2634.3,621.3,2.113906
83,1.180e-04,0.2961,2.6115,2.9076,23.1,0.6556,0.6106,0.0424,0.0814,0.0485,0.0487,2229.0,621.3,0.6490,0.6078,0.0448,0.0858,0.0590,0.0595,2601.0,621.3,2.110166
84,1.123e-04,0.2936,2.6096,2.9032,20.7,0.6291,0.5722,0.0422,0.0810,0.0348,0.0463,2141.1,621.3,0.6490,0.6078,0.0446,0.0854,0.0580,0.0588,2576.3,621.3,2.131124
85,1.067e-04,0.2946,2.5943,2.8889,25.5,0.6225,0.5847,0.0410,0.0788,0.0445,0.0481,2151.5,621.3,0.6424,0.6029,0.0444,0.0851,0.0574,0.0582,2551.6,621.3,2.146453
86,1.012e-04,0.2834,2.5811,2.8645,20.5,0.6291,0.5739,0.0420,0.0806,0.0495,0.0501,2141.9,621.3,0.6358,0.5992,0.0442,0.0847,0.0573,0.0579,2533.2,621.3,2.163846
87,9.576e-05,0.2945,2.5511,2.8456,26.3,0.6556,0.6033,0.0425,0.0816,0.0501,0.0540,2404.7,621.3,0.6358,0.5994,0.0441,0.0845,0.0568,0.0575,2514.4,621.3,2.162021
88,9.047e-05,0.2764,2.5732,2.8496,26.8,0.6821,0.6213,0.0426,0.0817,0.0508,0.0522,2161.3,621.3,0.6490,0.6246,0.0440,0.0843,0.0561,0.0568,2489.0,621.3,2.174365
89,8.530e-05,0.2918,2.5546,2.8464,27.9,0.6291,0.6144,0.0412,0.0792,0.0459,0.0511,2260.6,621.3,0.6424,0.6209,0.0439,0.0840,0.0556,0.0562,2467.4,621.3,2.179540
90,8.026e-05,0.2710,2.5703,2.8413,24.1,0.6689,0.6339,0.0408,0.0784,0.0464,0.0479,2191.1,621.3,0.6424,0.6207,0.0437,0.0838,0.0549,0.0555,2438.2,621.3,2.186074
91,7.534e-05,0.2881,2.5345,2.8225,24.3,0.6424,0.5986,0.0415,0.0798,0.0448,0.0456,2044.5,621.3,0.6424,0.6207,0.0435,0.0834,0.0542,0.0549,2420.4,621.3,2.194583
92,7.055e-05,0.2929,2.5340,2.8268,23.9,0.6623,0.6160,0.0413,0.0793,0.0489,0.0452,2072.4,621.3,0.6424,0.6203,0.0434,0.0832,0.0536,0.0543,2402.2,621.3,2.207530
93,6.590e-05,0.2863,2.5615,2.8477,24.1,0.6755,0.6191,0.0418,0.0802,0.0421,0.0496,2206.1,621.3,0.6424,0.6203,0.0433,0.0830,0.0534,0.0537,2379.4,621.3,2.222084
94,6.138e-05,0.2793,2.6055,2.8849,22.6,0.6490,0.6204,0.0420,0.0806,0.0462,0.0478,2126.6,621.3,0.6424,0.6203,0.0432,0.0828,0.0529,0.0533,2364.9,621.3,2.220456
95,5.701e-05,0.3039,2.5356,2.8395,22.0,0.6490,0.6260,0.0411,0.0789,0.0476,0.0485,2147.1,621.3,0.6490,0.6241,0.0431,0.0827,0.0523,0.0524,2337.9,621.3,2.214601
96,5.278e-05,0.2920,2.4699,2.7619,23.3,0.6490,0.6160,0.0415,0.0797,0.0486,0.0510,2322.4,621.3,0.6490,0.6241,0.0430,0.0824,0.0518,0.0522,2324.8,621.3,2.213299
97,4.869e-05,0.3055,2.5124,2.8179,25.7,0.6424,0.5980,0.0413,0.0793,0.0440,0.0488,2165.1,621.3,0.6424,0.6190,0.0429,0.0822,0.0513,0.0516,2307.8,621.3,2.211406
98,4.476e-05,0.2933,2.5147,2.8080,21.7,0.6623,0.6314,0.0413,0.0794,0.0464,0.0502,2235.0,621.3,0.6424,0.6190,0.0428,0.0821,0.0507,0.0511,2289.4,621.3,2.208388
99,4.098e-05,0.2891,2.4957,2.7848,21.6,0.6689,0.6392,0.0412,0.0791,0.0436,0.0446,2025.4,621.3,0.6424,0.6179,0.0427,0.0819,0.0501,0.0507,2274.0,621.3,2.206345
100,3.736e-05,0.2815,2.5223,2.8039,24.5,0.6556,0.6104,0.0411,0.0790,0.0525,0.0512,2316.7,621.3,0.6424,0.6179,0.0426,0.0817,0.0497,0.0503,2262.8,621.3,2.199594
101,3.390e-05,0.2982,2.5352,2.8334,22.7,0.6623,0.6358,0.0412,0.0791,0.0412,0.0461,2037.8,621.3,0.6358,0.6128,0.0424,0.0814,0.0493,0.0500,2247.5,621.3,2.191422
102,3.060e-05,0.2792,2.5597,2.8390,23.9,0.6291,0.6056,0.0417,0.0800,0.0417,0.0439,1923.7,621.3,0.6424,0.6179,0.0424,0.0814,0.0491,0.0496,2229.3,621.3,2.184402
```

**Val line at the best epoch (epoch 35) and final epoch (102).** From `stage_summary.json` (the selected checkpoint = best_v8_raw.pt epoch 35) and `v8_train_log.csv` row 36 (epoch 35) and row 103 (epoch 102):

Best epoch (35) — from `best_v8_raw.pt` metrics dict + `v8_train_log.csv` row 36:

```
epoch 35 | lr=4.209e-04 | cls=0.3505 seg=3.1593 total=3.5098 | sec=17.7
  RAW: acc=0.6954 macro_f1=0.6786 lesion_iou=0.0459 lesion_dice=0.0878
        sev_mae_predmasks=0.1026 sev_mae_trueleaf=0.1137
        pred_area=4729.0 true_area=621.3 over_pred_ratio=7.61
        best_thr=0.7 (val-chosen threshold, max IoU at thr=0.7)
  EMA: acc=0.4636 macro_f1=0.3688 lesion_iou=0.0013 lesion_dice=0.0025
        sev_mae_predmasks=0.0363 sev_mae_trueleaf=0.0447
        pred_area=1726.3 true_area=621.3
  ema_checksum=1.316943  (EMA moved)
  --> new best (raw) epoch 35 f1=0.6786 iou=0.0459
```

Final epoch (102) — from `v8_train_log.csv` row 103:

```
epoch 102 | lr=3.060e-05 | cls=0.2792 seg=2.5597 total=2.8390 | sec=23.9
  RAW: acc=0.6291 macro_f1=0.6056 lesion_iou=0.0417 lesion_dice=0.0800
        sev_mae_predmasks=0.0417 sev_mae_trueleaf=0.0439
        pred_area=1923.7 true_area=621.3 over_pred_ratio=3.10
  EMA: acc=0.6424 macro_f1=0.6179 lesion_iou=0.0424 lesion_dice=0.0814
        sev_mae_predmasks=0.0491 sev_mae_trueleaf=0.0496
        pred_area=2229.3 true_area=621.3
  ema_checksum=2.184402
```

**Number of val images:** 156 total in the val split (from `stage0_label_audit.json`: val.images=156, val.label_files=156). After excluding 5 empty-label images, **151 valid-class val images** (val.per_class sum = 151; also `best_v8_raw.pt` metrics: n_valid_cls=151, n_sev=151). The training loop logs "val: 151 images (5 excluded by audit)" when run. So **val n = 151 valid** (156 total including 5 excluded).

Note: `true_area=621.3` is **constant across all 102 epochs** (mean GT lesion area on the val set, computed from the real lesion masks). It does not change — it's a property of the val split, not a per-epoch metric. The over-prediction ratio is `pred_area / 621.3`.

---

## 4) Test results

**These are `not_measured`** because the eval kernel output was not available from this host. We did not run the eval kernel (no GPU here; the eval kernel needs Kaggle datasets mounted). We also must not fit or select on test. So:

- PlantSeg test (150 images): **not_measured** (acc, macro-F1, ECE, per-class F1, confusion matrix, lesion IoU/Dice, severity MAE, Spearman, over-prediction ratio on test).
- PlantDoc test (79 images, 8 tomato folders): **not_measured**.
- Epoch-120 checkpoint: **not_measured** (and the run stopped at epoch 102, so epoch 120 doesn't exist; the final checkpoint is epoch 102 = `last_v8.pt`).

**What we can report instead (VAL only, from the best checkpoint — no test fitting):**

From `best_v8_raw.pt` metrics dict (epoch 35, raw, best by VAL macro-F1) — these are **validation** numbers, not test:

```json
{
  "acc": 0.6954,
  "macro_f1": 0.6786,
  "lesion_iou": 0.0459,
  "lesion_dice": 0.0878,
  "best_thr": 0.7,
  "sev_mae_predmasks": 0.1026,
  "sev_mae_trueleaf": 0.1137,
  "mean_pred_lesion_area": 4729.0,
  "mean_true_lesion_area": 621.3,
  "over_pred_ratio": 7.61,
  "n_valid_cls": 151,
  "n_sev": 151,
  "per_class": {
    "0 (healthy)": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 20},
    "1 (bacterial_spot)": {"precision": 0.818, "recall": 0.429, "f1": 0.562, "support": 21},
    "2 (early_blight)": {"precision": 0.76, "recall": 0.576, "f1": 0.655, "support": 33},
    "3 (late_blight)": {"precision": 0.765, "recall": 0.684, "f1": 0.722, "support": 19},
    "4 (leaf_mold)": {"precision": 0.696, "recall": 0.8, "f1": 0.744, "support": 20},
    "5 (mosaic_virus)": {"precision": 0.6, "recall": 0.375, "f1": 0.462, "support": 8},
    "6 (septoria_leaf_spot)": {"precision": 0.31, "recall": 0.643, "f1": 0.419, "support": 14},
    "7 (yellow_leaf_curl_virus)": {"precision": 0.762, "recall": 1.0, "f1": 0.865, "support": 16}
  }
}
```

These are **val** macro-F1=0.6786, lesion IoU=0.0459 (threshold 0.7, the val-chosen best threshold), severity MAE (pred masks)=0.1026, severity MAE (true leaf)=0.1137, over-prediction ratio=7.61.

**Severity with TRUE masks (separating seg error from severity logic):** `sev_mae_trueleaf=0.1137` is severity MAE computed using the **true leaf mask** as the denominator (lesion_pred_area / true_leaf_area), so it isolates the severity logic from segmentation error in the leaf channel. `sev_mae_predmasks=0.1026` uses the **predicted** leaf area. Both are well above the constant baseline (0.0384 = val median severity) — so severity is **not reliable** on val. (The const-baseline MAE on test can only come from the eval kernel.)

**Comparison vs v7 (val, same code path v7 used):** v7 epoch 120 val = macro_f1=0.6433, lesion_iou=0.0282, dice=0.0549, sev_mae=0.3104 (from `train_full.log`/`runs_v7_120/train.log`). v8 best val = macro_f1=0.6786, lesion_iou=0.0459, dice=0.0878, sev_mae_predmasks=0.1026. So:
- Classification: 0.6433 → 0.6786 (small improvement).
- Lesion IoU: 0.0282 → 0.0459 (1.6×, the seg fix is real but still weak).
- Severity MAE: 0.3104 → 0.1026 (big drop, because v8 uses analytic severity from real masks; v7's severity head learned nothing from pseudo-masks).

**Epoch-120 checkpoint:** Does not exist — training stopped at epoch 102 (TIME CAP at 55 min; configured for 120). The final checkpoint is epoch 102 (`last_v8.pt`), whose metrics: raw acc=0.6291, macro_f1=0.6056, lesion_iou=0.0417, dice=0.0800, sev_mae_predmasks=0.0417, sev_mae_trueleaf=0.0439, over_pred_ratio=3.10; EMA acc=0.6424, macro_f1=0.6179, lesion_iou=0.0424, dice=0.0814, over_pred_ratio=3.59 (from last_v8.pt ema_shadow at epoch 102).

---

## 5) Visual check

**Contact sheet of 30 random test images** — **not produced.** We did not run the eval kernel (no GPU; eval output not available), so we have no predicted lesion/leaf/severity maps on test images. To produce it you would run the eval kernel (which exports ONNX and can render overlays) or a small inference script on 30 test images — needs the eval kernel + datasets mounted.

**Honest assessment given what we have:** we cannot visually inspect test predictions here. From the numeric evidence:
- **Over-prediction is severe:** at the best epoch, predicted lesion area = 4729 px vs true 621.3 px (7.61×). At epoch 102 it's 1923.7 vs 621.3 (3.10×). So predicted lesion masks are large blobs covering far more than the true lesion — a contact sheet would show big predicted lesion regions on most images, many of them spurious. This is the single biggest visual red flag.
- **Severity is implausible:** predicted severity = lesion_pred_area/leaf_pred_area; with over-prediction of 7.6× on lesion and leaf masks both over-predicting, severity numbers will be noisy. sev_mae_predmasks=0.1026 on val (vs baseline 0.0384) confirms severity is not reliable.
- **Leaf mask:** we have a leaf channel trained on real leaf masks; the leaf mask over-predicts too (pred_area includes both channels). The leaf-inside-leaf check is "not_measured" without eval, but the leaf channel did get real-leaf supervision so it's more likely to be reasonable than the lesion channel.
- **Misalignment:** the geometry gate passed (`geometry_ok: true`), and the shared `_augment` transform keeps masks aligned with the image by construction, so misalignment of masks to image is unlikely to be the issue.

Bottom line: I expect a contact sheet would show **over-prediction** as the dominant failure (large predicted lesions), not blur or misalignment. I cannot confirm without rendering.

---

## 6) Pending data checks

**Per-image class counts per split (not box-line counts).** We have **per-class aggregate counts** from `stage0_label_audit.json` (these are per-image class labels, one majority class per image, not per-box). Raw:

```json
// train (715 images, 683 class instances, 32 excluded)
"per_class": {"0": 80, "1": 75, "2": 113, "3": 114, "4": 106, "5": 40, "6": 91, "7": 64}

// val (156 images, 151 class instances, 5 excluded)
"per_class": {"0": 20, "1": 21, "2": 33, "3": 19, "4": 20, "5": 8, "6": 14, "7": 16}

// test (150 images, 140 class instances, 10 excluded)
"per_class": {"0": 20, "1": 10, "2": 23, "3": 26, "4": 21, "5": 14, "6": 18, "7": 8}
```

These are **per-image** majority-class counts (one label per image), not box-line counts. The "sum_class_counts" field = total number of images that have a valid class in that split. Excluded images (empty label files) are listed in `excluded_examples`.

Note: these are **aggregate counts**, not a per-image dump. A true per-image count (every image's class listed) would come from iterating the dataset; the label audit gives the same info in aggregate form. We didn't render a per-image table because the eval kernel didn't run.

**Source of the 120 healthy images + contact sheet of 24 healthy vs 24 diseased train images + shortcut check.** 
- The 120 "healthy" images: from `stage0_label_audit.json`, train class 0 (healthy) = 80 images (after excluding 32). There is **no separate list of 120 healthy images** in the train output; the "120" number may refer to a previous corpus version. In this v8 run, train healthy = 80 images. (If the user means 120 healthy from an earlier corpus, that's not in this output.)
- Contact sheet of 24 healthy vs 24 diseased train images: **not produced** (requires rendering; we didn't run eval/inference here).
- Shortcut check (logistic regression on image size + mean RGB + background stats, healthy vs diseased, accuracy on val): **not run** here. The eval kernel has `healthy_shortcut()` (`kaggle/v8/cropguard-v8-eval.py` L411-443) that does exactly this on val images and reports `val_holdout_acc`. We did not run the eval kernel, so we don't have the number. The code is there; to get the number you run the eval kernel (which computes it on val and puts it in `results.json` under `healthy_shortcut`). If it's > 0.9 it flags that healthy may be decidable from image stats alone (a warning, not a leak — it's a val-only check on whether the dataset has a trivial healthy signal).

---

## 7) Honest summary

**What works:**
- Training completes end-to-end on Kaggle (status COMPLETE, 102 epochs, all stages pass: sha256/geometry gates ok, pretrained weights loaded, memorization gate passes, best checkpoint saved).
- Pretrained fasternet_t0 loads (L1=0.2384, not random).
- EMA updates and is tracked (checksum moves every epoch).
- Classification on val is decent: macro_f1=0.6786 (val), acc=0.6954, healthy/leaf_mold/late_blight/yellow_leaf_curl are strong (F1 0.72-1.0).
- The v7 seg failure is **fixed in the sense that seg now learns**: seg loss drops 4.8→2.5, lesion IoU 0.007→0.046, and the 2-channel real-mask pipeline is in place. The fix (real lesion+leaf masks, 2-channel head, stride-4 C2 skip, Focal-Tversky+BCE) is real and reflected in the numbers.

**What does not:**
- **Lesion segmentation is still weak:** lesion IoU=0.0459, dice=0.0878 on val. That's 1.6× better than v7's 0.028, but still far from usable for pixel-level lesion localization. Lesions are 3-8 px at 320 — at the edge of what the head can localize (this is noted as a limitation in the eval code).
- **Severity is not reliable:** sev_mae_predmasks=0.1026 on val vs 0.0384 baseline — severity (computed from predicted masks) does not beat the constant baseline. Even with true leaf masks it's 0.1137. So severity computed from predictions is not trustworthy.
- **Gross over-prediction:** predicted lesion area 7.61× the true area at the best epoch (3.10× at epoch 102). This is the biggest practical problem — the model predicts large lesion blobs that don't match the true lesion extent. On a contact sheet this would look like big false-positive lesion regions.
- **Test metrics are entirely not_measured** — we could not get the eval kernel output. So we have no test acc, macro-F1, ECE, confusion matrix, per-class test F1, PlantDoc results, severity Spearman, etc. The eval kernel needs to be re-run (and its output downloaded) to get these.

**Suspicious / unresolved:**
- Over-prediction ratio 7.61× at best epoch. Is this a threshold issue (best_thr=0.7 chosen for max IoU on val, but IoU is dominated by the large predicted blobs)? The threshold was chosen on val (max IoU), so it's not a test leak, but it may be that the model's lesion predictions are diffuse and thresholding can't recover the small true lesions. This is worth investigating (e.g. min-blob filtering, threshold sweep) — the post-hoc improver (`/tmp/prep2/posthoc.py`) does exactly a lesion-threshold sweep + min-blob-area filter on val; it would need the eval output / a real checkpoint to run.
- EMA didn't beat raw — unusual (EMA usually smooths and helps). At epoch 35 EMA macro_f1 was only 0.3688 (EMA had not caught up yet); by epoch 102 EMA macro_f1=0.6179, still below raw's best of 0.6786. This suggests the raw checkpoint at epoch 35 was a genuine peak that EMA's slow shadow missed. Not a bug, but means the "best" is a raw checkpoint, which is slightly less robust than an EMA checkpoint would be.
- Severity MAE (true leaf) = 0.1137 > sev_mae_predmasks = 0.1026: using the true leaf mask makes severity slightly worse, which is counterintuitive if the leaf mask were good. This suggests the leaf channel also over-predicts (so true leaf area is smaller than predicted leaf area, making lesion/true_leaf larger). Both channels over-predict.
- "120 healthy images" doesn't match this run's train healthy count (80). If the user had a specific 120-healthy source in mind, it's not in this output.

**Anything that looks too good: check for leakage.**
- Val macro_f1=0.6786 with healthy F1=1.0 and yellow_leaf_curl F1=0.865 — could healthy be a shortcut? The eval kernel's `healthy_shortcut` check (logistic regression on image stats) would flag this if val holdout acc > 0.9. We didn't run it, so we can't confirm. **Not a confirmed leak**, but the healthy F1=1.0 is suspicious and worth the shortcut check. It is val-only (the shortcut is computed on val), so it's not a test leak, but if healthy is trivially decidable from image stats, the model may not be learning lesions for the healthy class.
- No test leakage: best epoch selected by VAL macro-F1 only; threshold chosen on VAL; severity baseline = VAL median. Test was never touched for fitting/selection. So if test numbers turn out good, it won't be from leakage — but we don't have test numbers.

**What I recommend doing next, and the time it would take:**
1. **Re-run the eval kernel** (`ganrajlongadge/cropguard-v8-eval`) so we get `results.json`/`results.md` with the actual test numbers (PlantSeg test, PlantDoc test, ECE, confusion matrix, severity Spearman, over-prediction on test, healthy-shortcut result). This is the single most important next step — without it the report is mostly val-only. Time: the eval kernel ran in the earlier smoke in ~a few minutes on Kaggle (CPU + a T4); scheduling + running + downloading output ≈ 10-20 min wall-clock, but it needs the eval kernel to be accessible (the download was denied, which may mean the eval kernel doesn't exist under this account or is private — check that first).
2. **Run the post-hoc improver** (`/tmp/prep2/posthoc.py`) against the real best_v8_raw.pt + the eval output, to measure TTA/EMA-avg/threshold-sweep/blob-filter/severity-calibration on val and apply kept options to test. Time: CPU inference ~20s/pass × a few passes ≈ 2-3 min, plus bootsrapping (1000 resamples) — maybe 5-10 min total on CPU. This directly addresses the over-prediction (threshold sweep + min-blob filter) and severity calibration. Caveat: `/tmp/prep2` was wiped on this host, so the script isn't here; it would need to be re-created or copied from the repo backup.
3. **Investigate the over-prediction:** min-blob-area filtering (remove predicted connected components under N px) and a lesion-threshold sweep on val (the post-hoc script does both). If a small blob filter brings over-prediction from 7.6× toward 1× while keeping IoU reasonable, that's the fix. Time: minutes (it's a val-only sweep).
4. **Verify the healthy shortcut** (run the eval kernel's `healthy_shortcut` or replicate it on val) to confirm healthy isn't a trivial image-stat shortcut. Time: minutes.
5. **If test numbers come back weak** (macro-F1 on test well below val's 0.6786, or lesion IoU still ~0.03 on test), consider the v9 training options (`/tmp/prep2/train_v9.py` from the earlier work — seg-input-size 384, lesion-crop-prob 0.5, class-balanced sampler, seg-loss-warmup 5), which were written but not run. Time: a Kaggle training run ≈ 55-70 min on a T4×2, but that burns quota; only do this if the eval results justify it.

**Hard constraint respected:** no test split used for fitting or selection; no new training started; the eval kernel was not re-run from here (we report test as not_measured). The only Kaggle API call made was `kaggle kernels status` + `kaggle kernels output` (read-only observation of the existing COMPLETE train kernel and download of its output); the eval kernel download was denied and we did not push/start anything.

---

### Appendix: raw source files referenced

- `kaggle/v8/cropguard-v8-train.py` (train driver — selection L797-812, check_pretrained L323-345, EMA L155-184, cosine LR L193-197, seg_loss call L558/L746, optimizer assertions L690-699, class weights L667-678, sampler L371-383, build_model L316-319).
- `model/architecture/v7_model.py` (2-channel seg head L53/L60-61/L118-125/L137-144; analytic_severity).
- `model/architecture/backbone.py` (v7 levels L11-16, L21-22).
- `model/training/losses.py` (seg_loss_v7 L227-266).
- `model/training/dataset.py` (real lesion/leaf masks + sev_cont L284-304, L366-367; shared _augment L126-170).
- `model/training/train_full_v7.py` (v7 single-channel seg, pseudo-mask, two_channel_seg path L278-284) — the v7 failure reference.
- Train output: `/tmp/kaggle_out_train/best_v8_raw.pt`, `last_v8.pt`, `v8_train_log.csv`, `stage_summary.json`, `stage2_training.json`, `stage0_summary.json`, `stage0_label_audit.json`, `stage1_memorize.json`, `stage1_memorize_nogeo.json`, `cropguard-v8-train.log`.
- v7 reference logs: `train_full.log`, `runs_v7_120/train.log` (val macro_f1=0.6433, lesion_iou=0.0282 at epoch 120).

### Appendix: files written for this report

- `/tmp/prep2/v8_raw_results.json` — machine-readable summary of everything we have from the train output, in a schema that mirrors the eval kernel's `results.json` (test fields left as `not_measured`).
- This report: `/tmp/v8_report.md`.
