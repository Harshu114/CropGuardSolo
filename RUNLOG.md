# CropGuard v7 Handover Execution Runlog

Date: 2026-10-11
Repo: CropGuard_Final

## Step 0: Orientation
- Repo structure examined:
  - Model: `model/architecture/v7_model.py`, `v7_heads.py`, `backbone.py`, `neck.py`
  - Training & Loss: `model/training/train.py`, `dataset.py`, `losses.py`, `losses_patch.py`, `consistency.py`
  - Eval & Calib: `model/training/eval_tta.py`, `eval_plantdoc.py`, `calibrate_abstention.py`
  - Docs: `docs/V7_DATA_AUDIT_2026-10-11.md`, `docs/V7_EVAL_2026-10-11.md` read and verified.
- Hardware verified:
  - GPU: NVIDIA GeForce RTX 2050 (4096 MiB VRAM)
  - CPU: 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (12 cores)
  - Step time (batch 16): ~122.6 ms -> Estimated epoch time (635 train samples): ~4.87 s
- Corpus verification:
  - Path `/home/logiclocus/tmp/cgpack/cropguard_v7_split` exists.
  - Image counts: train=635, val=136, test=130 (total 901 images).
  - Fallback hits to old box pseudo-masks: 0 (all 901 samples have dedicated lesion & leaf masks).
  - Constant severity baseline: train median = 0.0127 (after exclusion), val baseline MAE = 0.0384, test baseline MAE = 0.0409.

## Step 1: Fix Labels
- Leaf mask exclusion list generated (`exclusion_list.txt` & `leaf_exclusion_list.txt`).
- Total excluded images: 39 (10 images with leaf area < 1%, 29 images in the 1-5% area group).
- Unit test executed: `test_mask_exclusion.py` passes with 0 loss perturbation difference and 0 gradients on excluded samples with no NaNs.

## Step 2: Data Facts
- Class distributions measured across train/val/test splits.
- Pretrained weights loading verified: timm `fasternet_t0` pretrained weights load cleanly; random initialization diff confirmed (>455 L1 distance).
- Severity distribution after exclusion: median=0.0127, p10=0.0028, p90=0.0499.

## Step 3: 50-Image Overfit Test
- Run executed on 50 stratified samples.
- Classification accuracy: 1.0000 (Passes >= 0.95 bar).
- Segmentation analysis & findings logged:
  - Default `seg_loss_v7` (unweighted BCE + Dice on ~0.6% lesion pixels) collapses to background: IoU=0.0000.
  - Custom loss with `pos_weight=50.0` prevents background collapse (100% non-empty predictions, Spearman=0.9050), but BilinearDecoder single-stage /4 upsampling reaches IoU ~0.05-0.28 because fine pixel-level lesion boundaries require progressive upsampling features.
  - Step 3 gate status: Fails lesion IoU >= ~0.8 pass bar. STOPPED per instructions: "If any fail, STOP and debug labels/loss/data loading first and report what you found."

## Follow-up Action Items (A-D) Execution Log

### Item A: Resolution of Contradiction & Detailed Threshold Analysis
- Overfit run 1 (`task-161`, unweighted BCE + Dice): 0/50 non-empty predictions -> ALL-BACKGROUND COLLAPSE.
- Overfit run 2 (`task-283`, BCE with pos_weight=50 + Dice): 50/50 non-empty predictions, but low IoU (0.05).
- Precise failure diagnosis:
  - True lesion pixel fraction: 0.00628 (0.628% of pixels)
  - At threshold 0.3: predicted fraction 26.90% (42.8x over-prediction), IoU 0.0212, Dice 0.0414 (50/50 non-empty)
  - At threshold 0.5: predicted fraction 12.74% (20.3x over-prediction), IoU 0.0347, Dice 0.0666 (50/50 non-empty)
  - At threshold 0.7: predicted fraction 2.86% (4.5x over-prediction), IoU 0.0338, Dice 0.0646 (47/50 non-empty)
  - Actual diagnosis: OVER-PREDICTION & BLUR. The model learned to predict foreground everywhere around the lesion due to massive single-stage /4 upsampling blur and excessive positive weighting.

### Item D: Exclusion List Audit & Restoration
- Total <5% leaf area group in train: 39 images (10 with <1% leaf area, 29 in 1-5% group).
- Visual judgment & contact sheet analysis:
  - 10 <1% images: severe failures where GrabCut failed completely.
  - 29 images in 1-5% group: 12 judged failures where severe over-clipping occurred (severity blown up to 0.25-0.95 due to tiny truncated leaf masks).
  - 17 images restored to training as good leaf masks.
- Output counts: 22 total judged failures written to `exclusion_list.txt` and `leaf_exclusion_list.txt`.
- Severity distribution post-exclusion (n=613): median=0.0131, p10=0.0028, p90=0.0624 (p90 preserved).

### Item B: High-Resolution Architecture & Focal-Tversky Overfit
- Upgraded `BilinearDecoder` in `model/architecture/v7_heads.py`:
  - Fuses multiscale P3/P4/P5 with stride-4 C2 skip (48ch).
  - Progressive upsampling: 80x80 -> 160x160 (stride 2) with 3x3 conv + BN + Hardswish.
  - Progressive upsampling: 160x160 -> 320x320 (stride 1) with 3x3 conv + BN + Hardswish.
- Upgraded `seg_loss_v7`:
  - Focal-Tversky loss (alpha=0.3, beta=0.7, gamma=1.33) + BCE (pos_weight=20.0).
- 50-image overfit test run (`task-487`):
  - Classification accuracy: 1.0000 (100%)
  - Non-empty predictions: 50/50 (100.0%)
  - Severity Spearman: 0.6720
  - Mean IoU: 0.0465 (threshold 0.7), predicted fraction 5.20% vs true 0.63%.
  - Gate status: IoU does not reach 0.50 threshold on the 50 images.

### Item C: Class Problem Resolution
- Actual image counts from label files: 4 classes were completely missing (`healthy`, `leaf_miner`, `spider_mites`, `target_spot`).
- Built `healthy` class:
  - Deduplicated with pHash (887 existing hashes protected from leakage).
  - Extracted 120 clean healthy tomato images (80 train, 20 val, 20 test).
  - Generated lesion mask = 0, severity = 0, green leaf mask.
- Dropped 3 empty classes: `leaf_miner`, `spider_mites`, `target_spot`.
- Final class list (8 classes):
  `['healthy', 'bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold', 'mosaic_virus', 'septoria_leaf_spot', 'yellow_leaf_curl_virus']`
- Updated label text files to dense 8 classes across all splits.
- PlantDoc evaluation alignment: all 8 tomato folders map 1-to-1 to the 8 classes (0 tomato images excluded, 19 non-tomato folders skipped).

### Item E: Runtime Measurement
- Measured on RTX 2050 (4 GB VRAM) with batch_size=8, AMP enabled:
  - Step time: 247.9 ms
  - Measured time per epoch: 22.31 seconds (0.37 min)
  - Projected total time for 30 epochs: 11.2 minutes (well within 90-minute timebox).

---

# v8 HANDOVER AUDIT — Items 1–3 (2026-10-12, CPU-only local session)

## Item 1 — orientation (read-only)
- Repo: `/home/logiclocus/HarshalPrivate/Projects/collegeProjects/All tries of cropguard/CropGuard_Final`.
- v8 driver: `kaggle/v8/cropguard-v8-train.py` (39 KB, mtime 2026-10-12 00:29) + `kaggle/v8/kernel-metadata.json`.
- Kaggle kernel `ganrajlongadge/cropguard-v8-train` = COMPLETE (kaggle kernels status). `ganrajlongadge/cropguard-v8-eval` was NEVER pushed (`Permission 'kernels.get' was denied`); only local files `kaggle/v8-eval/` + `kaggle/v8/cropguard-v8-eval.py` exist.
- v8 artifacts downloaded to `/home/logiclocus/tmp/kout_v8/`: `best_v8_raw.pt` (9.8 MB), `last_v8.pt` (39 MB, has ema_shadow + opt), `cropguard-v8-train.log` (54 KB), `v8_train_log.csv` (102 rows), `stage0_summary.json`, `stage0_label_audit.json`, `stage1_memorize.json`, `stage1_memorize_nogeo.json`, `stage2_training.json`, `stage_summary.json`, `code/leaf_exclusion_list.txt`. NO `best_v8_ema.pt` (EMA never won selection).
- GPU: `nvidia-smi` → "couldn't communicate with the NVIDIA driver"; no `/dev/nvidia*`; no nvidia modules loaded; torch 2.13.0+cu130 `cuda_available False`; RTX 2050 GA107 present at PCI 0000:01:00.0. → Local training is CPU-only.
- v8 ran 102/120 epochs: log "TIME CAP: 54.7 min used, ~0.54 min/epoch, cap 55.0 min -> stopping cleanly at epoch 102".
- v8 vs v7 (from code + logs, verified):
  1. v8 loss = cls CE (inverse-sqrt class weights capped 3x, label_smoothing 0.05) + `seg_loss_v7` only. No consistency loss (v8 never calls `multitask_loss`; v7's did add cross-task consistency).
  2. v8 augmentation = PHOTOMETRIC ONLY (Stage-1 gate failed with geometry ON IoU=0.19, passed geometry OFF IoU=0.67 → PLAN B). v7 used full geometric+HSV aug.
  3. v8 evaluates BOTH raw and EMA each epoch, selects by val macro-F1 (tie-break lesion IoU) → picked RAW epoch 35. v7 evaluated EMA only; the v7 LOG was produced by a composite score `f1+iou−sev_mae` (epoch 45: 0.6090+0.0229−0.2419=0.3900 exactly), while current `train_full_v7.py` (edited 22:34, after the 22:18 log) says macro-F1 only.
  4. v8 added pre-training gates: dataset sha256, forced-geometry test, pretrained-weight check, label audit, 8-image memorization gate. v7 had none of these.
  5. Same architecture both: fasternet_t0 (pretrained) + BiFPNLite(96) + BilinearDecoder 2-ch seg + lesion-gated cls; severity analytic = lesion/leaf area, NOT learned; nc=8.
  6. v8 excludes label-audit failures from val/test (151 val used) but the TRAIN-side exclusion is a NO-OP bug (see 2f/2e below): log line "train: 715 images (0 dropped by label audit)".
  7. v8 stopped at 102/120 epochs (time cap); v7 completed 120.

## Item 2 — sanity checks (all commands + raw outputs in /home/logiclocus/tmp/cgcheck2/out_*.txt)

### 2a pretrained weights — PASS
Command: `python3 /home/logiclocus/tmp/cgcheck2/chk_a.py`
Output: `mode=pretrained first-conv L1 vs random init = 0.237754 (> 0.01)`; random-vs-random reference L1 = 0.0231; pretrained first-conv mean/std = 0.0018/0.316 vs random 0.0198. Cached at `~/.cache/huggingface/hub/models--timm--fasternet_t0.in1k/snapshots/*/model.safetensors`. Kaggle Stage-0 shows the same (mode=pretrained l1=0.23837).

### 2b selection code
`kaggle/v8/cropguard-v8-train.py:797-813` → `score, tie = m["macro_f1"], m["lesion_iou"]` over raw+EMA; saves `best_v8_raw.pt` / `best_v8_ema.pt`. Chosen: `best_v8_raw.pt` epoch=35 which=raw. `best_v8_ema.pt` does not exist. stage_summary.json: `selected_epoch=35 selected_by="val macro-F1 (tie-break lesion IoU)" tag="selected:raw" n_valid_cls=151`.

### 2c raw vs EMA + EMA updating — reported numbers are RAW; EMA moves every epoch
`chk_c.py`: CSV max raw_f1 = epoch 35 (0.6786) = the selection; max ema_f1 = epoch 68 (0.6428). EMA checksum changes at every epoch (epoch1 3.469940 → epoch102 2.184402; unchanged-pair fraction 0.0); log `ema_moved=True` on every line; last_v8.pt EMA-shadow L1 vs raw params on patch_embed = 0.000448. At epoch 35 EMA still lagged (f1 0.3688 vs raw 0.6786).

### 2d optimizer/LR — all trainable, formula exact
`chk_d_e.py`: backbone 77/77, neck 55/55, seg_head 28/28, cls_head 6/6, det_head 14/14 in optimizer; 2,533,254 total params all requires_grad=True; frozen list = NONE; seg_head 81,666 params in optimizer at lr 5e-4. Detach sites: only by-design (cls-head mask detach v7_heads.py:140, EMA shadow, eval/no_grad, dfl alpha no_grad).
cosine formula vs CSV lr (epoch: formula / CSV): 1: 9.889e-05/9.889e-05; 30: 4.441e-04/4.441e-04; 60: 2.676e-04/2.676e-04; 90: 8.026e-05/8.026e-05; 102 (final): 3.060e-05/3.060e-05 — all OK (steps/epoch=90, total=10800, warmup=450, min_lr floor 1e-6).

### 2e five real training steps — all finite, all components live
`chk_d_e.py` on the real loader (photo-only aug, as run): per-component loss (lesion BCE / 3×Focal-Tversky / leaf BCE / leaf Dice / cls CE) e.g. step1: 0.8582 / 2.9000 / 0.6911 (dice 0.5096) / 2.0887, total 7.0475, `finite=True`; step5: 0.8726 / 2.8781 / 0.4969 (dice 0.4002) / 2.0624, total 6.7102. Local replica of the seg math == the actual `seg_loss_v7` return to 4 decimals every step. Grad norms (seg/neck/bb) ≈ 1.3–1.7 / 0.45–0.7 / 0.28–0.63 — all nonzero. `healthy_in_batch`≥1 in 4/5 batches, finite. No consistency loss in v8 (grep proof included). 5 steps took 11.0 s(!).

### 2f mask alignment — aligned; two dataset findings
- `model/training/test_dataset_aug.py <root>` → OVERALL: PASS (PART A all 4 flip/scale combos match ≤2 px; PART B n=20 mean lesion-inside-leaf 1.0000).
- `chk_f.py`: native size mismatch between image and lesion/leaf masks: 0 (all 715/156/150). EXIF orientation tags: train 5 / val 1 / test 3 — the loader pipeline ignores EXIF for image AND mask consistently (no transpose anywhere).
- Contact sheet (8 real DataLoader batches, aug off, saved `work_f/contact_sheet_8batches.png`): lesion overlays sit on the visible necrotic spots; lesion-inside-leaf mean 1.0000 min 1.0000 (n=31 shown); image/mask/leaf all 320×320.
- YOLO boxes vs lesion masks disagree BY DESIGN: only mean 0.348 of lesion pixels fall inside boxes (76% of images <0.5), because some boxes annotate single tiny spots while the lesion mask covers every spot (see `work_f/box_vs_lesion.png`). NOT a misalignment.
- class0 ↔ empty-lesion agreement is a perfect bijection: 80 healthy = 80 empty; 0 mismatches either way over all 715 train images (635 diseased all nonempty).
- BUG (train-only): `build_train_ds` compares the stem against the exclusions DICT instead of `exclusions["train"]` → "train: 715 images (0 dropped by label audit)" while val correctly drops 5 → 151. The 32 label-invalid train images (empty label files) stayed in training (they carry cls=-1 so they were already out of cls loss; their seg masks are valid). Val/test exclusions ARE applied.

### 2g 8-image memorization, no augmentation — PASS (local, CPU)
`chk_g.py` (3 healthy `tomato_healthy_train_000/001/002` + 5 bacterial-spot images, AdamW 1e-3, 500 steps, augment OFF): IoU/Dice at best threshold: step100 0.1926/0.3230, step200 0.2384/0.3851, step300 0.5995/0.7496, step400 0.7523/0.8587, **step500 0.9149/0.9556 → PASS (≥0.5)**; seg loss 5.05→1.27; cls memorized 8/8 (cls loss floor 0.2648 = label-smoothing floor); healthy images: lesion BCE 0.0396/0.0385/0.0384 (finite, no NaN), predictions healthy for all 3 at every checkpoint. JSON: `work_g/memorize_8_noaug.json`. (Kaggle gate: aug-ON FAIL IoU 0.1935@500; aug-OFF PASS IoU 0.6708@500 — same qualitative result.)

## Item 3 — v8 log at best / last epoch
- Best-epoch line (epoch 35, RAW): `E 35/120 [18s lr=4.21e-04] cls=0.3505 seg=3.1593 | RAW acc=0.6954 macro_f1=0.6786 iou=0.0459 dice=0.0878 sevMAE(pred)=0.1026 sevMAE(trueleaf)=0.1137 pred_area=4729.0 true_area=621.3 over=7.61x | EMA acc=0.4636 macro_f1=0.3688 ...`
- Last epoch (E102): `RAW acc=0.6291 macro_f1=0.6056 iou=0.0417 dice=0.0800 sevMAE(pred)=0.0417 ... pred_area=1923.7 true_area=621.3 over=3.10x | EMA acc=0.6424 macro_f1=0.6179 iou=0.0424 ...`
- Stage-3 val summary (selected best_v8_raw.pt): acc=0.6954 macro_f1=0.6786 iou=0.0459 dice=0.0878 sevMAE(pred)=0.1026 sevMAE(trueleaf)=0.1137 pred_area=4729.0 true_area=621.3 over=7.61x; per-class F1: 0:1.000(n=20), 1:0.562(n=21), 2:0.655(n=33), 3:0.722(n=19), 4:0.744(n=20), 5:0.462(n=8), 6:0.419(n=14), 7:0.865(n=16). `sev_beats_baseline=false` (0.1026 vs 0.0384).
- CSV scan: raw IoU max 0.0479 (epoch 46), last-10-epoch mean 0.0414 → lesion IoU is FLAT ~0.04 across all 102 epochs. Severity: only epoch 84 (0.0348) dipped under the constant baseline; all others above.
- Val n=151 (5 label-audit exclusions), not 156.

---

# ROOT CAUSE FOUND: lesion masks are 1-px polygon OUTLINES (2026-10-12, continued)

## Why did v8 segmentation fail? (all commands below, CPU)

### Evidence — targets, not architecture
- `python3 chk_maskstats.py` (all splits): median **181 connected components** per lesioned mask @320 (p90=424), largest component p50 ≈ **25 px**; fraction of pixels surviving 3x3 erosion mean=0.000, `frac_images<0.05=1.000`; constant-blob ceiling IoU on val = **0.0000** (no pixel is lesion in >50% of train images). class0↔empty still a perfect bijection; lesion-outside-leaf = 0.0000.
- `python3 chk_masknative.py`: comps p50/p90 = **90/297 at NATIVE resolution** (largest comp p50 = 104px) -> masks are inherently fragmented; the 320 NEAREST resize doubles it (181 comps).
- `python3 chk_maskthick.py`: erosion survival mean=0.000 on **all 635** lesioned masks; ASCII dump = 1-px open curves; 378/635 enclose loops (8-conn).
- Lesion mask pixel values strictly {0,255} (no hidden fill value being missed).
- **SOURCE**: `data_pipeline/scripts/prepare_plantseg.py:97-118` `poly_to_mask()` walked COCO polygon edges vertex-to-vertex writing 1px per step — the interior was never filled (docstring said "binary lesion mask"; output was outline-only).
- Secondary: the v8 driver's "photometric-only" PLAN B (cropguard-v8-train.py:514-523) forced flip=False/scale=1.0 inside `_augment` only; HSV jitter + the 30% `_paste_background` leaf-cutout (dataset.py:262) stayed ON for the whole run.

### Consequence
Every task downstream inherited the bug: seg targets = lacy scribbles (unlearnable at model resolution; 8-image gate could only pass by memorizing 8 curve sets), IoU metric ~0 for any smooth prediction, severity GT = perimeter/leaf ratio (p50 0.0125) instead of area ratio, boxes = outline fragments ("tiny per-spot boxes" in item 2f).

### Fix implemented
- Root-cause code fix: `prepare_plantseg.py poly_to_mask()` now fills polygon interiors (even-odd scanline, pure numpy). Verified: 2 test rects -> 7000 px filled (outline-only ~380); interior points marked.
- Corpus repair (fill existing outlines): dilate(3x3) -> flood-fill outside from border -> region = dilated|interior -> erode(3x3). Stats: comps p50 **90 -> 6**, largest comp p50 **104 -> 20003 px**, erosion survival **0.000 -> 0.916**, area x**22.3** (p10 8.4 / p90 74.1), 635/635 masks changed, **0** needed a kernel > 3.
- Montage (fills match polygon silhouettes exactly): `/home/logiclocus/tmp/cgcheck2/work_fill/outline_vs_filled.png`
- New corpus built: `/home/logiclocus/tmp/cgpack/cropguard_v8_filled/` — masks_lesion = solid regions (6.6 MB total), images/labels/masks_leaf symlinked to the v7 corpus (originals untouched).

### Metrics after fix
- New severity (filled masks, dataset.py rules): train n=603 mean=0.4593 p50=0.3648 baselineMAE=0.3125; val n=131 mean=0.3912 p50=0.2340 MAE=0.3023; test n=120 mean=0.4368 p50=0.2751 MAE=0.3370. (The old baseline 0.0384/0.0409 was a perimeter-ratio baseline and is superseded.)
- Re-eval `best_v8_raw.pt` (trained on outlines) on val: vs ORIGINAL outlines best IoU 0.0466 (thr 0.7); vs FILLED masks best IoU **0.1241** (thr 0.3), @0.5 0.0823. Per-class @0.5 vs filled: bacterial 0.0649, early_blight 0.1020, late_blight 0.0672, leaf_mold 0.0824, mosaic 0.0550, septoria 0.0750, curl 0.1154, healthy 0.0000. Directionally consistent: even the outline-trained model overlaps filled regions ~2.7x better than outlines.
- 50-image memorization test (outline targets, noaug, v8 stack, `chk_h_50.py`, log `out_chk_h2.txt`, 16.6 min): **final FAIL — 50-set IoU@0.5 = 0.0634, best-threshold (0.7) = 0.0734** (script verdict uses the best-thr value). Trajectory 0.0210 (100) -> 0.0352 (200) -> 0.0422 (300) -> 0.0585 (400) -> 0.0634 (500) while cls_acc hit 49/50 by step 200 and stayed. Healthy images: pred lesion px = 0 (correct), healthy BCE 0.3550 -> 0.0255; their IoU 0.0000 is the empty-GT artifact. Per-class IoU@0.5: early_blight 0.1118 best, late_blight 0.0286 worst; best single image 0.2150 (early_blight_120), worst lesioned 0.0076 (late_blight_10). Conclusion: with outline targets, cls memorizes instantly but seg cannot fit even 50 fixed images -> the target itself is the bottleneck. (The 8-image gate passed at 0.9149 only because 8 curve sets are traceable via threshold-ramp memorization on a tiny fixed set.)

### Corrections from the doc-verification pass (re-ran raw checks)
- Colleague doc's "50-image v7-era test ran with geometry aug ON" is wrong: `run_overfit_item_b.py` uses `augment=False` (as does `run_overfit_50.py`); it did use nc=11 + multitask_loss, so it still isn't a clean v8-stack comparison.
- "over-prediction 3.1-7.6x" understates the full-run series: range over all 102 epochs is 3.10x (ep102) to 89.27x (ep1).
- "threshold always pinned at 0.7" is not verifiable from stored data (no threshold column in the CSV; only stage_summary best_thr=0.7).
- "val IoU flat over 102 epochs" — it rises 0.0074->0.0342 over the first ~12 epochs, then flat 0.04-0.048.

### Next step (awaiting go-ahead)
- Retrain on the filled corpus on Kaggle: upload the 6.6 MB masks patch + a v8-derived train script that overlays it; same recipe otherwise (single variable changed). ~55 min GPU.
- Then items 4-6 (test eval, data checks, calibration/export) against the retrained checkpoint.

---

## 2026-10-12 — v9: SECOND corpus defect (leaf masks exclude necrosis) -> union fix -> repaired-mask Kaggle bundle + local smoke PASS (autonomous retrain)

### Discovery: the filled lesions did not sit inside the leaf masks
While wiring the filled masks into the v8 driver, the stage-0 geometry gate (PART B) FAILED on filled targets: mean lesion-inside-leaf = 0.6386, min = 0.0909. Root cause: `masks_leaf` (grabcut-derived) covers HEALTHY tissue only — necrotic areas are holes/bays. The old 1-px outlines had passed PART B at 1.0000 only because a 1-px curve hugs the leaf side of the necrosis boundary; the filled region lands mostly in the holes.

Command (re-run 2026-10-12 for raw output): `python /tmp/v9leafchk.py` — outlines vs filled measured against the ORIGINAL leaf masks:
```
train: n=635  OUTLINE inside-leaf mean=1.0000 min=1.0000 | FILLED inside-leaf mean=0.4429 min=0.0069  frac<0.9: outline=0.000 filled=0.835
   worst tomato_early_blight_google_0427.png  outline_px=  2457 inside=1.000 | filled_px= 356649 inside=0.007
   worst tomato_late_blight_google_0092.png   outline_px=  2189 inside=1.000 | filled_px= 243266 inside=0.009
val:   n=136  OUTLINE inside-leaf mean=1.0000 min=1.0000 | FILLED inside-leaf mean=0.4558 min=0.0157  frac<0.9: outline=0.000 filled=0.794
test:  n=130  OUTLINE inside-leaf mean=1.0000 min=1.0000 | FILLED inside-leaf mean=0.4424 min=0.0088  frac<0.9: outline=0.000 filled=0.815
```
A hole-fill-only experiment (`chk_leaf_fix.py`: `binary_fill_holes(leaf)`) lifted the mean 0.44 -> 0.61 but the worst masks barely moved (necrotic bays open to the leaf border are not holes). Final fix: `leaf_v9 = fill_holes(leaf>127) | (lesion_filled>0)` — whole leaf = healthy tissue ∪ diseased area; makes lesion ⊆ leaf by construction and fixes the severity denominator. Verification (fresh run):
```
train: n=635 lesion-inside-leaf mean=1.000000 min=1.000000 frac<0.99=0.0000
val:   n=136 lesion-inside-leaf mean=1.000000 min=1.000000 frac<0.99=0.0000
test:  n=130 lesion-inside-leaf mean=1.000000 min=1.000000 frac<0.99=0.0000
```

### Repair script = single source of truth (re-run 2026-10-12)
Command: `python3 data_pipeline/scripts/repair_masks_v9.py --v7 /home/logiclocus/tmp/cgpack/cropguard_v7_split --out /home/logiclocus/tmp/cgpack/cropguard_v9_fixed` (log `/tmp/v9repair2.log`)
```
leaf exclusions: 22 stems (.../CropGuard_Final/leaf_exclusion_list.txt)
train: n=715  lesion filled/outline median=22.3x  leaf changed=715  lesion-inside-leaf(union) mean=1.0000 min=1.0000
val:   n=156  lesion filled/outline median=19.0x  leaf changed=156  lesion-inside-leaf(union) mean=1.0000 min=1.0000
test:  n=150  lesion filled/outline median=19.5x  leaf changed=150  lesion-inside-leaf(union) mean=1.0000 min=1.0000
=== severity (lesion_filled / leaf_union), dataset.py rules ===
train n=663 (label-valid, leaf-excl removed) mean=0.3016 p50=0.2263 baselineMAE(mean)=0.2261  [incl-leaf-excl n=683]
val   n=151 (label-valid, leaf-excl removed) mean=0.2705 p50=0.1687 baselineMAE(mean)=0.2293  [incl-leaf-excl n=151]
test  n=140 (label-valid, leaf-excl removed) mean=0.2986 p50=0.1774 baselineMAE(mean)=0.2556  [incl-leaf-excl n=140]
```
val baseline **0.2293** is what the v9 kernel reports against (constant predictor = split mean); the old outline baseline 0.0384 is superseded (different targets). Output corpus: `cropguard_v9_fixed/` = real `masks_lesion/` (filled) + real `masks_leaf/` (union) + images/labels symlinked to v7; v7 untouched.

### Kaggle bundle (ready, not yet run)
- Dataset `ganrajlongadge/cropguard-v9-filled-masks` (private, ready): `masks_lesion/` (from cropguard_v8_filled) + `masks_leaf/` (from cropguard_v9_fixed) uploaded with `--dir-mode zip` (extracts at mount root); total 9,231,266 bytes; lesion md5 30/30 identical vs v8_filled montage check.
- Kernel `kaggle/v9/cropguard-v9-train.py` + `kernel-metadata.json` (id `ganrajlongadge/cropguard-v9-train`, GPU+internet, dataset_sources = cropguard-v8-code, cropguard-v7-diagnostics-input, cropguard-v9-filled-masks). v9 = v8 recipe, single variable changed (the masks): `prepare_corpus()` builds `WORK/corpus_v9` (v7 images/labels symlinked + both repaired mask dirs) and gates before any GPU work: mask counts 715/156/150 both dirs, 0 missing masks, lesion fill ratio >= 5x, lesion-inside-leaf >= 0.99, dataset.py sha256.
- Pickle portability fix: the v8-inherited PLAN-B photometric-only toggle used a closure monkey-patched onto the dataset; Python 3.14's forkserver DataLoader workers pickle the dataset and that crashed with `_pickle.PicklingError: Can't pickle local object <function build_train_ds.<locals>.photo_only ...>`. Replaced by module-level `_define_photo_only()` registering a picklable `_PhotoOnlyDataset(CropGuardDataset)` (qualname fixed, `ds.__class__` swap; import-safe at module level so forkserver/spawn children can resolve it). Verified: ForkingPickler.dumps OK (1567 bytes) and the override is active (mask px stable across samples under photo-only aug).

### Local CPU smoke #5 (same command as Kaggle, capped) — PASS end-to-end
`V8_WORK=/tmp/v9smoke/work V8_CODE_ROOT=<repo>/model V8_INPUT_ROOT=/tmp/v9smoke/input/v7 V9_FILLED_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v9_fixed V9_CORPUS=/tmp/v9smoke/corpus_v9 V8_STAGE1_STEPS=20 V8_FORCE_AUG=photo V8_TRAIN_SUBSET=16 V8_VAL_SUBSET=16 V8_EPOCHS=1 V8_CAP_MIN=6 python kaggle/v9/cropguard-v9-train.py` (log `/tmp/v9smoke/smoke5.log`)
```
[v9] leaf_exclusion_list.txt: 22 stems (/tmp/v9smoke/corpus_v9/code/leaf_exclusion_list.txt)
[v9] mask counts train/val/test: lesion=715/156/150 leaf=715/156/150
[v9] overlay fingerprints over 32 sampled masks: lesion filled/outline median=20.4x ; lesion-inside-leaf mean=1.0000 min=1.0000
[v9] sha256 gate PASS (this is the fixed dataset.py)
  === PART B: 20 real train samples, lesion inside leaf ===
  n=20  mean lesion-inside-leaf=1.0000  min=1.0000  -> PASS
OVERALL: PASS
[v9] pretrained check: mode=pretrained first-conv L1 vs random init = 0.238369 (> 0.01)
[v9]   train: images=715 label_files=715 sum_class_counts=683 excluded=32
  step   20: lesion IoU=0.4008 cls=0.3554 seg=3.1492 total=3.5046 grad_norm(seg/neck/bb)=1.53e+00/9.81e-01/9.86e-01
[v9] E  1/1 [141s lr=9.89e-05] cls=2.0697 seg=5.4112 | RAW acc=0.0000 macro_f1=0.0000 iou=0.2915 dice=0.4514 sevMAE(pred)=0.5958 sevMAE(trueleaf)=0.4490 pred_area=61539.3 true_area=19082.3 over=3.22x | EMA acc=0.0000 macro_f1=0.0000 iou=0.1864 dice=0.3142 ... | ema_moved=True
[v9] TOTAL 3.2 min; stage states: {'corpus_v9': {... 'overlay_median_ratio': 20.40, 'lesion_inside_leaf_min': 1.0}, 'stage0': 'ok', 'stage1': 'photo', 'stage2': 'ok'}
```
(Smoke metrics are 1 epoch on a 16-image subset — pipeline proof only, not results. The stage2 epoch is the actual proof the picklable-class fix works: forkserver workers ran the epoch.)

### Next step
- `kaggle kernels push -p kaggle/v9` and monitor to completion; artifacts land in /kaggle/working (best_v8_raw.pt, stage_summary.json, stage2_training.json, v8_train_log.csv).
- Then evaluate v9 vs v8 on val (best-epoch metrics, threshold sweep, per-class IoU, severity vs 0.2293 baseline), and proceed to test evaluation / calibration / ONNX export.

### Kernel launched
- `kaggle kernels push -p kaggle/v9` -> `Kernel version 1 successfully pushed` (https://www.kaggle.com/code/ganrajlongadge/cropguard-v9-train)
- `kaggle kernels status ganrajlongadge/cropguard-v9-train` -> `has status "KernelWorkerStatus.RUNNING"` (poll log `/tmp/v9kern_status.log`)

### v9-eval kernel prepared while training runs
- `kaggle/v9-eval/cropguard-v9-eval.py` (926 lines, from cropguard-v8-eval.py): OLD = cropguard-v8-train output checkpoint (`best_v8_ema.pt` -> `best_v8_raw.pt` -> `last_v8.pt`), NEW = cropguard-v9-train output; BOTH evaluated on the rebuilt `corpus_v9` with the repair fingerprints gated before any GPU work. Val/test/PlantDoc + temperature/abstain + ONNX FP32/INT8 + CPU-latency sections unchanged; ONNX written as `cropguard_v9_*.onnx`.
- `kaggle/v9-eval/kernel-metadata.json`: kernel_sources = cropguard-v8-train + cropguard-v9-train; dataset_sources = v7-diagnostics-input, v8-code, plantdoc-test, v9-filled-masks. Push AFTER v9-train completes (kernel_sources needs finished outputs).
- Local smoke of the copied corpus builder: `V9E_WORK=/tmp/v9eval_smoke/work V9_CORPUS=/tmp/v9eval_smoke/work/corpus_v9 V8_INPUT_ROOT=/tmp/v9smoke/input/v7 V9_FILLED_ROOT=.../cropguard_v9_fixed python -c '<import v9-eval; prepare_corpus()>'`:
```
[eval] v9 corpus built: /tmp/v9eval_smoke/work/corpus_v9 (v7=/tmp/v9smoke/input/v7, repaired=/home/logiclocus/tmp/cgpack/cropguard_v9_fixed)
[eval] mask counts train/val/test: lesion=715/156/150 leaf=715/156/150
[eval] repair fingerprints: lesion filled/outline median=20.4x ; lesion-inside-leaf min=1.0000
```
- Note: v8's own `stage_summary.json` (acc 0.6954, macro_f1 0.6786, lesion_iou 0.0459 @thr 0.7, sev_mae_predmasks 0.1026) was computed on the OUTLINE corpus -- its IoU/severity are not comparable to v9-era numbers; the like-for-like v8-vs-v9 comparison is the eval kernel above (v8 checkpoint re-evaluated on the repaired corpus).

## 2026-10-12 — v9 code audit (user asked: what exactly changed, is it data or architecture, hunt leftover bugs)

### Change inventory (evidence-based)

ARCHITECTURE / MODEL / DATASET / LOSSES: **zero changes.** Evidence:
```
$ find model -name "*.py" -exec stat -c "%y  %n" {} \; | sort   (newest 3)
2026-10-11 22:34:06  model/training/train_full_v7.py
2026-10-11 23:58:54  model/training/dataset.py
2026-10-11 23:59:24  model/training/test_dataset_aug.py
(nothing touched on 2026-10-12; v7_model.py = 2026-10-10 23:00, v7_heads.py = 2026-10-11 21:15)
$ sha256sum model/training/dataset.py
ff0102ed509d595abe08b79b8e1902fd6756cb7404f5efd102d1a507b90696db   (matches the sha256 gate constant)
$ diff -u kaggle/v8/cropguard-v8-train.py kaggle/v9/cropguard-v9-train.py | grep -c "^[+-]"
259 lines (+168/-17) and NONE of them touch the model, dataset, losses or augmentation math.
```
DATA: the v9 Kaggle dataset `ganrajlongadge/cropguard-v9-filled-masks` (9,231,266 bytes, 715/156/150 in both mask dirs) with the two repairs from `data_pipeline/scripts/repair_masks_v9.py`:
  - masks_lesion: 1-px polygon outlines -> filled polygon regions (lesion pixel gain median 22.3x/19.0x/19.5x per split)
  - masks_leaf: healthy-tissue-only -> fill_holes(leaf) | lesion (lesion-inside-leaf 0.44 -> 1.0000 everywhere)
CODE (only v9 copies, v8 files untouched): `kaggle/v9/cropguard-v9-train.py` (corpus assembly + gates + baseline const + picklable `_PhotoOnlyDataset`), `kaggle/v9-eval/cropguard-v9-eval.py` (v8-eval clone, 153 changed lines: OLD=v8 ckpt, NEW=v9 ckpt, both on corpus_v9), the two `kernel-metadata.json`, `data_pipeline/scripts/repair_masks_v9.py`, RUNLOG.

### Leftover bugs found (recheck pass)

BUG A — `TRAIN_SUBSET` truncated only the local list, never `ds.samples`, and the log line lied:
```python
# before (v8-inherited, kaggle/v9/cropguard-v9-train.py:666-669)
ds.samples = kept
if TRAIN_SUBSET:
    kept = kept[:TRAIN_SUBSET]
log(f"train: {len(kept)} images ({dropped} dropped by label audit)")
```
Evidence it fires: smoke log `[v9] train: 16 images (0 dropped by label audit)` but stage2 then trained the FULL list (class counts [80,537,773,856,751,290,682,487] = 4456 boxes = the whole 715-image train set; epoch 141s).
Impact on Kaggle: ZERO — Kaggle runs with `TRAIN_SUBSET=0`, the branch never fires; the fix is a no-op there (so the running kernel is behaviorally identical to this fixed file).
FIXED locally (only this hunk differs from the pushed version):
```python
ds.samples = kept
if TRAIN_SUBSET:
    ds.samples = ds.samples[:TRAIN_SUBSET]
log(f"train: {len(ds.samples)} images ({dropped} dropped by label audit)")
```
`python3 -m py_compile kaggle/v9/cropguard-v9-train.py` -> OK.

BUG B — the label-audit exclusion for TRAIN compares stems against the exclusions DICT (split keys), not `exclusions["train"]`, so nothing is ever dropped:
```python
# kaggle/v9/cropguard-v9-train.py:663-664 (identical to v8:507-508)
kept = [s for s in ds.samples
        if os.path.splitext(os.path.basename(s[0]))[0] not in exclusions]
```
`exclusions = {"train": {...32 stems...}, "val": {...5...}, "test": {...10...}}` (audit_labels line 418), so `stem not in exclusions` checks `{"train","val","test"}` -> always True.
Evidence: real v8 Kaggle log `[v8] train: 715 images (0 dropped by label audit)` x3 (stages 1,1,2) while stage0 audited `excluded=32`; smoke v9 same (`0 dropped`). Val/test paths use the dict correctly (`exclusions["val"]`, lines 810/985) which is why val logged `5 excluded by audit` and n=151.
DECISION: deliberately NOT fixed. v8's Kaggle run has the same behavior, so v8 and v9 both trained on all 715 train stems -> the v9-vs-v8 comparison stays single-variable (mask repair only). The 32 invalid stems carry cls=-1 and are skipped by the cls loss anyway (`valid = batch["cls"] >= 0`), they only contribute seg supervision. Fix for v10 (NOT before the v9-vs-v8 comparison concludes): `not in exclusions["train"]`.

Note: a second, separate exclusion mechanism DOES work: `leaf_exclusion_list.txt` (22 stems) is consumed by dataset.py at __getitem__ (sev_cont=-1, leaf channel -1) and setup_code copies it to /kaggle/working/code (v8 log: `leaf_exclusion_list.txt: 22 stems (.../cropguard-v7-diagnostics-input/code/leaf_exclusion_list.txt)`; the copy to the read-only `root/training/` fails with Errno 30 and is harmless because cwd is /kaggle/working/code).

### Status
- `kaggle kernels status ganrajlongadge/cropguard-v9-train` -> `KernelWorkerStatus.RUNNING` (poll log /tmp/v9kern_status2.log; monitor b2ybqr4nh).

## 2026-10-12 — scope decision (user): "everything in my model and accurate" -> detection + XAI added to the roadmap

User wants the full multi-task model measurable end to end. Feasibility check (evidence):
- dataset.py ALREADY collates detection targets in every batch: `"det_boxes": [b["det_boxes"] for b in batch]`, `"n_boxes": torch.tensor(...)` (dataset.py:375-376; per-sample at :361-362). The v9 run ignores them.
- The model forward already emits detection outputs when asked: `out["det_cls"]`, `out["det_box"]`, `out["det_sizes"]`, `out["anchors"]` (v7_model.py:108-138).
- The training loss already exists with YOLOv8-standard weights (det_cls 0.5, det_box 7.5, det_dfl 1.5): `multitask_loss(...)` + `DetectionLoss` + `tal_assign` + `ciou_loss` (losses.py:130/75/345-417).
=> v10 = v9 driver with `active_tasks={"det","cls","seg"}` + multitask_loss(two_channel_seg=True). One variable changed vs v9. Needs mAP@0.5 / mAP@0.5:0.95 measured on val+test (old v5-era server reference, NOT comparable to this lineage: mAP@0.5 0.4972->0.5101 flip-TTA on 1000 val).

XAI: LayerCAM exists in the app (`_multiscale_layercam`, model_server.py:242; `layercam`, inference.py:41) but targets the v5 CropGuardMultiTask. Plan: adapt hooks to CropGuardMultiTaskV7 (P3/P4/P5, gradient of the predicted class logit), run on val with the v9 checkpoint, and report a measured number (CAM energy inside GT lesion vs leaf vs random baseline). Server wiring = deployment change, ask user before touching the running app.

Sequence: v9-train (running, since 19:25) -> v9-eval (cls+seg+severity, OLD vs NEW) -> prepare+smoke v10 under way locally -> push v10 -> v10-eval (adds mAP) -> XAI local run with v9 ckpt -> report everything with raw logs.

## 2026-10-12 — v10 driver built + local smoke PASS (1.0 min); LayerCAM-for-V7 script + null self-test

v10 = v9 + detection head trained. Diff v9->v10 (`diff -u kaggle/v9/... kaggle/v10/...`) contains ONLY: docstring, `[v10]` prefix, `DET_W={"cls":0.5,"box":7.5,"dfl":1.5}` + `det_loss_parts()` (single DetectionLoss call, same as losses.multitask_loss det branch), forward `active_tasks={"det","cls","seg"}` in stage1+stage2, det loss added to `loss`/`total`, det sums + CSV columns + epoch-log `det=` field, artifact renames (best_v10_raw/ema.pt, last_v10.pt, v10_train_log.csv). py_compile OK.
Smoke (`V8_WORK=/tmp/v10smoke/work ... V8_STAGE1_STEPS=20 V8_FORCE_AUG=photo V8_TRAIN_SUBSET=16 V8_VAL_SUBSET=16 V8_EPOCHS=1 V8_CAP_MIN=6 python3 kaggle/v10/cropguard-v10-train.py`, log /tmp/v10smoke/smoke1.log, exit 0):
```
[v10]   step   20: lesion IoU=0.3419 cls=0.3556 seg=3.3417 total=18.8230 ...
[v10] train: 16 images (0 dropped by label audit)      <- TRAIN_SUBSET fix confirmed working (was 715)
[v10] E  1/1 [7s] cls=2.0992 seg=5.7173 det=5.139/0.764/8.514 | RAW ... | ema_moved=True
[v10] TOTAL 1.0 min; stage states: {... 'stage0': 'ok', 'stage1': 'photo', 'stage2': 'ok'}
```
Artifacts: best_v10_raw.pt (9.8 MB), last_v10.pt (39 MB), v10_train_log.csv written.

LayerCAM XAI for the V7 family: NEW `model/training/layercam_v7.py` — ports the server's `_multiscale_layercam` (P3+P4+P5, ReLU(grad) element-wise weights, predicted-class logit) from the v5 class to `CropGuardMultiTaskV7` via a neck forward-hook + enable_grad scope; reports measured CAM energy fraction inside GT lesion/leaf with lift = cam_frac/area_frac (random map => lift ~1).
Null self-test (random weights, /tmp/camtest): lesion_lift=1.0295, leaf_lift=1.0088 (4 val images) — exactly the expected "no signal" behavior, so the metric is calibrated. Real numbers pending the v9 checkpoint download.

## 2026-10-12 20:15 — v10-train PUSHED; v10-eval built + smoke launched; kernel statuses

`kaggle kernels push -p kaggle/v10` -> version 1. NOTE the title-derived slug: the kernel was created as `ganrajlongadge/cropguard-v10-train-detection-enabled` (title "CropGuard v10 Train (detection enabled)"); the console warned "title does not resolve to the specified id". Local `kernel-metadata.json` id updated to the actual slug so future pushes update this same kernel. Status: QUEUED (monitor /tmp/v10kern_status.log). v9-train still RUNNING at 20:17:15 (poll log /tmp/v9kern_status2.log); v8 reference time was 62.8 min (started 19:25) -> ETA 20:25-20:35.

v10-eval built: `cp kaggle/v9-eval/cropguard-v9-eval.py kaggle/v10-eval/cropguard-v10-eval.py` + edits, all reviewed:
- OLD = cropguard-v9-train mount (best_v8_ema/raw.pt, last_v8.pt — v9 driver kept v8 artifact names); NEW = cropguard-v10-train mount (best_v10_ema/raw.pt, last_v10.pt).
- Detection section ported from model/training/eval_tta.py (same math, no torchmetrics): `nms`, `batch_decode` (dfl_decode + conf 0.25 + max 100), `merge_tta` (flip), `ap_from_conf_flags` (PR-curve AP), `det_metrics` (class-aware best-IoU matching at 10 IoU thresholds). predict() now runs `active_tasks={"det","cls","seg"}` and collects dets with AND without flip-TTA; run_set attaches `det_no_tta` / `det_tta` into each set's metrics; results.md gains 4 mAP rows.
- ONNX: OnnxWrap now returns 4 outputs (cls_logits, seg_logits, det_cls, det_box); agreement check per-output; filenames cropguard_v10_fp32/int8.onnx.
- Verdict gains an explicit "detection clears the untrained-head baseline by >0.05 mAP@0.5 on test" line, else a NOT-supported line.
`python3 -m py_compile kaggle/v10-eval/cropguard-v10-eval.py` -> OK.
Local CPU smoke launched (background id b1zi16o7y, log /tmp/v10eval_smoke.log):
`V10E_WORK=/tmp/v10eval_smoke V9_CORPUS=/tmp/v9smoke/corpus_v9 V8_CODE_ROOT=$PWD/model V8_INPUT_ROOT=/tmp/v9smoke/input/v7 V9E_V9TRAIN=/tmp/v9smoke/work V10_TRAIN_OUT=/tmp/v10smoke/work python3 -u kaggle/v10-eval/cropguard-v10-eval.py`
using the 1-epoch smoke checkpoints (v9 smoke = OLD, v10 smoke = NEW); PlantDoc absent locally -> expected "not measured". Results appended below when it exits.

### v10-eval smoke RESULT (exit 0, 3.2 min, log /tmp/v10eval_smoke.log)
All code paths executed on the 1-epoch smoke checkpoints (numbers are garbage on purpose; this validates plumbing only):
- both ckpts load missing=0 unexpected=0; val=156/test=150; PlantDoc "not measured" as expected.
- det rows: `old/plantseg_val det: mAP@0.5=nan ... (gt=959 boxes, pred=0)`; same for new and test. DIAGNOSED, not a decode bug: raw max det confidence on val is 0.0104 (NEW) / 0.0149 (OLD) — the det head's bias init is the YOLO 1%-prior (logit ~-4.6), and 1 epoch barely moves it; zero boxes above conf 0.25 is CORRECT. Verified directly: `NEW(v10 smk) img 0 max conf=0.0104 mean=0.0100 n>0.25=0 anchors=2100`.
- ONNX FP32 10.23 MB wrote; 4-output agreement vs PyTorch over 20 imgs: cls 1.5e-08, seg 1.0e-07, det_cls 6.2e-06, det_box 1.5e-06. INT8 3.13 MB wrote; int8_val metrics computed.
- CPU latency (1 thread, i5-11400H — NOT a phone): fp32 median 74.37 ms, int8 median 67.90 ms.
- AP-port equivalence: /tmp/ap_equiv_check.py imports the REAL eval_tta.ap_per_class and compares against ap_from_conf_flags on 5x10 random (conf, flag) sets -> worst |diff| = 0.000e+00 (PASS), edge cases nan==nan.
- HEALTHY-SHORTCUT CHECK FIRED (this is the pre-existing leakage probe in the eval, unchanged by me): image statistics alone (size/aspect/RGB/green_frac) predict "healthy" with val-holdout acc 0.9841 (n=156, 60/40 split; fit 0.9677). Flagged=true. This is a property of the corpus/split, not of the model — must be disclosed in the paper (healthy may be partly decidable from acquisition cues).
2026-10-12 20:15 KERNEL STATUS: v9-train RUNNING (poll 20:22:25), v10-train QUEUED (poll 20:22:44) — GPU concurrency=1, so v10 starts only after v9 finishes.

### LayerCAM de-risk run with the REAL v8 checkpoint (local CPU, 40 val images)
`python3 -u model/training/layercam_v7.py --ckpt /home/logiclocus/tmp/kout_v8/best_v8_raw.pt --data /tmp/v9smoke/corpus_v9 --split val --n 40 --out /tmp/camtest_v8real`
```
{"n": 40, "n_with_lesion": 40, "n_with_leaf": 40,
 "lesion_cam_energy_frac_mean": 0.2974, "lesion_area_frac_mean": 0.1608, "lesion_lift": 1.8501,
 "leaf_cam_energy_frac_mean": 0.7680, "leaf_area_frac_mean": 0.6495, "leaf_lift": 1.1825}
```
vs null (random weights): lesion_lift 1.03 / leaf_lift 1.01. The trained model's CAM concentrates on lesion tissue 1.85x above chance — script validated with real weights. Numbers are from the v8 (pre-repair) model; the reportable run will use the v9 checkpoint once v9-train output is downloaded.

## 2026-10-12 21:45 — v9-train runtime anomaly + code check (no polling)
- User-pasted Kaggle UI: "CropGuard v9 Train / Running for 7723.1s / GPU T4 x2 / Output 0 B" (~2h09m; v8 reference total = 62.8 min on Tesla T4).
- Background status monitors stopped (pkill; pgrep shows none).
- Code check of kaggle/v9/cropguard-v9-train.py (v10 shares identical structure):
  - L51 CAP_MINUTES=55 and L805 t_start: cap applies to stage2 ONLY; stage0/1/3 have no wall-clock guard.
  - L460 subprocess.run(test_dataset_aug.py) has NO timeout= -> can hang stage0 indefinitely.
  - L522-524 DataLoader num_workers=2, persistent_workers=True, no timeout= -> a stuck worker blocks forever.
  - L664 Bug B confirmed: `not in exclusions` (dict) -> 0 train stems dropped; kept for v8 parity.
  - No definitive bug found that explains +65 min; Kaggle log (Logs tab) needed to localise.
- Proposed hardening (NOT applied yet): global deadline checked in every stage loop, subprocess timeout=600, DataLoader timeout=300 + persistent_workers=False.

## 2026-10-12 21:50-22:40 IST — v9 overrun: diagnosis, hardening applied to v9+v10, both smoke-PASS

### What was measured (no polling)
- `kaggle kernels status ganrajlongadge/cropguard-v9-train` -> RUNNING, started 13:54:38 UTC; at 16:18 UTC elapsed = 2 h 24 min (v8 reference = 62.8 min).
- While it was running its log was unobtainable BY DESIGN, proven with a control:
  - v9 `list_kernel_session_output` -> `log` = 0 bytes, `files` = 0.
  - v9 `GET api.kaggle.com/v1/kernels/logs/stream/...` -> **HTTP 500 after ~61 s**; same on `www.kaggle.com` (raw probe `/tmp/probe_v9_log2.py`).
  - CONTROL v8 (COMPLETE) -> `log` = 55336 bytes, 10 files (best_v8_raw.pt, stage0_summary.json, v8_train_log.csv, ...). Same endpoint, so the 500 is not a broken API: Kaggle only publishes a session's log/output when the session ENDS.
- After the user cancelled it: status CANCEL_ACKNOWLEDGED, log still empty (Kaggle discards logs for cancelled runs). No further evidence available from Kaggle.

### Why this is a hang, not something to wait out
- stage2's cap check runs at the TOP of each epoch and `per_epoch` only exists once `epochs_done >= 1` (incremented at each epoch's END), so stage2 stops at ~CAP_MINUTES + one epoch.
- `CAP_MINUTES = 55`. v8's own log shows the mechanism working: `TIME CAP: 54.7 min used, ~0.54 min/epoch, cap 55.0 min -> stopping cleanly at epoch 102`.
- So stage2 cannot explain 2 h 24 min. The overrun is stage 0, stage 1, or **stage2 epoch 1 never finishing** (no cap check has fired yet) -> a silent hang. Waiting would not have produced a result.

### Fix applied to BOTH `kaggle/v9/cropguard-v9-train.py` and `kaggle/v10/cropguard-v10-train.py`
Runtime-only; the training math is untouched (v8/v9/v10 comparability preserved).
1. `GLOBAL_DEADLINE_MIN` (default 100) enforced by new `start_watchdog()`: a daemon thread that force-exits with a `stage_deadline_exceeded.json` marker. A separate thread is the only way to break a C-level block (stuck worker / subprocess).
2. `subprocess.run(test_dataset_aug.py, ...)` -> `timeout=SUB_TIMEOUT_S` (600); `TimeoutExpired` becomes a clear RuntimeError instead of a stage-0 hang.
3. `loader_for` -> `timeout=LOADER_TIMEOUT_S` (300) + `persistent_workers=False`; worker count via `NUM_WORKERS` (2).
4. Deadline re-checks in the stage1/stage2 loops, a stage2 per-200-step progress line, and a 30 s watchdog heartbeat line.

### Verification (raw)
- `python3 -m py_compile kaggle/v9/... kaggle/v10/...` -> OK. The hardening block is byte-identical between v9 and v10 (only line offsets differ).
- `/tmp/test_hardening.py` on BOTH drivers:
  ```
  [A] loader_for timeout=7 persistent_workers=False -> PASS
  [A2] workers=0 timeout=0 persistent_workers=False -> PASS
  [B] geometry hang killed after 3.0s -> PASS: forced-geometry test hung: no exit after 3s -- aborting instead of blocking
  RESULT: ALL PASS
  ```
- `/tmp/test_watchdog.py` (v9): `rc=1 elapsed=30s`; marker written `{"ok": false, "elapsed_min": 0.5, "state": {}}`.
- Full local smoke on the REAL corpus (`V8_INPUT_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v7_split`, `V9_FILLED_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v9_fixed`):
  - v9: `EXIT=0`, `TOTAL 1.0 min; stage states: {..., 'stage0': 'ok', 'stage1': 'photo', 'stage2': 'ok'}`; corpus gate re-measured `overlay_median_ratio=20.40x`, `lesion_inside_leaf_min=1.0`; geometry gate PASS; pretrained mode=pretrained L1=0.238369.
  - v10: `EXIT=0`, `TOTAL 1.1 min; ... 'stage2': 'ok'`; `E 1/1 [9s lr=5.00e-05] cls=2.0992 seg=5.7173 det=5.139/0.764/8.514`.
- Both smokes still print `train: 16 images (0 dropped by label audit)` -> **Bug B still present on purpose** (v8/v9 parity); its fix stays deferred as planned.

### Re-push (hardened)
- Kaggle state before push: v9-train and v10-train both `CANCEL_ACKNOWLEDGED` (no un-hardened run left alive).
- `kaggle kernels push -p kaggle/v9`  -> `Kernel version 2 successfully pushed.`
- `kaggle kernels push -p kaggle/v10` -> `Kernel version 2 successfully pushed.`
- Status right after: v9_train = QUEUED, v10_train = QUEUED (GPU concurrency 1 -> v9 runs first, v10 follows).

**Standing-rule caveat:** a Kaggle batch kernel publishes its log/output ONLY at session end (proved above with the v8 control), so "show the log growing" is impossible for a running Kaggle job. The watchdog heartbeat + `stage_deadline_exceeded.json` marker exist precisely so that the NEXT run is self-documenting: it cannot hang past `GLOBAL_DEADLINE_MIN` (100), and if it does stop early it will leave a marker naming the stage it died in.
- Queue state at 16:36:22 UTC: v9-train = QUEUED, v10-train = QUEUED (both version 2). Not yet RUNNING; stopped polling to respect the no-long-sleep-polling rule. Next check is a single bounded status read.

### 16:37-16:51 UTC — queue latency + background monitor
- Bounded checks 16:37:05 / 16:38:22 / 16:39:39 / 16:40:55 / 16:42:11 / 16:43:28 -> v9 still QUEUED. `kaggle kernels list` shows v9 `lastRunTime` still 2026-10-12 13:54:38.867 and v10 14:42:06.323 (i.e. the v2 runs have NOT been scheduled yet; `lastRunTime` updates when a run actually starts). This is queue latency, not an error - v10 v1 also sat QUEUED ~1.5 h behind v9 earlier tonight.
- Set up `/tmp/monitor_v9.sh` (nohup, pid recorded, hard stop after 4 h) -> appends status transitions to `/tmp/v9_monitor.log` and auto-downloads v9 output to `/home/logiclocus/tmp/kout_v9` when v9 leaves QUEUED/RUNNING. First lines:
  ```
  2026-10-12 16:46:07 monitor start (pid 83388)
  2026-10-12 16:46:08 v9=[...QUEUED] v10=[...QUEUED]
  ```
- 16:51 UTC recheck: no transition yet (monitor alive, logs changes only).

## 2026-10-12 22:35-23:05 IST — Detailed research paper written (Markdown)

`resources/research_papers/CropGuard_Detailed_Research_Paper.md` — NEW, 1581 lines / 12,853 words / 98.6 KB, 23 sections, 84 balanced code fences (ASCII block diagrams), 147 table rows.
Coverage: convolution from first principles (kernel/stride/padding/receptive field/FLOPs), BN, ReLU vs GELU vs Hardswish (and why the neck uses GELU while the heads use Hardswish for INT8), residual paths, depthwise-separable + PConv; full ASCII block/process diagrams (end-to-end system, training stages, inference+export, task-to-trunk fan-out, BiFPN topology, seg decoder layer-by-layer, lesion-gated cls, DFL decode, abstention); complete loss mathematics (Focal-Tversky+BCE pos_weight=20, TAL assignment, DFL two-bin CE, CIoU, class weights, PCGrad, consistency); data pipeline + augmentation; staged training; eval protocol incl. the healthy-shortcut probe; results; "why this particular combination" (11 subsections); reliability engineering (the hang diagnosis + hardening); 12 ranked limitations; deployment; reproducibility; provenance map + "what to add when pending runs land".

Every number is tagged `[M]` measured / `[P]` pending / `[N]` not measured. Newly MEASURED in this step (not previously in any doc):
```
backbone  2,214,900  87.43%
neck        146,592   5.79%
det_head     24,616   0.97%
cls_head     65,480   2.58%
seg_head     81,666   3.22%
abstain           0   0.00%
TOTAL     2,533,254            (shared trunk backbone+neck = 93.22%, heads = 6.78%)
shapes @320: seg_logits(1,2,320,320) cls_logits(1,8) det_cls(1,2100,8) det_box(1,2100,32) anchors(2100,3)
```
Verification: `wc` line/word count; code-fence count 84 (even -> balanced); h2 count 23; arithmetic re-checked in Python (`total == 2533254 -> True`). One error was found and fixed during verification: an initial "96.6 % shared trunk" was wrong (backbone-only figure mislabelled as backbone+neck); corrected to **93.22 %** in both places it appeared.
No run status changed: v9-train and v10-train still QUEUED at 17:00 UTC (~28 min); monitor still alive.

## 2026-10-12 22:35-22:45 IST — v9/v10 RE-PUSHED as version 3; both RUNNING; GPU quota verified OK

### Provenance check (user asked explicitly)
CONFIRMED: the previous AI's v9/v10 changes were **accepted and kept**, not overwritten. Verified by grep on the current files:
```
kaggle/v9/cropguard-v9-train.py : prepare_corpus=4 _PhotoOnlyDataset=5 find_filled_root=2 V9_BASELINE_MAE=4 FILLED_ROOTS=3 lesion-inside-leaf=1
kaggle/v10/cropguard-v10-train.py: same, plus DET_W=7 det_loss_parts=3 active_tasks={"det","cls","seg"}=2
```
The hardening was applied **on top of** their files with `str_replace`; nothing was reverted to the v8 codebase.

### Why the runs stalled at QUEUED (root cause found)
- `kaggle quota`: **GPU 6.12 h used / 23.88 h remaining of 30 h** (refresh 2026-10-17), TPU 20.00 h untouched. **Quota is NOT the blocker** — this was verified rather than assumed.
- Both kernels sat in `CANCEL_ACKNOWLEDGED` with `lastRunTime` still 13:54:38 / 14:42:06, i.e. the version-2 runs had never been scheduled. The stuck cancelling session was holding the slot.
- After the stuck state cleared, `kaggle kernels push -p kaggle/v9` -> version 3 -> status **RUNNING at 17:09:49 UTC**, ~1 s after push. v10 version 3 -> **RUNNING at 17:10:07 UTC**. So **both GPUs run concurrently** (the earlier "GPU concurrency = 1" note in the handoff was wrong).

### Bug review performed this step (files read, findings)
1. `_paste_background` (dataset.py L185-208) is O(h*w), no hang. **v9's whole-leaf mask actually FIXES it**: in v8 the leaf mask was healthy-tissue-only, so `cut[lm==0]=0` zeroed the lesion region before pasting. Not a bug in v9.
2. `test_dataset_aug.py` touches only 20 samples + a synthetic square -> cannot be a long hang source. It is now bounded by `timeout=SUB_TIMEOUT_S` anyway.
3. `prepare_corpus()` fragility: if the corpus already exists AND the v7 outline root is absent, `ratios` stays empty, `med = nan`, `not (nan >= 5.0)` is True -> raises "lesion fill overlay did NOT take effect (nanx)". Cannot trigger on Kaggle (fresh `/kaggle/working` each session) but is a misleading error message. Left as-is (fails closed, which is the safe direction).
4. `dataset.__getitem__` uses `np.bincount(..., minlength=11)` on 8 classes — the padding does not affect `argmax`, so it is cosmetic, not a bug (and it matches the driver's `NC + 3`).
5. **Bug B re-confirmed, deliberately NOT fixed**: `build_train_ds` compares the stem against the exclusions **dict** instead of `exclusions["train"]`, so 0 train images are dropped (smoke prints `train: 16 images (0 dropped by label audit)`). Harm is bounded and arguably positive: those images have `cls = -1`, so they are masked out of the classification loss while still supplying segmentation supervision. v8 already ran with this behaviour, so changing it NOW would break the v8-vs-v9 controlled comparison. Fix belongs *after* the comparisons.
6. Original 2 h 24 min hang root cause: **still undetermined** (no log was obtainable, and the local smoke passes). The hardening is the mitigation, not a proven cure.

### Housekeeping
- Stale `monitor_v9.sh` was dead (a `nohup … &` launched inside a SYNC command does not survive tool cleanup) — that is why it never logged the cancel. Relaunched via the BACKGROUND execution mode so it is properly managed.
- `kaggle quota` and `kernels_api_service.cancel_kernel_session` both exist in this CLI/SDK version (useful: cancellation IS scriptable, unlike what was assumed earlier).

---

## 2026-10-12 19:14 UTC — v9/v10 train COMPLETE, outputs downloaded, both eval kernels pushed

### Kernel status (queried, not assumed)
```
cropguard-v9-train                     COMPLETE   failure_message = ""
cropguard-v10-train-detection-enabled  COMPLETE   failure_message = ""
cropguard-v9-eval                      RUNNING    (version 1, pushed 19:13 UTC)
cropguard-v10-eval                     RUNNING    (version 1, pushed 19:13 UTC)
cropguard-v8-train                     COMPLETE   (OLD side of the v9-eval comparison)
```
Owner slug is `ganrajlongadge`, **not** `logiclocus` — the first status query this step failed with "Permission 'kernels.get' was denied" purely because of the wrong owner in the slug. Not a bug in the kernels.

### Clean-run evidence (from the downloaded logs)
```
v9  59.5 min  watchdog armed (deadline 100 min)  stage states: stage0=ok stage1=geo stage2=ok
v10 60.0 min  watchdog armed (deadline 100 min)  stage states: stage0=ok stage1=geo stage2=ok
```
- No `stage_deadline_exceeded`, no timeout, no traceback in either log. The hardening was armed but never had to fire.
- Both hit the repaired-corpus gates on the real Kaggle corpus: `overlay_median_ratio=20.400`, `lesion_inside_leaf mean=1.0000 min=1.0000`, dataset `test_dataset_aug.py` part B `n=20 PASS`.
- Both built `corpus_v9` from the same inputs (`cropguard-v7-diagnostics-input` + `cropguard-v9-filled-masks`), so v9 vs v10 is a controlled comparison.

### Metrics (val, 151 valid cls/sev samples) — v8 vs v9 vs v10

| metric | v8 (outline masks, pre-repair) | v9 (repaired masks) | v10 (repaired + detection) |
|---|---|---|---|
| acc | 0.6954 | 0.7152 | **0.7285** |
| macro-F1 | **0.6786** | 0.6601 | 0.6662 |
| lesion IoU | 0.0459 | **0.4582** | 0.4137 |
| lesion Dice | 0.0878 | **0.6284** | 0.5853 |
| sev MAE (pred masks) | 0.1026 | **0.1509** | 0.2486 |
| mean true lesion area | 621.3 | 15670.4 | 15670.4 |
| pred/true area ratio | 7.61x | **1.39x** | 2.00x |
| constant-sev baseline | 0.0384 | 0.2293 | 0.2293 |
| beats constant baseline | false | **true** | false |
| selected epoch | 35 | 79 | 40 |

### Headline finding — the mask repair is what mattered
Segmentation went from **IoU 0.046 -> 0.458 (10x)** and **Dice 0.088 -> 0.628 (7.2x)**, with over-prediction collapsing 7.61x -> 1.39x. The v8 `mean_true_lesion_area = 621 px` vs v9 `15670 px` confirms the v8 targets really were thin outlines, so this is a target-quality fix, not a model-capacity change. v9 is also the **only** version whose severity beats its constant baseline (0.1509 < 0.2293), where v8 failed (0.1026 > 0.0384).

### Honest negative result — v10 is NOT a win
Enabling detection bought a little classification (acc 0.7152 -> 0.7285, macro-F1 +0.0061) but **cost segmentation (IoU -0.0445, Dice -0.0431) and severity (0.1509 -> 0.2486, i.e. it stops beating the constant baseline)**, and over-prediction roughly doubled (1.39x -> 2.00x). v10's best checkpoint was epoch **40 of 79** while v9's was the final epoch 79 — consistent with the detection loss destabilising the later epochs. **Recommendation: v9 is the current best model on the segmentation/severity axis; do not promote v10 without a re-run or a det-loss weight sweep.** Do not report v10 as an improvement.

### Artifact naming (checked, intentional — not a mix-up)
`cropguard-v9-train` writes **v8-named** files (`best_v8_raw.pt`, `last_v8.pt`, `v8_train_log.csv`) — documented deliberately in the v9 script's module docstring L10 ("Artifact filenames keep the v8 names"). v10 writes `*_v10*`. This is safe because each eval resolves OLD/NEW by **mount root**, not filename:
```
v9-eval : OLD = /kaggle/input/cropguard-v8-train      NEW = /kaggle/input/cropguard-v9-train
v10-eval: OLD = /kaggle/input/cropguard-v9-train      NEW = /kaggle/input/cropguard-v10-train-detection-enabled
```
Both candidate lists match what the runs actually produced (`best_v8_ema.pt` absent in both v8 and v9 -> falls through to `best_v8_raw.pt`, which is the `selected:raw` checkpoint named in `stage_summary.json`).

### Checkpoint integrity — verified locally before spending GPU
Replicated the eval's exact `load_ckpt` path (`torch.load(..., weights_only=False)` -> `obj["model"]` -> `CropGuardMultiTaskV7(nc=8).load_state_dict(..., strict=False)`):
```
v9  best_v8_raw.pt   missing=0 unexpected=0 epoch=79 which=raw
v9  last_v8.pt       missing=0 unexpected=0 epoch=79
v10 best_v10_raw.pt  missing=0 unexpected=0 epoch=40 which=raw
v10 last_v10.pt      missing=0 unexpected=0 epoch=79
```
All four are valid and architecture-matched, so the eval kernels will not silently fall into their `not_measured` branch.

### Downloads and quota
- `/home/logiclocus/tmp/kout_v9` (49 MB) and `/home/logiclocus/tmp/kout_v10` (50 MB); incl. `stage_summary.json`, `stage0_*`, `stage1_memorize.json`, `stage2_training.json`, train CSVs and logs.
- GPU quota before the eval pushes: **8.11 h used / 21.89 h remaining of 30 h** (refresh 2026-10-17).
- Eval scripts re-compiled clean (`py_compile` exit 0 both) and **unchanged** since 19:28 / 20:16 (the train drivers are the ones that changed at 21:55 local).
- Background monitor `/tmp/monitor_eval.sh` (pid 152215) logs status transitions and auto-downloads eval output to `kout_v9eval` / `kout_v10eval` on completion; 150 min hard stop.

---

## 2026-10-12 19:20-19:50 UTC — v9-eval / v10-eval: 3 real bugs found, fixed and regression-tested

The eval kernels went COMPLETE four times before producing something trustworthy. Each COMPLETE was NOT a success; the artifacts had to be opened each time. Full chain below.

### Run 1 (eval v1) — silent total failure: wrong mount paths for `kernel_sources`
Both kernels COMPLETE in ~70 s. `results.json` was ~2 KB and read `not_measured: ["OLD checkpoint not found", "NEW checkpoint not found (training output mount)", "ONNX export skipped (no NEW checkpoint)"]`, with every metric `not measured`.

The resolvers only knew the flat `/kaggle/input/<slug>` and `/kaggle/input/datasets/<owner>/<slug>` forms. A no-GPU probe kernel (`kaggle/probe-mounts/`) with the identical source lists returned the real layout:
```
/kaggle/input/datasets/ganrajlongadge/cropguard-v9-filled-masks   
/kaggle/input/notebooks/ganrajlongadge/cropguard-v8-train   best_v8_raw.pt, last_v8.pt, ...
/kaggle/input/notebooks/ganrajlongadge/cropguard-v9-train   best_v8_raw.pt, last_v8.pt, ...
top level of /kaggle/input = ['datasets', 'notebooks']
```
**`kernel_sources` mount under `/kaggle/input/notebooks/<owner>/<slug>`.** Every hardcoded candidate returned `isdir=False`. Note this failure mode is silent: exit status 0, `KernelWorkerStatus.COMPLETE`, a well-formed report, and `not measured` everywhere.

**Fix** (`kaggle/v9-eval`, `kaggle/v10-eval`): `/notebooks/...` first in each root list (flat paths kept as fallbacks); a new `_find_ckpt_dir(slug_hint, roots, filenames)` that accepts a directory only if it actually CONTAINS a checkpoint and breaks ties on the slug in the path (v8-train and v9-train both ship `best_v8_raw.pt`, so path -- never filename -- is the discriminator); a bounded `/kaggle/input` walk as a layout-independent fallback; and `log_pt_inventory()` which dumps the input top level and every `.pt` found whenever a mount is missing, so the next layout change is diagnosable from the log.

**Regression test `kaggle/test_eval_mount_resolution.py`** (12 assertions, PASS): extracts the real functions via AST and runs them against a simulated tree through an `os` shim. Part A `notebooks/` layout *plus empty decoy dirs at the flat paths*; Part B empty dirs only -> must return `None` (this is the case the old `if os.path.isdir(base): return base` got wrong); Part C checkpoints at an unlisted path -> walk fallback must find them and out-rank a same-depth `unrelated-archive` decoy holding an identically named file.

### Run 2 (eval v2) — mounts fixed, real metrics, ONNX still broken
Every dataset and both checkpoints resolved. ONNX export failed with
```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```
`load_ckpt()` moves the model to CUDA, but `dummy = torch.randn(1, 3, SIZE, SIZE)` was built CPU-only, so tracing died on the first conv.

**Fix**: derive `dev = next(model.parameters()).device`, build `dummy` on it, and keep a CPU re-export as a fallback inside a `try/except` (ONNX is device-neutral). The numeric-agreement loop also needed `x.to(dev)` for the PyTorch reference forward and `.detach().cpu().numpy()` for the comparison -- the same bug, second half.

**Regression test `kaggle/test_onnx_export.py`** (PASS): runs the real `export_onnx` on CPU with the real v9 checkpoint (fp32 10.12 MB, INT8 3.03 MB, max |ONNX - PyTorch| 3.5e-4) and structurally pins that the tracing input is built from the model's device, so a re-hardcoded CPU `randn` fails the test even without a GPU.

### Run 3 (eval v3) — FP32 written, INT8 lost
`cropguard_v9_fp32.onnx` appeared (10.1 MB) but the INT8 graph did not:
```
ModuleNotFoundError: No module named 'onnxruntime'
```
onnxruntime is **not** in the base Kaggle image -- which is exactly why the v7 drivers pip-install it (`kaggle/cropguard-v7-train.py` L133-139). The v8/v9/v10 eval scripts never had that step.

**Fix**: `ensure_onnxruntime()` in both eval scripts, called only when a NEW checkpoint exists: no-op if the import already works, otherwise `pip install -q onnxruntime onnx`, then re-import and log the outcome. Honours `NO_ORT=1` to match the v7 drivers' escape hatch. `test_onnx_export.py` also asserts the guard behaves (`NO_ORT=1 -> False`, ort present -> `True`).

### Run 4 (eval v4) — clean: `not_measured: []` for both
FP32 + INT8 for both versions, latency measured, nothing unmeasured.

### Run 5 (eval v5) — honesty defect in the generated verdict
Not a crash: the verdict prose said `"severity beats the baseline by a clear margin"` whenever `sev_mae < constant_baseline`, with **no threshold at all**. For v9 that is fair (0.1176 vs 0.2343 = 49.8% relative). For v10 it is an overstatement: 0.2134 vs 0.2343 = **8.9%**, and the same model **FAILS** on val (0.2538 vs 0.2098). The script's own docstring promises "results.md is generated FROM results.json, so the prose and the numbers cannot disagree" -- an unqualified "clear margin" broke that promise.

**Fix**: require `>=10%` relative improvement for "clear margin", say "only X% (below the 10% relative bar) -> NOT reliably better" otherwise, and add the val-side comparison with an explicit "test/val disagree, so treat the severity win as unresolved" when the two splits point opposite ways. Verified by executing the real verdict block against the real numbers:
```
v9 : beats the baseline by 49.8% (>=10% relative, a clear margin). / val also beats the baseline.
v10: beats the baseline by only 8.9% (below the 10% relative bar) -> NOT reliably better. / val does NOT beat the baseline -- test/val disagree.
```

### Final numbers (eval v4/v5, val = 151 and test = 140 PlantSeg images)

| metric (test) | v9 | v10 |
|---|---|---|
| top-1 | 0.6929 | 0.7357 |
| macro-F1 | 0.6471 | 0.6977 |
| lesion IoU | **0.5114** | 0.4209 |
| lesion Dice | **0.6767** | 0.5925 |
| sev MAE (pred masks) | **0.1176** | 0.2134 |
| constant baseline | 0.2343 | 0.2343 |
| beats baseline | yes, 49.8% | only 8.9%, and fails on val |
| over-prediction | 1.37x | 1.87x |
| mAP@0.5 | n/a (head untrained) | 0.0225 |
| ONNX fp32 / int8 | 10.12 / 3.03 MB | 10.23 / 3.13 MB |
| max \|ONNX-PyTorch\| | 7.6e-06 cls, 7.2e-05 seg | ~8e-06 all four outputs |
| CPU latency fp32 / int8 | 70.4 / 81.8 ms | 69.9 / 86.2 ms |
| INT8 val acc | 0.7219 | 0.6887 |

Note the OLD-vs-NEW deltas the script computes are all **within CI noise** (e.g. v9 test top-1 -0.0071 against a NEW CI half-width of 0.0715). The segmentation gain is the large, unambiguous one: v8 IoU 0.0761 -> v9 0.5114 on test.

### Verdict on v10 (honest)
V10 buys classification (+0.0429 test top-1, still inside CI) and **costs segmentation** (Dice 0.6767 -> 0.5925, IoU 0.5114 -> 0.4209), doubles over-prediction, and its detection head reaches only mAP@0.5 = 0.0225 against an untrained-head `nan` baseline -- the script itself now concludes "detection claim not supported yet". **Do not promote v10 as an improvement.** Keep v9 as the best segmentation/severity model.

### Artifacts archived and hashed
- `model/weights/v9/` and `model/weights/v10/`: checkpoint, `last_*`, both ONNX graphs, `stage_summary.json`, train CSV, `results.json`, `results.md`.
- `model/weights/WEIGHTS_MANIFEST.json`: **18 -> 26 entries** (checkpoint + last + both ONNX per version) with md5 and byte counts computed from the archived files, and metrics read out of `results.json` rather than retyped. Caught and corrected two wrong key names during the write (`sev_mae_predmasks`/`sev_const_baseline_mae` do not exist; the real keys are `sev_mae_pred_masks`/`sev_mae_constant_baseline`), which had silently produced `nan`/`None`.
- Nothing is staged or committed: `model/weights/*.pt` is gitignored and the v7 ONNX files are untracked, so the blobs stay out of git as the repo's README intends.

### Local checks that passed (files unchanged since)
- `py_compile` clean on both eval scripts after every edit.
- `kaggle/test_eval_mount_resolution.py` -> `OVERALL: PASS` (12 assertions).
- `kaggle/test_onnx_export.py` -> `OVERALL: PASS` (device rule, NO_ORT guard, real-weight FP32+INT8 export, numeric agreement).

### Run 5 artifacts downloaded and archived (2026-10-13 01:26 UTC)
`cropguard-v9-eval` v5 and `cropguard-v10-eval` v5 both COMPLETE with `not_measured: []`. Outputs pulled with the Python SDK (`api.kernels_output`) after the **CLI `kaggle kernels output` stalled ~7.5 min with 0 CPU time** on the v9-eval download; the SDK returned the same files instantly. Worth remembering as the first thing to try when a download hangs.

Reproducibility check before overwriting the archive -- compared the v5 `results.json` against the archived v4 one field-by-field:
```
v9 : 4 differing fields, ALL of them onnx.cpu_latency_* (70.4/74.7 -> 66.7/71.9 ms fp32, 81.8/85.1 -> 78.1/82.0 int8)
v10: 12 differing fields = the same 4 latency fields + 8 `nan != nan` false positives in the OLD (untrained-head) detection block
```
Every accuracy, segmentation, severity, calibration, ONNX-size, ONNX-agreement and INT8-val metric is **identical across runs** -- the eval is deterministic and only the CPU benchmark timing moves. The `.pt` and both `.onnx` artifacts are **byte-identical** (md5 unchanged), so only `results.json` / `results.md` / the eval log were refreshed and the manifest hashes stayed valid.
- `model/weights/WEIGHTS_MANIFEST.json`: the 8 v9/v10 entries' `source` strings now name **eval version 5 (final; mount paths fixed, ONNX+INT8 produced)** instead of version 4.
- The archived `results.json` now carries the corrected verdict prose: v9 "beats the baseline by 49.8% (>=10% relative, a clear margin) / val also beats the baseline"; v10 "only 8.9% (below the 10% relative bar) -> NOT reliably better / val does NOT beat the baseline -- test/val disagree, so treat the severity win as unresolved".

### `.gitignore` gap closed (found while checking what would be committed)
The rule was only `model/weights/*.pt`, so the per-version subdirs were **not** covered: `git check-ignore` returned nothing for `model/weights/v9/*.pt` and `model/weights/v10/*.onnx`, and `git status` listed all three new dirs (v7 92 MB, v9 62 MB, v10 62 MB) as untracked. A plain `git add -A` would have committed **~250 MB of blobs**, contradicting `model/weights/README.md` ("the `.pt` blobs are gitignored (≈1 GB); WEIGHTS_MANIFEST.json is committed").

Extended to every depth, deliberately keeping the small text records addable:
```
model/weights/*.pt
model/weights/**/*.pt
model/weights/**/*.onnx
model/weights/**/*.onnx.bak
```
Verified: no file >1 MB under `model/weights/` is now addable, while `results.json`/`results.md`/`metrics_*.json`/`stage_summary.json`/`*_train_log.csv`/`README.md` and `WEIGHTS_MANIFEST.json` are all still tracked-able. Also caught `reference/v6/student_best.pt` (11 MB) and `reference/v6/best_stage5.pt` (31 MB), which the two-level pattern would have missed (hence `**`), and `v7/cropguard_v7_fp32.onnx.bak` (9.6 MB).

### Pre-existing failure in the repo's own suite (NOT caused by this work)
`python3 tests/test_architecture_consistency.py` -> **6/8 passed**. The two failures are the v7 serving/canonical pair:
```
FAIL test_v7_serving_and_canonical_have_identical_state_dicts: v7 key sets differ
  only in serving:   seg_head.bw.2.{weight,bias,running_mean,running_var,num_batches_tracked}
  only in canonical: seg_head.bw.1.{num_batches_tracked,running_mean,running_var}, seg_head.up1.*
     + size mismatches: seg_head.fuse_skip.1 -> 48 vs 24 channels, seg_head.bw.0 -> [48,48,3,3] vs [24,1,3,3]
```
`app/api/model_v7_serving.py` (mtime **2026-10-10 23:02**) is a stale vendored copy: the canonical `model/architecture/v7_heads.py` (**2026-10-11 21:15**) replaced the third `bw` block with `up1.*`, and the serving copy was never re-synced. This is exactly the drift the guard exists to catch, and it is **dormant**, not live: nothing imports `model_v7_serving` (`grep -rn CropGuardMultiTaskV7 app/` finds only its own docstring), and `app/api/model_server.py` still loads `cropguard_model_v5.pt` through `from model import CropGuardMultiTask`. It becomes a hard blocker the moment a v7/v9/v10 checkpoint is promoted to the API -- report it, do not paper over it.
All four files involved predate this session (my edits are `RUNLOG.md`, `kaggle/*`, `.gitignore`, `model/weights/WEIGHTS_MANIFEST.json`); nothing in the v9/v10 eval work touches architectures or the serving copy.

---

## 2026-10-13 02:xx UTC — the v7 serving-copy drift, fixed and falsification-tested

The report above is now superseded: the drift was fixed and `tests/test_architecture_consistency.py` is **9/9 (exit 0)**. Kept as the record of what was wrong and how it was proven.

### The defect
`app/api/model_v7_serving.py:BilinearDecoder` was the pre-Oct-11 design; canonical `model/architecture/v7_heads.py:BilinearDecoder` had been reworked and the vendored copy was never re-synced.

| | serving copy (Oct 10) | canonical (Oct 11) |
|---|---|---|
| skip width | `width // 4` (24ch) | `width // 2` (48ch) |
| `fuse_skip` fused_ch | 24 | 48 |
| `bw` | depthwise 3x3 **+ pointwise 1x1** + BN | plain 3x3 + BN |
| upsampling | none — one final bilinear | progressive `up1` (→160, 32ch) + `up2` (→320, 24ch) then 1x1 `head` |
| `head` input ch | 24 (fused) | 24 (`up2` output) — same number, different path |

Every other module (backbone, neck, det head, lesion-gated cls head, abstain head) already matched, which is why the failure surfaced only as `seg_head.*` key/shape mismatches.

### The fix
Replaced the serving copy's `BilinearDecoder` with the canonical implementation (adjusted to the file's local `_hardswish()` helper), with a docstring stating the identity requirement and naming the prior divergence. Minimal and scope-exact: the state-dict guard covers the whole model, so if anything else had drifted the test would still fail.

### Verification — four independent checks
1. **Suite green:** `tests/test_architecture_consistency.py` → **9/9, exit 0** (was 6/8, exit 1).
2. **Real weights load (the actual acceptance criterion):** both archived checkpoints now load into the serving copy with **0 missing / 0 unexpected**, and a real forward runs:
   ```
   v9  (model/weights/v9/best_v8_raw.pt)   nc=8, 289 tensors -> missing=0 unexpected=0 -> seg_logits (1,2,320,320)
   v10 (model/weights/v10/best_v10_raw.pt) nc=8, 289 tensors -> missing=0 unexpected=0 -> seg_logits (1,2,320,320)
   ```
   `nc` is derived from the checkpoint (`cls_head.mlp.2.weight.shape[0]`), not hardcoded.
3. **New regression test, and it can fail.** Added `test_trained_v7_family_checkpoint_loads_into_the_serving_copy`: picks the newest `app/api/cropguard_model_v7.pt` or `model/weights/v*/best_*.pt`, loads it with `strict=True`, and runs a real forward asserting the 2-channel seg output. Falsification: reverting *only the decoder* to the stale design made **3 guards fail** (`..._identical_state_dicts`, `..._forward_agrees_between_the_two_copies`, and the new real-weight test) at 6/9; restoring the file returned md5 `fd134e6538b023b3316b0a5b43cc8c8c` and 9/9. So the guard is not vacuous and the fix is the only difference.
   An earlier attempt to falsify by monkeypatching the imported module **silently did nothing** — `_load_serving_v7()` re-execs the file on every call, so patching the returned module object cannot affect a later call. Worth remembering: for this file, drift must be simulated on disk.
4. **The skip path is honest.** The new test needs a ~10 MB gitignored blob, so it raises `Skipped` when none is present (deliberately *not* reusing the existing `ROOT`-based `CKPT` convention, which requires a **committed** blob — `app/api/cropguard_model_v5.pt` is tracked; the v9/v10 ones are not). `main()` now reports skips separately and excludes them from the pass count: with the checkpoints hidden it prints `8/8 passed, 1 skipped (not counted as passes)` and exits 0, so a fresh clone can never show a green pass for the check that did not run.

### Also corrected in the same pass
`model/weights/README.md` gained the v9/v10 section. A first attempt inserted it mid-file, swallowing the *Accuracy at a glance* caveats intro and leaving a duplicate `## Which one is the model?` heading; the file was rewritten with the section at the end (heading list now unique). Three figures in that table had been taken from the **training-time val log** (`train_stage_summary`) rather than the eval protocol — v9 val severity 0.151→**0.157** with baseline 0.229→**0.210**, and v10 Dice 0.548/0.593→**0.557/0.592**. All 20 numbers in the table are now cross-checked programmatically against `v9/results.json` and `v10/results.json`.

The remaining blocker for actually shipping v9 is no longer the architecture — it is that the v7-family checkpoints are `nc=8` (PlantSeg classes) while the served v5 is `nc=11`, so promotion is a deliberate API/class-map change.



## 2026-10-13 02:20 — v9/v10 COMPLETE + evaluated; DECISION: hybrid serving (option A)
- Both train kernels COMPLETE; outputs pulled to /home/logiclocus/tmp/kout_v9, kout_v10 and
  model/weights/v9, model/weights/v10 (results.json/md + fp32/int8 ONNX present).
- v9 (best_v8_raw.pt, epoch 79): PlantSeg val 0.7152 / test 0.6929; PlantDoc test 0.4177 CI[0.316,0.532],
  macro-F1 0.3890, ECE 0.3485, abstain coverage @10% on PlantDoc = 0.0000.
  Lesion IoU 0.4456 val / 0.5114 test; severity MAE 0.1566 val / 0.1176 test vs const baseline 0.2098 / 0.2343
  (beats baseline on both splits, test margin 49.8%), Spearman 0.7796 val / 0.8870 test, over-pred 1.37x.
- v8 (OLD, same eval path): PlantDoc test 0.5063 -> v9 REGRESSED in the field (-0.089 point estimate).
- v10 negative result confirmed: acc 0.7285 val but lesion IoU 0.414, severity 0.249 FAILS const baseline 0.229,
  mAP@0.5 0.022, best epoch 40/79 -> detection loss destabilises. Do not promote v10.
- DECISION (mine, user asked for the call): OPTION A — keep v5 nc=11 classifier, serve v9 seg+severity.
  Rationale: (1) full swap ships 0.418 field top-1 with ECE 0.349 and 0% abstention on OOD photos;
  (2) nc=8 = nc=11 minus leaf_miner/spider_mites/target_spot, dropped because EMPTY in corpus, so the swap
  forces confident misdiagnosis of pests, not a coverage gap; option B is blocked on data, not code;
  (3) frontend depends on the 11 label strings (app/web/src/lib/api.ts:27-39 treatment text keyed by exact
  class name; ResultView.tsx:99 "recognises these 11 classes only"; model_server.py:39-42 contract warning);
  (4) severity is v5's only outright failure (sev_r2 -1.41, app/api/metrics.json:12) and v9's only large,
  split-consistent win; (5) reversible via env var, two-model precedent MODEL_PATH_AUX in legacy v2.
- CAVEAT before enabling severity for users: v9 seg/severity was NEVER measured on PlantDoc
  (predict(is_plantdoc=True) skips seg; every PlantDoc seg row reads "not measured").
  Required probe: run v9 seg head over the 79 PlantDoc images, report predicted lesion-area fraction
  vs PlantSeg, before sev_reliable=True.
- Also owed: re-measure latency of BOTH models on the same machine (do not mix Xeon 66.7 ms with the
  26.1 ms figure), and a contradiction rule when v5=healthy but v9 finds lesions.

## 2026-10-13 02:55 — PROBE: decision-relevant measurements (script /tmp/probe_v9_v5.py, out /tmp/probe_v9_v5.json)
Command: `timeout 900 python3 /tmp/probe_v9_v5.py` (CPU). v9 ckpt model/weights/v9/best_v8_raw.pt
(nc=8, epoch 79, missing=0 unexpected=0); served v5 app/api/cropguard_model_v5.pt (nc=11, missing=0 unexpected=0),
loaded via app/api/model.py by path to avoid the `model` name clash.
1) v9 CLASSIFICATION on PlantDoc reproduced by this harness: top1 0.4051 (n=79) vs Kaggle eval 0.4177 -> 1.3pt
   gap, preprocessing difference; harness is trustworthy at that level.
2) SERVED v5 ON PLANTDOC, first time ever measured: top1 0.4430 (n=79), mean conf 0.7637.
   -> v5 vs v9 field gap = 3.8 pts on n=79; v9's own CI half-width was ~10.8 pts -> NOT a real difference.
   v5 used leaf_miner/spider_mites/target_spot on 13/79 images where GT has zero support for them (its wider
   label space costs it accuracy here). So the A-vs-B trade is NOT about classifier accuracy.
3) v9 SEGMENTATION under domain shift (no GT exists on PlantDoc; distribution check only):
   pred lesion area frac 0.2091 (PlantDoc) vs 0.2057 (PlantSeg val); leaf area frac 0.593 vs 0.623;
   empty leaf masks 0/79 and 0/156 -> no mask collapse in the field.
   pred severity median 0.377 field vs 0.264 lab.
4) SEVERITY BY CLASS, parsed from model/weights/v9/results.json _cache.new (reproduces reported MAE 0.1566 exactly):
   healthy n=20 true 0.0000 pred 0.0119 MAE 0.0119 (max 0.129)  <- the "do nothing" answer is safe
   bacterial_spot 0.4057->0.4749 MAE 0.1865 | early_blight 0.2134->0.3258 MAE 0.1436
   late_blight 0.2906->0.3777 MAE 0.1102 | leaf_mold 0.3331->0.4506 MAE 0.1197
   mosaic_virus 0.1730->0.4301 MAE 0.3224 | septoria 0.4582->0.4678 MAE 0.2045
   yellow_leaf_curl 0.3306->0.4873 MAE 0.2480 | label -1 (n=5) true 0.0 pred 0.3294
   -> EVERY diseased class is biased HIGH; 11/156 predictions saturate at severity 1.000.
   -> DISCLOSE: the headline "beats constant baseline by 49.8% on test" is partly earned by the 20 healthy
      zeros; diseased-only MAE ~= 0.172. Spearman 0.887 (test) is the more robust claim.
5) Field-healthy check (n=8 PlantDoc GT-healthy): pred severity 0.004..0.326, median 0.038, mean 0.104
   -> 2 of 8 clean field leaves read ~0.30 severity. Weak n, wrong direction, watch it.
DECISION (final, unchanged) = OPTION A hybrid, but on a CORRECTED rationale: not because v5 classifies
better in the field (measured: it does not, 0.443 vs 0.405, inside noise), but because ONLY v5 can name
leaf_miner / spider_mites / target_spot, and nc=8 would force those onto a *disease* label = wrong
treatment class (fungicide vs insecticide), the most harmful error this app can make.
Implementation constraint found: app/web/src/lib/api.ts ResultView already HIDES the meter when
severity_reliable is false, so the hybrid can ship the overlay first and gate the number with no UI change.
Recommended follow-up (needs user OK, it is a data/paper decision): the field top-1 ~0.44 for BOTH models is
the paper's biggest weakness; the fix is adding PlantDoc *train* tomato images to classification training
(never the test split; needs a duplicate check against the 79 test images).

## 2026-10-13 03:20 — TWO OF MY CLAIMS CORRECTED + selective-prediction measurement
Reviewer caught a real inversion in my report. Confirmed against code and stored artifacts:
- kaggle/v9-eval/cropguard-v9-eval.py:497-505 defines coverage = (probs.max >= thr).mean() = FRACTION ANSWERED,
  and results.json stores {"plantdoc_test": {"coverage": 0.0, "n_answered": 0, "n_abstained": 79}} for BOTH models.
  So "abstain valve does not fire on PlantDoc" was BACKWARDS: it abstained on 79/79 field images. My error.
- docs/V7_BRIEF.md:70 = "ONNX (v7, opset 17, 320^2, CPU 1 thread): mean 26.1 ms, p95 28.3 ms" -> v7 and MEAN,
  not v5/median as I implied. My error, smaller.
Commands: `timeout 900 python3 /tmp/correction_checks.py`; matched-T rerun of the same sweeps at T=1.0.
Selective prediction on PlantDoc field (n=79), same rule the eval uses:
  v9 at deployed operating point (T=1.3007, thr=0.9341): coverage 0.000 -> answers NOTHING in the field.
  v9 at T=1.0                     : thr 0.9 -> 25 answered @0.560 ; thr 0.9341 -> 9 answered @0.778
  served v5 at T=1.0              : thr 0.9 -> 28 answered @0.607 ; thr 0.99 -> 11 answered @0.727
  conf->correct AUC (T=1.0, matched): v9 0.680 vs v5 0.668 -> NO real difference in confidence quality.
  => v5's apparent field advantage at 0.9341 was a TEMPERATURE artifact (v9 was softened by T=1.3007,
     v5 had none). The val-fitted T (which cut val ECE 0.157->0.095) is what destroys field coverage.
  => "keep v5 because it knows when it is right" is NOT supported; Option A rests only on the 3 pest classes.
Leaf-denominator question (reviewer's suggestion, checked):
  corr(leaf_area_fraction, severity) = -0.492 field / -0.256 val.
  High-severity tail is mostly NUMERATOR-driven: median lesion px 34,179 (tail) vs 16,938 overall in field.
  BUT a real inconsistency exists: leaf area fraction <0.15 on 4/79 field and 3/79 val images, where median
  severity = 1.086 / 1.129, i.e. predicted lesion px EXCEED predicted leaf px.
  model_v7_serving.py:256-263 analytic_severity and cropguard-v9-eval.py:441 BOTH do lesion/leaf then
  clamp(0,1) with NO lesion^leaf intersection -> the clamp silently absorbs those cases.
  Consequence: the reported val severity MAE 0.1566 includes 11/156 predictions pinned at exactly 1.000.
  Fix to measure: sev = (lesion & leaf)/leaf. Cannot measure locally yet -- local cgpack masks are the
  OUTLINE era (solidity 0.002-0.017 on sampled files), so GT-matched severity needs the filled-mask dataset.
Contradiction-guard precedence (product call, my answer): v5=Healthy must NOT veto the overlay, because the
healthy-shortcut probe (val_holdout_acc 0.9841 from bare image stats) makes "v5 says healthy" an untrustworthy
authority. Rule = show v5's label + v9's overlay, and when they disagree beyond the fitted band set
severity_reliable=false with a "cannot reconcile / review" state and no treatment text.

## 2026-10-13 03:30 — v11: field-image training set built, driver written, smoke PASS, kernel PUSHED
Leakage verification (my own run, not the subagent's): md5 + 64-bit dHash over corpus(1021) + field(79).
  exact md5 duplicates in TRAIN = 2 | in VAL = 0 | in corpus TEST = 0
  dHash<=6 near-dups in TRAIN (non-exact) = 3  (ham0, ham1, ham6)
  -> 5 corpus-train images dropped from training in v11: tomato_early_blight_google_0043 / _0235,
     tomato_late_blight_google_0091, tomato_leaf_mold_google_0104, tomato_septoria_leaf_spot_google_0223
     (the last is md5-identical to a field image labelled bacterial_spot while its train filename says
     septoria_leaf_spot -> contradictory labels on the same photo)
  -> re-measured field top-1 on the clean subsets:
     ALL 79      : v9 0.4051 | v5 0.4430
     minus 2 md5 : v9 0.4026 | v5 0.4416
     minus all 5 : v9 0.3919 | v5 0.4459   <- published Kaggle v9 number was 0.4177
Dataset build (`python3 /tmp/build_v11_cls.py`): extracted PlantDoc train/Tomato* = 824 images across 9
  folders; Tomato_two_spotted_spider_mites_leaf has only 2 images and was skipped (cannot restore that class).
  Dropped 75 dup-of-corpus + 5 dup-of-field-test (md5 or dHash<=6 against corpus AND the 79 field images).
  KEPT 742. Per class: healthy 44, bacterial_spot 91, early_blight 58, late_blight 88, leaf_mold 77,
  mosaic_virus 39, septoria 126, yellow_leaf_curl 219. 178 MB, manifest.csv (file,class_name,class_index).
Driver `kaggle/v11/cropguard-v11-train.py` = hardened v9 + ONE variable (field stream):
  PlantDocClsDataset returns {image,cls} only. NOT a CropGuardDataset subclass because dataset.py:293-304
  silently returns leaf=ones/mask=zeros for missing masks, which would teach seg that field leaves are
  lesion-free. Field batch runs active_tasks={"cls","seg"} with gt_mask=None -> gate uses the model's OWN
  predicted mask (v7_model.py:124-133, the inference path) and its masks are NEVER passed to seg_loss_v7.
  Separate class_weights for the field stream (lab counts are BOX counts, field counts are IMAGE counts;
  merging them would silently reweight the lab stream). Lab sampler/LR/EMA/loss untouched.
  Artifacts renamed best_v11_raw/ema.pt, last_v11.pt, v11_train_log.csv (so a v9 and v11 mount can't be
  confused). CSV gains a pd_cls column between cls and seg.
Smoke (`V8_FORCE_AUG=photo V8_EPOCHS=2 V8_TRAIN_SUBSET=24 V8_VAL_SUBSET=12 V8_BATCH=4 V11_PD_BATCH=4
  V8_NUM_WORKERS=2 python3 cropguard-v11-train.py`, corpus_v9 built from local filled masks 715/156/150):
  RUN 1 FAILED silently: NameError torch in PlantDocClsDataset.__getitem__ (module never imports torch at
  module scope). Driver still exited 0 and wrote stage2_failure.json -> EXIT CODE IS NOT PROOF OF SUCCESS;
  always check work/*failure*.json. Fixed with a local `import torch`.
  RUN 2 PASS: stage0 ok / stage1 photo / stage2 ok, no failure json, "dropped 5/5 field-test duplicates",
  "field stream: 742 PlantDoc images ... counts=[44,91,58,88,77,39,126,219] weights=[1.369,...,0.614]
  loss_weight=1.0", E1 cls=2.0979 pd=2.0908 seg=6.8257, E2 cls=2.0473 pd=2.1089 seg=5.9572.
  pd_cls ~= ln(8)=2.079 as expected for a random-init 8-way head; metrics are meaningless in a 2-epoch
  random-init smoke and are reported as plumbing evidence only.
Kaggle: dataset `ganrajlongadge/cropguard-plantdoc-tomato-train` created. NOTE two real gotchas:
  (1) `datasets create` REQUIRES a licenses[] block; I declared attribution-4.0-international (CC BY 4.0),
      NOT CC0 -- PlantDoc is someone else's work and CC0 would be a false licensing claim. ATTRIBUTION.txt
      records provenance, the 8 folders used, the dedup filter and the "class labels only, never a seg target"
      constraint.
  (2) the first listing showed only 2 files; after `datasets version` with images.zip the images/ entries
      appear. Do not trust a single `datasets files` call as proof of upload completeness.
Kernel pushed: ganrajlongadge/cropguard-v11-train-field-images-in-cls-stream v1 (title-derived slug; the
  metadata id was updated to match). Queued behind nothing (v9/v10 both COMPLETE).

## 2026-10-13 03:50 — SEVERITY FIX implemented and measured through the shipped serving code
Measurement (`/tmp/severity_intersection.py`, v9 best_v8_raw.pt, corpus_v9 val n=156, thr 0.5, per-image):
  variant      MAE vs sev_cont   MAE vs GT area ratio   pearson r   predictions pinned at 1.000
  naive            0.1508              0.2378             0.7539      11
  hard AND         0.1308              0.2101             0.7744       2
  soft product     0.1296              0.2029             0.7733       0
  Root cause quantified: mean 1672 predicted lesion px lie OUTSIDE the predicted leaf; raw ratio > 1 on
  11/156 images, and clamp(0,1) converted that into 11 predictions pinned at exactly 1.000.
  NOTE on protocols: this n=156 (includes the 5 cls=-1 images) so 0.1508 is NOT directly the published
  0.1566, which used n_sev=151. Same-direction comparison only until the eval is re-run with the fix.
Chose SOFT over hard AND: better MAE, zero pinned predictions, and it stays differentiable so the other
trainer's consistency loss can still backprop through analytic_severity.
Edited BOTH copies in lockstep (tests/test_architecture_consistency.py compares forward incl. sev_cont):
  - model/architecture/v7_model.py analytic_severity  -> num = (lesion*leaf).sum, den = leaf.sum + eps
  - app/api/model_v7_serving.py  analytic_severity  -> same arithmetic, same return shapes
Verification: `python3 tests/test_architecture_consistency.py` -> 9/9 passed (unchanged).
Then re-measured THROUGH the serving module (importlib on app/api/model_v7_serving.py, strict load
missing=0 unexpected=0, m.analytic_severity on val n=156): MAE 0.1296, r 0.7733, 0 predictions at 1.000.
Next: the published eval must be re-run with the corrected severity (v11 eval kernel) so the headline
number uses this formula, not the clamped one.

## 2026-10-13 04:00 — HYBRID SERVING built and verified through the real API + field calibration
Field-domain calibration (`/tmp/field_calibration.py`; fit on the 742 deduped PlantDoc TRAIN images,
reported on the 74 clean field TEST images; test set never used for fitting):
  v9 : T_field=1.7080 thr=0.3736 -> coverage 0.838 (62/74) acc_on_answered 0.4355  (unconditional 0.3919)
  v5 : T_field=3.0671 thr=0.3216 -> coverage 0.662 (49/74) acc_on_answered 0.5306  (unconditional 0.4459)
  BOTH models at the LAB-fitted thr 0.9341 -> coverage 0.000 (0/74). Confirms the corrected reading.
  CAVEAT: the two rows are at different realized coverages, so do NOT claim one model abstains better.
  Honest headline: field confidence carries little information (AUC 0.680 / 0.668), so abstention buys
  only +0.04 to +0.09 accuracy. It is a mitigation, not a fix.
Hybrid serving implemented (additive, reversible via env):
  app/api/model_server.py  + SEG_NC=8, LESION_CONFLICT_SEVERITY=0.10, get_seg_model(), seg_model_status(),
    second forward with active_tasks={"seg"} ONLY (v9's 8-class head is never run, so it cannot leak into
    the label space), analytic_severity for the number, lesion mask written as {id}_lesion.png (distinct
    suffix so it can never overwrite v5's {id}_mask.png), response gains segmentation.lesion_mask_png_url /
    lesion_fraction / seg_model / seg_conflict and severity.lesion_ratio + validated_on note,
    timings_ms.seg_model. severity.reliable stays FALSE (no field GT for severity).
  app/api/main.py          + import get_lesion_mask_path, new route GET /v1/lesion_masks/{image_id}.png
  Load path: weights_only=False for the training dict, strict=True, failure degrades to v5-only.
VERIFIED through the running server (uvicorn on :8000, MODEL_PATH=v5, SEG_MODEL_PATH=v9 best_v8_raw.pt):
  field image test_Tomato Early blight leaf_1.jpg ->
    classification early_blight 0.8064 (v5, correct)
    segmentation  leaf_fraction 0.3995 | lesion_fraction 0.2750 | seg_model seg-nc8-epoch79 | conflict False
    severity      v5 continuous 0.913 stage Severe reliable=False (unchanged contract) | v9 lesion_ratio 0.3794
    timings_ms    preprocess 8, model(v5+TTA) 250, postprocess 186, seg_model 134  (CPU, not phone)
    GET /v1/lesion_masks/... -> http_code=200 3495 bytes, mask 320x218, 19181 white px (fraction 0.2750)
  degraded path (server WITHOUT SEG_MODEL_PATH, port 8001): same class early_blight 0.8064, all three new
    fields None, old fields intact -> the hybrid is a strict superset, no contract break.
  24 field images (3 per tomato folder) through the API: conflicts flagged 1.
    The catch: a Tomato_leaf_mosaic_virus photo where v5 said Healthy @0.675 while v9 severity=0.2396.
    The two GT-healthy field leaves that v5 also called Healthy scored 0.0225 and 0.0192 (healthy band).
    v9 field severity distribution min 0.014 median 0.294 max 0.940.
    Also visible: v5 spends 6/24 predictions on leaf_miner/spider_mites where GT is a disease -- the
    incumbent's pest classes cut both ways and must be reported honestly.
Mistakes I made this turn (recorded so they are not repeated):
  1) `pkill -f "python3 main.py"` matched its own command line and killed my own job; servers had to be
     killed by PID afterwards. Never use self-matching pkill patterns.
  2) The v11 driver smoke run 1 exited 0 while stage2 had actually failed (NameError torch in my new
     dataset class). The driver catches everything -> must check work/*failure*.json, not the exit code.
  3) `rm -rf images` after zipping left the local dataset inconsistent with the manifest, and the zip was
     built flat (-j) so extracting it does NOT recreate images/. Restored with mkdir+mv; Kaggle does expose
     images/ (verified with `kaggle datasets files`).
  4) `kaggle datasets create` needs a licenses[] block; first attempt silently uploaded 0 images.
STATUS: v11 kernel pushed and RUNNING (ganrajlongadge/cropguard-v11-train-field-images-in-cls-stream v1).
  Check with: kaggle kernels status ganrajlongadge/cropguard-v11-train-field-images-in-cls-stream

## 2026-10-13 04:10 — LayerCAM XAI on the real v9 checkpoint (task 8)
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 python3 model/training/layercam_v7.py --ckpt
 model/weights/v9/best_v8_raw.pt --data /home/logiclocus/tmp/v11smoke/work/corpus_v9 --split val
 --n 24 --out /tmp/cam_v9 --overlays 3`  (thread-capped + nice'd: laptop is thermally limited)
n=24, all 24 have lesions and leaf:
  lesion_cam_energy_frac_mean 0.3586 | lesion_area_frac_mean 0.2334 -> lesion_lift 1.5360
  leaf_cam_energy_frac_mean   0.7726 | leaf_area_frac_mean   0.6943 -> leaf_lift    1.1128
Reference points already measured: random-weight null lift 1.0295 (lesion) / 1.0088 (leaf);
v8 best checkpoint on 40 images scored lesion_lift 1.8501.
Read honestly: v9's CAM still concentrates on lesions (1.54x chance) but LESS than v8 (1.85x), and the
two runs differ in n (24 vs 40), so this is directional, not a controlled comparison. Do not report
"v9 XAI improved"; report "both localise, v8 more sharply on this sample".

## 2026-10-13 13:55 — v11 (field images in the classification stream) MEASURED + leakage-checked
Kernel `ganrajlongadge/cropguard-v11-train-field-images-in-cls-stream` is COMPLETE.
NOTE: `kaggle kernels files` reports bogus sizes for this kernel (899 B for a 10 MB checkpoint);
the real download is correct (best_v11_raw.pt 10,292,480 B, last_v11.pt 40,767,709 B). Do not read
file sizes from `kernels files`.

Training (from the Kaggle log, in-kernel):
  best epoch 65/84 (TIME CAP at 54.5 min), TOTAL 59.4 min
  val: acc=0.7550 macro_f1=0.6951 iou=0.4451 dice=0.6160 sevMAE(pred)=0.1669 sevMAE(trueleaf)=0.1880
  corpus checks: counts 715/156/150, overlay_median_ratio 20.40, lesion_inside_leaf_min 1.0

Measured locally through the same harness/code path as v8/v9/v10 (no new code path):
  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  python3 /home/logiclocus/tmp/papermetrics/harness.py --phase collect --models v11 --splits val,test
  python3 /home/logiclocus/tmp/papermetrics/metrics.py
  python3 /home/logiclocus/tmp/papermetrics/harness.py --phase plantdoc --models v8,v9,v10,v11
  (v11 loaded strict: missing=0 unexpected=0)
  v11 val : acc=0.7550 f1=0.6951 iouG@.5=0.4305 iouP@.5=0.3842 leafiouG=0.2938 sevMAE=0.1689 (baseline 0.2246) map50=nan
  v11 test: acc=0.7429 f1=0.7014 iouG@.5=0.4869 iouP@.5=0.4170 leafiouG=0.2724 sevMAE=0.1264 (baseline 0.2453) map50=nan
  FIELD (PlantDoc test, 79): v8 0.4684 | v9 0.4051 | v10 0.4430 | v11 0.6203   (clean-74: 0.4595/0.3919/0.4324/0.6216)
  => v11 is best on val acc, test acc, test macro-F1 and field top-1, and still beats the severity baseline by 48.5%.
     Cost: lesion IoU 0.5114 -> 0.4869 global (test) and sevMAE 0.1163 -> 0.1264; both remain far above/below the
     v8-era numbers. v11 has NO det head trained (map50=nan, 0 predicted boxes), same as v8/v9.
  McNemar (2000-resample, exact two-sided) adds no significance: v9_vs_v11_val p=0.3269, v10_vs_v11_test p=1.0000,
     v8_vs_v11_test p=0.3771. The +21 pt FIELD jump is the only difference that is large; on-corpus accuracy is not.

Leakage check on the field gain (mandatory before reporting a result this large):
  python3 /tmp/leakcheck_v11.py /tmp/v11ds/images     # the 742 images v11 actually trained on
  field TEST tomato 79 | train 742
  EXACT md5 collisions: 0
  NEAR duplicates dHash<=6: 0
  VERDICT CLEAN.
  For the record, the RAW PlantDoc train tomato set (824 images) DOES contain 5 near-duplicates of field test
  images (4 unique: yellow_virus_17_1, yellow_virus_17_2, late_blight_25, bacterial_spot_9) plus one md5-identical
  cross-label pair (Tomato_Septoria_leaf_spot/train_..._20.jpg == Tomato_leaf_bacterial_spot/test_..._1.jpg, which is a
  PlantDoc label error, not something we introduced). All 5 are ABSENT from the shipped 742, so the dedup pipeline
  worked. Keep that raw-set finding for the dataset-quality section.
  CAVEAT, must be in the paper: v11 seeds PlantDoc TRAIN tomato images into the classification stream, so 0.6203 is
  domain ADAPTATION on the same dataset as the field test, NOT zero-shot field transfer. The honest sentence is
  "adding field images to the classification stream raises field top-1 from 0.4051 to 0.6203 on a held-out split of
  the same dataset"; do not write "v11 generalises to the field".
Artifacts: /tmp/v11_summary.json, /tmp/kout_v11/, D1_D2_metrics.json / D3_paired.json / plantdoc_metrics.json
(regenerated with v11 included).

## 2026-10-13 14:05 — v11 UNCAPPED (v2) pushed; cap is now opt-in
Requested: "remove the cap but do all of the 84 epochs first". The v11 run stopped at epoch 84/120
because the 55-min cap fired, NOT because it had converged -- so every v11 number was cap-limited.
Change (kaggle/v11/cropguard-v11-train.py), minimal and math-neutral:
  1. CAP_MINUTES default "55" -> "0", and the in-loop check is now guarded `if CAP_MINUTES > 0 and ...`.
     0 = no cap. A positive V8_CAP_MIN restores the old behaviour exactly.
  2. GLOBAL_DEADLINE_MIN default "100" -> "600". This was REQUIRED, not cosmetic: 120 epochs at the
     measured ~0.65 min/epoch is ~85 min of stage 2 on top of ~10 min of stages 0/1, so the old
     100-min watchdog would have truncated the run in precisely the way the cap used to. 600 min is
     a hang backstop, not a training cap.
  3. BUG FIX found by the smoke test: the field-stream DataLoader passed `timeout=-1` when
     NUM_WORKERS=0, and torch raises `ValueError: timeout option should be non-negative` at
     construction. The driver could therefore never be smoke-run on CPU. Now `else 0`.
     loader_for() (line ~618) was already correct; only the pd_loader was wrong.
The new run performs all 120 epochs, i.e. it redoes the 84 v11 completed and the 36 it never reached,
as one continuous run with no resume seam in the LR schedule.

Verification (local, CPU, cap exercised BOTH ways):
  # cap OFF -> runs every configured epoch, no TIME CAP line
  nice -n 15 env V8_WORK=/tmp/v11b_smoke/work V8_CODE_ROOT="$PWD/model" \
    V8_INPUT_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v7_split \
    V9_FILLED_ROOT=/home/logiclocus/tmp/cgpack/cropguard_v9_fixed \
    V9_CORPUS=/home/logiclocus/tmp/v11smoke/work/corpus_v9 V11_PD_ROOT=/tmp/v11ds \
    V8_STAGE1_STEPS=20 V8_FORCE_AUG=photo V8_TRAIN_SUBSET=16 V8_VAL_SUBSET=16 \
    V8_EPOCHS=2 V8_CAP_MIN=0 V8_DEADLINE_MIN=25 V8_NUM_WORKERS=0 \
    timeout 900 python3 -u kaggle/v11/cropguard-v11-train.py
  -> E 1/2, E 2/2, stage2 'ok'; field stream loaded: "742 PlantDoc images from /tmp/v11ds (skipped 0)"
  # cap SET -> still fires (regression check that the guard did not disable a real cap)
  V8_EPOCHS=4 V8_CAP_MIN=0.02 ... -> "TIME CAP: 0.0 min used, ~0.04 min/epoch, cap 0.02 min
                                       -> stopping cleanly at epoch 1"
  python3 -m py_compile kaggle/v11/cropguard-v11-train.py -> OK

Metadata fix: kaggle/v11/kernel-metadata.json had a STALE id
(ganrajlongadge/cropguard-v11-train-field-images) that no longer matched the live kernel
(ganrajlongadge/cropguard-v11-train-field-images-in-cls-stream). Pushing with the stale id would have
forked a brand-new kernel instead of versioning the real one. Id corrected first, then pushed.
  kaggle kernels push -p kaggle/v11  -> "Kernel version 2 successfully pushed"
  kaggle kernels status <slug>       -> KernelWorkerStatus.RUNNING
Expected wall clock ~95 min (120 epochs x ~0.65 min/epoch + ~10 min stages 0/1).
Artifacts land under /kaggle/working as v11_train_log.csv (now 120 rows) / best_v11_raw.pt /
stage_summary.json. NOTE: `kaggle kernels files <slug>` reports bogus sizes for this kernel; judge
completeness from the downloaded files, never that listing.

## 2026-10-13 14:25 — number-by-number verification of the manuscript (93 claims)
Two checker scripts, one per lineage. Each claim is (label, paper_value, artifact, json_path,
tolerance); the value is READ OUT of the JSON, never from memory, and a mismatch is printed as a
mismatch rather than smoothed.

  python3 /tmp/verify_paper_numbers.py    # lineage B: 70 claims
  -> "checked 70: 70 match, 0 MISMATCH, 0 missing"
     covers tab:v9v10 + tab:fieldadapt (lesion IoU global + per-image, leaf IoU, accuracy,
     macro-F1, severity MAE + baselines, v10 mAP/box counts), the four field top-1 values
     (all-79 and clean-74), all five corpus McNemar p-values, the six field McNemar values and
     the four Wilson bounds, and the v11 in-kernel training summary (acc/macro-F1/IoU/Dice/sevMAE).
  python3 /tmp/verify_lineageA.py         # lineage A: 23 claims
  -> "checked 23: 23 match, 0 MISMATCH, 0 missing"
     covers the parameter budget (TOTAL 2533254, backbone 2214900, neck 146592, seg_head 81666,
     cls_head 65480, det_head 24616), the v8 checkpoint's stored metrics (macro-F1 0.6786,
     acc 0.6954, IoU 0.0459, sev MAE 0.1026), mean predicted/true lesion area 4729.0381/621.3378,
     over-prediction 7.6111, containment 0.6267, kappa 0.6488, the outline GT severity
     mean/median 0.0383/0.0099, the v9-corpus containment/severity/clamp, and n_train_labelled 663.
  The 0.0387 constant-severity baseline the abstract quotes is confirmed in verifyH.json
  (outline/baseline_mae_on_val/train_mean = 0.0387), so "MAE 0.1026 vs 0.0387" is resolved.

ONE real inconsistency found, and then a correction to my own correction:
  The v8 VALIDATION macro-F1 in lineage B has two values in this project: 0.6786 (stored in the
  checkpoint / reported by the evaluation kernel) and 0.6770 (our recomputation; verifyL.json and
  the D1_D2 harness agree). I first "fixed" the two lineage-B tables to 0.6770 -- WRONG: lines
  76/118 of 07_results.tex are also lineage B (tab:v8val), so that change made the paper
  internally inconsistent with itself. Reverted to 0.6786 in both tables, which is the paper's
  deliberate convention (quote the stored/kernel value, disclose the recomputation). The caption
  of tab:fieldadapt and a new PROJECT_FACTS paragraph now state the 0.6770 recomputation and the
  0.0016 difference EXPLICITLY, so both numbers are on the record and neither is averaged away.
  Everything else agrees to four decimals across both pipelines.

Rebuild after the fix: 28 pages, 0 undefined citations/refs, audit.py 7/7 ok, 710 distinct numeric
tokens all present in PROJECT_FACTS.md.
Not machine-checkable and therefore NOT claimed as verified: prose-only comparative statements
("$5.4\times$", "$\approx$37%", "$+0.035$") were checked by hand against the same artifacts;
citation page numbers are out of scope for this pass.
