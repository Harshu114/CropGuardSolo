# CropGuard v7 runbook (one day, Kaggle-capable)

Everything below is implemented and smoke-tested locally; this is the exact
order to run it. The v6 run (`model/weights/reference/v6/`) is the baseline v7
must beat on the same val protocol:

```
v6 stage-5 final VAL: det_cls 0.429 det_box 0.164 det_dfl 3.113 cls 0.285 seg 0.280 sev 0.065
```

---

## 0. Pre-flight (do this FIRST — it is the #1 way to waste a session)

**Offline assets.** `Backbone(pretrained=True)` now RAISES if timm weights
cannot be downloaded (a silent random-init fallback can burn 30 epochs on noise).
Pre-populate a cache and attach it as a Kaggle dataset:

```bash
# local, once, with internet:
python3 -c "import timm; timm.create_model('fasternet_t0', pretrained=True)"
cp -r ~/.cache/huggingface /path/to/kaggle_assets/hf_cache        # backbone
# optional (better leaf masks):
wget -O kaggle_assets/sam_vit_b.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

On Kaggle, attach `kaggle_assets` and set:

```bash
export HF_HOME=/kaggle/input/kaggle-assets/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
```

`baseline_smoke.py` (step 5) proves the weights actually loaded before you
commit hours.

**Class map.** PlantSeg disease names are NOT PlantVillage names. Verify before
training:

```bash
python3 data_pipeline/scripts/derive_dataset.py \
    --check-annos PlantSeg/coco_annotations --map data_pipeline/configs/class_map.yaml
```

Unmapped names are listed and the exit code is non-zero. Update
`class_map.yaml` until it is clean — skipped images are silent accuracy loss.

---

## 1. Get the data (plantseg_tomato, 3 GB)

```bash
curl -L -o plantseg.zip https://zenodo.org/records/13762907/files/plantseg.zip
unzip plantseg.zip -d plantseg
git clone https://github.com/tqwei05/PlantSeg      # coco_annotations/
python3 data_pipeline/scripts/prepare_plantseg.py \
    --root plantseg --annos PlantSeg/coco_annotations --out plantseg_tomato --plant tomato
```

## 2. Derive every label (one dataset, all heads)

```bash
python3 data_pipeline/scripts/derive_dataset.py \
    --in plantseg_tomato --out cropguard_v7 \
    --negatives-in plantseg          # 33 non-tomato hosts -> not_tomato set
```

Produces `images/`, `masks_lesion/`, `masks_leaf/`, `labels/`,
`severity_<split>.csv`, `data.yaml`.

- boxes: connected components of the lesion mask
- leaf masks: GrabCut by default; `--leaf-backend sam --sam-ckpt ...` for SAM
  (**run this as a CACHED step** — it is the slow part; write the masks once and
  re-attach them, never regenerate inside a training session)
- severity: lesion/leaf, **only on close-up single-leaf images** (≥55% leaf
  coverage, 1 component). Others get `-1` = not assessed instead of a meaningless
  number. Grades use the standard area scale (5% / 25%).

## 3. Dedup + split (duplicate groups never straddle splits)

```bash
python3 data_pipeline/scripts/dedup_split.py --in cropguard_v7 --out cropguard_v7_split
```

Reports how many near-duplicates it found before splitting — expect a large
number on PlantSeg; that is exactly the leak this removes.

## 4. Train

```bash
python3 model/training/train.py \
    --root cropguard_v7_split --arch v7 --out runs/v7 \
    --stages 1,2,3,4,5 --stage-epochs 10 --batch-size 32 --size 320
```

Stage semantics are unchanged (1 det, 2 cls, 3 seg, 4 joint, 5 joint+severity),
with three v7 behaviours driven by stage:

| stage | what is new in v7 |
|---|---|
| 2 | cls head trains on plain GAP (segmenter is random then — gating on it would be noise) |
| 4-5 | lesion-gated pooling turns on, GT→predicted mask mix goes 0.8→0.2 across the stage, consistency loss activates, EMA tracks weights |

Checkpointing (`last.pt`, `best_stage{N}.pt`, `student_best.pt`) and
`--resume` already survive the 9 h session cap; keep `--resume` on.

## 5. Verify, then evaluate

```bash
python3 model/training/eval_tta.py --arch v7 --ckpt runs/v7/student_best.pt \
    --root cropguard_v7_split --split test --out metrics_v7.json
```

Reports mAP@0.5(:0.95), Top-1, **macro-F1, per-class F1, ECE**, seg mIoU/Dice,
severity MAE/RMSE/R²/QWK. Compare the per-class F1 breakdown against v6 — that
is where you see WHICH disease regressed.

## 6. Calibrate abstention (do not skip: this is a shipped feature)

```bash
python3 model/training/calibrate_abstention.py \
    --ckpt runs/v7/student_best.pt --root cropguard_v7_split --split val \
    --target-abstain 0.05 --write
```

Fits temperature by NLL and writes `abstain.temperature` into the checkpoint,
plus prints the prob/energy thresholds at a 5% abstain rate.

## 7. Edge check

```bash
python3 model/training/export_onnx.py --arch v7 --ckpt runs/v7/student_best.pt \
    --out cropguard_v7.onnx --compare
```

**Check operator support FIRST on a dummy export before trusting a full run**
(FasterNet's partial-conv slice/concat is the risk). Latency is measured in
ONNX Runtime with 1 thread (RPi proxy). Note the honest cascade result: backbone
and BiFPN run every time, so the cascade only skips the seg decoder.

## 8. If the numbers are bad — debug ladder

1. **Overfit 50 images** (`--max-images 50 --stage-epochs 40`). If it cannot
   memorise them, the bug is in labels/loss/loading, not the architecture.
2. **Look at the labels**: overlay masks and boxes on 50 images
   (`derive_dataset.py` outputs make this easy).
3. **Baseline compare**: pretrained classifier alone vs the multi-task model. If
   multi-task is worse, one head is interfering — train heads separately and
   compare (that is what staged training is for).
4. **Per-class F1** from step 5: rare classes vs look-alike diseases
   (early vs late blight vs septoria are all brown spots).
5. **Train vs field gap**: if large, add more field-style augmentation
   (background paste is already on) and more real field images.

## Deliberately skipped (expert advice)

PCGrad (on a one-day budget), NAS, ensembles, test-time augmentation
(too slow for a phone — `--tta` exists but is off by default).
