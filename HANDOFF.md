# 🤝 Project Handoff: CropGuard

> **Rewritten 2026-10-08.** The previous version of this file described GUM and
> task-adapters as implemented, called the project modularized and complete, and
> listed "final training run" as pending. The serving API could not start, those
> two modules never existed in the trained model, and the training script was not
> runnable. Everything below is checked against the code and the shipped
> checkpoint. Where something is broken or weak it says so.

## 🎯 What this project is

A multi-task tomato-leaf disease diagnostic system: one forward pass produces
**detection, classification, segmentation and severity**. Built around a
FasterNet-T0 backbone (~2.63 M params) so it runs on CPU or mobile.

## ✅ Verified working (2026-10-08)

| Component | State | Evidence |
|---|---|---|
| Model loads | ✅ | `strict=True` load of `cropguard_model_v5.pt`, 292 tensors, 2.63 M params |
| API serves | ✅ | `python main.py` → `/v1/health` ok, `/v1/predict` returns 200 |
| Real-image inference | ✅ | 12 real test images: mask + CAM written 12/12, boxes in bounds |
| Frontend vocabulary | ✅ | 11/11 classes present in `web/src/lib/DISEASE_INFO` |
| Losses / dataset / eval | ✅ | byte-identical to the v5 training run that produced the checkpoint |
| Training pipeline | ✅ | `model/training/train.py` runs stages 1–5; smoke-tested locally |
| Kaggle training | ✅ | `ganrajlongadge/cropguard-v6-train` (5 stages × 10 epochs) |

## 🛠 Architecture (what the checkpoint actually contains)

- **Backbone:** `fasternet_t0` via `timm`, `out_indices=(1,2,3)` → C3/C4/C5 = 80/160/320 ch.
- **Neck:** BiFPN-lite, 3-level bidirectional fusion, softmax-normalized learnable weights.
- **Detection:** YOLOv8-style decoupled head, DFL box encoding, 2100 anchors at 320 px.
- **Classification:** SE → GAP → MLP pooled over **P3+P4+P5** (multi-scale).
- **Segmentation:** LightHam-lite decoder (Hamburger block + transposed-conv upsampling).
- **Severity:** gated multimodal fusion of seg/det/cls embeddings + area ratio.

### ❌ Not in the model, despite earlier claims

The previous handoff described a "Guided Upsampling Module (GUM) using P2 features"
and "task-specific residual adapters". **Neither exists in the trained
checkpoint** — its state_dict has no `guide.*` and no `adapters.*` keys, and
`heads.py` contains no `TaskAdapter` class. The code that referenced them was
what broke the API. If these are pursued as research they need implementing *and*
a retrain; adding them to the serving copy alone will make the checkpoint fail to
load (that is exactly what `strict=True` is there to catch).

## 📊 Honest performance (`app/api/metrics.json`, test split, 1286 images)

| metric | v5 (served) | paper target |
|---|---|---|
| mAP@0.5 | **0.511** | 0.892 |
| mAP@0.5:0.95 | 0.431 | 0.624 |
| cls_top1 | 0.853 | 0.987 |
| seg_mIoU | 0.880 | 0.863 |
| sev_mae | 0.068 | 0.042 |
| sev_r2 | **−1.41** | — |

**These figures were re-checked on a leak-free split** and held up — leakage was
not inflating them. `data_pipeline/scripts/clean_corpus.py` removed 1,425
duplicate images, dropped 52 images whose duplicate groups carried conflicting
labels, and moved 32 group representatives so no duplicate group straddles a
split (test 1,286 -> 1,240). Re-running `eval.py` on that clean split with the
same checkpoint gave `app/api/metrics_clean_split.json`:

| metric | original (1,286) | clean (1,240) |
|---|---|---|
| mAP@0.5 | 0.5114 | **0.5244** |
| mAP@0.5:0.95 | 0.4311 | **0.4422** |
| cls_top1 | 0.8527 | 0.8504 |
| seg_mIoU | 0.8797 | 0.8826 |
| sev_r2 | −1.409 | −1.326 |

Since the clean numbers are equal or better, the reported results can be quoted
without a leakage caveat. Re-run this any time with:

```bash
python3 data_pipeline/scripts/clean_corpus.py --root <corpus> --out /tmp/clean_corpus
cd app/api && python3 eval.py --root /tmp/clean_corpus --ckpt cropguard_model_v5.pt \
    --split test --nc 11 --out metrics_clean_split.json
```

Two caveats that must not be papered over:

1. **`seg_mIoU` is not lesion segmentation.** The corpus masks are filled
   *leaf bounding boxes* (binary {0,255}; the "128 = lesion" value never occurs in
   any of the 32,897 masks). The head predicts a leaf *region*. The API therefore
   reports `leaf_fraction`, not `lesion_fraction`.
2. **The severity head is worse than a constant predictor** (r² −1.41). Its labels
   were derived from how much of the frame the leaf box covers, which for a
   close-up leaf photo is always ≥30% → "Severe" for essentially every image. The
   API returns `severity.reliable = false` so clients hide it.

Fixing these two needs **better labels**, not more epochs: real lesion masks (e.g.
PlantSeg) and a real severity definition.

## 📂 Layout

```
model/
  architecture/   backbone.py, neck.py, heads.py, model.py  (canonical definitions)
  training/       train.py (+dataset/losses/eval_tta/measure_tta/rank_ckpts)
  weights/        all 16 checkpoints, renamed <version>_<accuracy> + manifest
app/
  api/            FastAPI: model_server.py, main.py, model.py (serving copy),
                  losses.py, dataset.py, eval.py, inference_utils.py
  web/            Next.js frontend
data_pipeline/    source/validate/augment scripts
mobile/           Flutter app; generate_real_assets.py builds its sample assets
docs/             architecture notes
```

`app/api/model.py` is a **vendored copy** of `model/architecture/model.py` because
the API runs as a script and ships with `app/api` as its Docker build context, so
the repo-root package isn't importable in the container.

That duplication is guarded — if you change one, the test fails until you change
the other:

```bash
python3 tests/test_architecture_consistency.py    # or: pytest tests/
```

It asserts identical state_dicts, identical forward outputs, a successful
`strict=True` load of the shipped checkpoint, and — so the guard cannot itself be
vacuous — that a deliberately wrong model is reported as different.

## 🚀 Running it

**API**
```bash
cd app/api
MODEL_PATH=$PWD/cropguard_model_v5.pt python3 main.py   # http://localhost:8000
python3 verify_e2e_real.py 6        # posts real corpus images, checks the contract
```

**Data.** The corpus root is configurable as `CROPGUARD_DATA`; nothing hardcodes
a machine-local path any more.
- Kaggle dataset: `ganrajlongadge/cropguard-corpus-claude` (29,611 train / 2,000 val / 1,286 test)
- local copy: `../ClaudeCropGuard/datasetclaude`

**Training (local)**
```bash
CROPGUARD_DATA=/path/to/datasetclaude python3 model/training/train.py \
  --stages 1,2,3,4,5 --stage-epochs 10 --batch-size 32 --amp --out runs
```

**Training (Kaggle GPU)** — the realistic route (this laptop has 4 GB VRAM):
```bash
kaggle kernels push -p <kernel dir>     # see ganrajlongadge/cropguard-v6-train
kaggle kernels status  ganrajlongadge/cropguard-v6-train
kaggle kernels output  ganrajlongadge/cropguard-v6-train -p ./out
```
Measured cost on Kaggle: ~350–400 s/epoch, peak 6.4 GB VRAM at batch 32, ~5–6 h
for all 50 epochs. `train.py` refuses GPUs with compute capability < 7.

## ⏳ Next steps (in priority order)

1. **Fix the labels** — obtain real lesion masks (PlantSeg: 11,458 in-the-wild
   images with expert-annotated disease polygons, includes tomato) and a real
   severity target. This is the only thing that moves severity/segmentation.
2. **Report segmentation honestly** in the paper: leaf-region IoU, or retrain on
   real masks and re-measure.
3. Detection is the strongest head (mAP@0.5 0.511); consider more detection-focused
   training before chasing classification gains.
4. Wire the Flutter app to the live API (it currently bundles offline samples).

## 🔑 Notes

- **The v5 pipeline already existed** at `ClaudeCropGuard/code_work/v5_run/`
  (train.py/model.py/dataset.py/losses.py/eval.py + all stage checkpoints +
  logs). It has now been brought into `model/`. The shipped checkpoint is
  byte-identical (md5 `7cf29afa…`) to `v5_run/runs/student_best.pt`.
- There was **no version control** in this repo until now; the earlier refactor
  destroyed the working `model.py` unrecoverably. Keep committing.
- Model input: 320×320 RGB, letterboxed on grey-114 (matches training augmentation).
