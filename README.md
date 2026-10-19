# CropGuard: Start Here

## 🌿 What it is

**CropGuard** is a multi-task diagnostic system for tomato leaf disease. One
forward pass gives a complete picture, not just a label:

- **Where** the disease is (detection)
- **What** it is (classification, 11 classes)
- **Which pixels** are affected (segmentation — see the caveat below)
- **How severe** it is (severity — see the caveat below)

Pipeline: `image → FasterNet-T0 backbone → BiFPN-lite neck → multi-task heads → JSON`.

Model: **2.63 M params**, 320×320 RGB input, runs on CPU.

## ⚠️ Read this before quoting numbers

Two heads are weak on the current checkpoint, and the codebase says so rather
than hiding it:

- **Severity is not usable.** `sev_r2 = -1.41` — worse than predicting a
  constant. The API returns `severity.reliable = false`; clients should hide it.
- **Segmentation is a leaf region, not a lesion map.** The corpus masks are filled
  leaf boxes, so the reported `seg_mIoU` 0.880 is leaf-region IoU. The API field is
  `leaf_fraction`, deliberately not `lesion_fraction`.

Full measured results, and what to do about them, are in **[HANDOFF.md](HANDOFF.md)**.

## 🏗 Architecture

`FasterNet-T0` (via `timm`, levels C3/C4/C5) → `BiFPN-lite` (softmax-weighted
bidirectional fusion) → four heads:

| head | design |
|---|---|
| Detection | YOLOv8-style decoupled head, DFL box encoding, 2100 anchors |
| Classification | SE → GAP → MLP, pooled over **P3+P4+P5** (multi-scale) |
| Segmentation | LightHam-lite decoder (Hamburger block + transposed-conv upsampling) |
| Severity | gated multimodal fusion of seg/det/cls embeddings + area ratio |

## 📁 Repository structure

```
model/
  architecture/   backbone.py, neck.py, heads.py, model.py   (canonical definitions)
  training/       train.py, dataset.py, losses.py, eval_tta.py, ...
  weights/        every checkpoint, named <version>_<accuracy>, + manifest + README
app/
  api/            FastAPI server (model_server.py, main.py) + eval + diagnostics
  web/            Next.js frontend
data_pipeline/    source_data.py, validate_dataset.py, augment_pipeline.py
mobile/           Flutter app (+ generate_real_assets.py for its sample assets)
docs/             architecture notes
resources/        research papers (references.bib, research_paper.tex)
```

## 🚀 Quick start

**Backend API**
```bash
cd app/api
pip install -r requirements.txt
MODEL_PATH=$PWD/cropguard_model_v5.pt python3 main.py    # http://localhost:8000
```
Endpoints: `/v1/health`, `/v1/predict`, `/v1/masks/{id}.png`, `/v1/cams/{id}.png`.

Verify it against **real** corpus images (not synthetic samples):
```bash
python3 verify_e2e_real.py 6          # posts photos, checks the full contract
python3 _check_v5_serving.py 12       # model-level checks incl. frontend vocabulary
```

**Web frontend**
```bash
cd app/web && npm install && npm run dev    # http://localhost:3000
```

**Training** — data root is set with `CROPGUARD_DATA` (no hardcoded paths):
```bash
CROPGUARD_DATA=/path/to/datasetclaude python3 model/training/train.py \
  --stages 1,2,3,4,5 --stage-epochs 10 --batch-size 32 --amp --out runs
```
A 4 GB-VRAM laptop is not enough for a full run; use Kaggle GPU (~5–6 h for all
50 epochs, ~350–400 s/epoch, 6.4 GB peak at batch 32).

## 📊 Data

11 classes, from `ganrajlongadge/cropguard-corpus-claude`:
29,611 train / 2,000 val / 1,286 test, with YOLO labels **and** masks, plus `data.yaml`.

## 🛠 Roadmap

1. **Real lesion masks + a real severity target** — the only change that fixes the
   two weak heads. `PlantSeg` (11,458 in-the-wild images, expert disease polygons,
   includes tomato) is the candidate; see HANDOFF.md.
2. Re-measure segmentation against real masks and correct the paper's numbers.
3. Wire the Flutter app to the live API (it ships bundled offline samples today).

*GUM (guided upsampling) and task-specific adapters were previously listed here as
roadmap items, then wrongly described as implemented. They are not in the model and
were removed from the serving code, which they had broken.*

---

*Professional implementation of hybrid deep learning for agricultural diagnostics.*
