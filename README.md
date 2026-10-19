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

Two halves, deliberately split, because the images are already public elsewhere
and re-hosting 2.2 GB of photos nobody needs twice is waste.

**Images — Kaggle, anonymous download, no `dvc` required.**
`https://www.kaggle.com/api/v1/datasets/download/harshalnandeshwar/cropguard-fully-annotated`
2,300,126,145 bytes, CC BY 4.0, derived from Roboflow `harshal-nandeshwar/maskforcropguard`.
29,280 image files here — 20,488 train / 4,384 val / 4,408 test — which is the
upstream 29,279 images plus one derived `_cropguard` crop in the test split.

**Annotations and checkpoints — DVC.** Fetch with `dvc pull` after cloning:

| out | contents |
|---|---|
| `data/dataset_v3_clean/labels_v3` | 14,343 YOLO files, `class cx cy w h`, train split only |
| `data/dataset_v3_clean/masks_v3` | 14,343 binary PNG masks, mode L, values {0, 255}, at mixed sizes (256×256 and 640×640) |
| `data/dataset_v3_clean/qa_sheets` | 16 per-class contact sheets |
| `model/weights` | 55 checkpoints, 565 MB |

Three things to know before you train on this, all measured rather than inherited
from older notes in this repo:

1. **The label class space is 16, not 11.** `data/dataset_v3_clean/data.yaml` gives
   `nc: 16` and the names. The served head is `NUM_CLASSES = 11`
   (`app/api/model_server.py:50`), which is a *subset* of those 16. No 16→11 mapping
   exists in this repository, so `train.py` will not consume these labels unmodified.
   Only ids 1–13 and 15 ever appear as boxes; `0` (Healthy) carries no box by
   definition and `14` (tomato) is unused.
2. **Annotations cover the train split only.** `labels_v3` and `masks_v3` contain
   14,343 files each and no `val`/`test` directories, while images exist for all
   three splits. So 6,145 of the 20,488 train images are unlabeled, and val/test
   cannot be scored from this corpus alone.
3. **Masks are binary leaf masks, not lesion masks** — see the segmentation caveat
   above and HANDOFF.md.

`review_list.txt` names the 4,853 images the audit flagged for human review. The
`review/` directory itself is not published: every file in it is a byte-identical
copy of a file already in `images/`, so it would only re-upload photographs the
Kaggle dataset already serves.

*An earlier version of this section pointed at `ganrajlongadge/cropguard-corpus-claude`
and quoted 29,611 / 2,000 / 1,286 splits. That dataset is not publicly retrievable,
and those splits do not describe this corpus.*

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
