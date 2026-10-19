# Repo structure — what is live, what is not

Three things exist side by side on purpose:

| | architecture | status |
|---|---|---|
| **v5** | `model/architecture/model.py` (+ `app/api/model.py` serving copy) | shipped; checkpoint loads strictly (292 tensors, 2.63 M params) |
| **v6** | same code as v5, retrained | COMPLETE Kaggle run, archived in `model/weights/reference/v6/` |
| **v7** | `model/architecture/v7_model.py` (+ `app/api/model_v7_serving.py`) | NEW, smoke-tested, no checkpoint yet |

## Live paths

```
model/architecture/     model.py        v5 (frozen — do not edit; serving copy is asserted equal)
                        v7_model.py     v7 assembly
                        v7_heads.py     v7 heads (bilinear seg + lesion-gated cls + abstention)
                        backbone.py     fasternet_t0 (v5: 3 levels, v7: 4 with a /4 skip)
                        neck.py         BiFPNLite (shared by v5 and v7)
                        heads.py        v5 heads (frozen with model.py)
model/training/         train.py        staged training, --arch {v5,v7}, EMA, GT-mask schedule
                        losses.py       multi-task losses (+ 2-channel seg for v7)
                        consistency.py  v7 cross-task term (healthy <=> no lesion)
                        dataset.py      loader + background-paste augmentation
                        eval_tta.py     metrics incl. per-class F1 and ECE
                        calibrate_abstention.py   temperature scaling + thresholds
                        export_onnx.py  ONNX export + CPU latency (+ cascade bench)
                        train_kaggle.py / eval_tta.py / rank_ckpts.py   run helpers
model/weights/          archived checkpoints (WEIGHTS_MANIFEST.json lists scores)
                        reference/v6/   the finished v6 run (baseline for v7)
data_pipeline/scripts/  prepare_plantseg.py  PlantSeg -> corpus layout
                        derive_dataset.py    derive ALL labels (boxes/leaf/severity/negatives)
                        dedup_split.py       perceptual-hash dedup + duplicate-aware split
                        clean_corpus.py / validate_dataset.py / source_data.py
                        augment_pipeline.py  offline augmentation
data_pipeline/configs/  class_map.yaml  PlantSeg -> CropGuard class mapping + label policy
app/api/                model.py, model_v7_serving.py, model_server.py, main.py, ...
                        cropguard_model_v5.pt   the only checkpoint in the build context
tests/                  test_architecture_consistency.py   v5 AND v7 serving/canonical parity
docs/                   V7_RUNBOOK.md   exact order of operations for a v7 run
legacy/                 moved-out diagnostics, old v2 tree, duplicate checkpoints
```

## Conventions that keep this honest

- `model.py` and `app/api/model.py` must stay equal (the API cannot import the
  repo-root package). `tests/` asserts identical state_dicts and identical outputs
  for BOTH the v5 and v7 pairs. Changing one copy without the other fails loudly.
- The v5 checkpoint must keep loading strictly into `model.py`. v7 changes live in
  new symbols (`CropGuardMultiTaskV7`, `v7_*`) rather than edits to v5 classes.
- `CropGuardMultiTask.arch_name == "v7"` is what makes `train.py` switch loss
  paths; v5 keeps its default loss behaviour untouched.
