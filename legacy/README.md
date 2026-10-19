# legacy/ — kept, but not part of the active build

Everything here is retained for history/traceability only. **Nothing in `app/`,
`model/`, `data_pipeline/` or `tests/` imports or runs these files.** If you are
looking for the current pipeline, see `STRUCTURE.md` at the repo root.

| folder | what it holds | why it moved |
|---|---|---|
| `app_api_diagnostics/` | `_diag_*.py`, `_check_*.py`, their outputs and `api.log` | one-off investigations (aspect-ratio, per-class degradation, mask sanity, V5 serving checks). They belong to completed debugging sessions, not to the shipped API. |
| `app_api_old_v2/` | the previous `app/api/_old_v2/` tree, incl. its stale tests and 16-class manifest | describes the v2 (16-class) model, which cannot even load into the nc=11 build. |
| `checkpoints_dupes/` | `cropguard_model_final.pt`, `cropguard_model_hybrid.pt`, `cropguard_model_v1_backup.pt` (11 MB each) | byte-identical copies of checkpoints already archived in `model/weights/`. Removing them from `app/api/` keeps the Docker build context lean; see `model/weights/WEIGHTS_MANIFEST.json` for the canonical copies. |
| `notebooks/train_cropguard.ipynb` | the pre-v6 Kaggle notebook | superseded by the v6 run (archived in `model/weights/reference/v6/`) and the v7 plan in `docs/`. |

## Restoring something

The files are untouched, so `git mv` (or a plain `mv`) them back if needed.
`app/api/self_test.py` used to depend on `app/api/_user_images/`; it now scans
`$CROPGUARD_TEST_PHOTOS`, `~/Downloads` and `app/api/_user_images/`, skipping
missing photos instead of erroring.
