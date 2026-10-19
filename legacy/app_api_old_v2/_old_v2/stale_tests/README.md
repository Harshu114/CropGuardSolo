# Stale test scripts (quarantined, do NOT run)

These were written against the **old 16-class v2/v3-hybrid** model. They hardcode
`nc=16` and reference checkpoints that no longer exist
(`cropguard_model_final.pt`, `v3_hybrid/student_best.pt`), so they fail or, worse,
silently build a 16-class network and load it with a mismatched head.

The live model is now v5, **11 classes**. Working replacements live in the parent
`api/` directory:

| use | script |
|---|---|
| full serving verification (mask/CAM/boxes/class vocabulary) | `api/_check_v5_serving.py` |
| single-image HTTP prediction + mask/CAM round-trip | `api/verify_predict.py` |
| batch field-photo report over real photos | `api/self_test.py` |
