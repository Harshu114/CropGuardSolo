# CropGuard Mask Annotation Pipeline — How the Training Masks Were Made

## What this document is for

This document records where the masks and labels used to train CropGuard came from, what annotation tooling was prepared in this project, what the released corpus actually contains, and how SAM fits into the picture.

It is written so the data preparation can be reproduced, audited, or reopened later by anyone working on the project.

## Short version

The project did **not** produce its released training masks by hand-labeling every image in a raw SAM session.

Instead, it:

1. prepared SAM-based annotation tooling,
2. pulled several public tomato disease sources,
3. normalized class names across sources,
4. validated and cleaned boxes and masks,
5. split the data with leakage checks,
6. exported a corpus with images + YOLO labels + binary leaf masks at 320x320, and
7. later repaired defective lesion masks for the harder segmentation/severity work.

So SAM is part of the annotation toolkit in this repo, but it is **not** the whole story, and the released corpus masks are not "SAM output as-is."

## What the project shipped for mask annotation

The repo contains a reusable annotation toolkit around SAM, plus the pipeline modules that clean and export whatever masks get generated.

### SAM annotation tooling

Relevant files:

- `ResourcesForCropGuard/data_prep/sam_annotation_colab.ipynb`
- `ResourcesForCropGuard/data_prep/SAM_ANNOTATION_GUIDE.md`
- `ResourcesForCropGuard/data_prep/scripts/sam_auto_label.py`
- `ResourcesForCropGuard/data_prep/scripts/sam_batch_annotator.py`

The Colab notebook and the SAM annotation guide describe a SAM-based workflow for bulk-annotating crop disease images. The main ideas are:

- load SAM or SAM-HQ,
- prompt with points or boxes,
- convert SAM masks into YOLO boxes or polygons,
- process images in batches,
- optionally save masks as PNGs,
- resume from an index if a run is interrupted.

There is also a grounded-SAM style path in `sam_auto_label.py`:

- GroundingDINO proposes boxes from text prompts for the disease vocabulary,
- SAM refines those boxes into masks,
- the result is written as YOLO labels and optionally as masks.

So the project did prepare a pipeline in which SAM can propose masks from prompts or boxes, and those masks can be exported in the same format the trainer expects.

### Roboflow as an alternative path

The SAM guide also documents a GUI route:

- upload to Roboflow,
- use Auto-Annotate with SAM,
- export in YOLO format.

That is recorded as an alternative, not as the only way the corpus was built.

## What the released corpus actually contains

From the corpus on disk and the project's own README, the published corpus is:

- `ClaudeCropGuard/datasetclaude/`
- with `images/`, `labels/`, `masks/`, `data.yaml`, and `dataset-metadata.json`

Corpus facts recorded in the project:

- 32,897 images in the Claude rebuild corpus
- 1:1 image / label / mask correspondence
- 11 unified classes
- splits in the neighborhood of 29,611 / 2,000 / 1,286
- masks are grayscale PNGs, mode `L`, 320x320, binary in practice with values 0 and 255

### Important semantic point

The released corpus masks are **leaf masks**, not lesion masks.

In the project's own words, the corpus masks are leaf-only binary masks. They tell you where the leaf is, but they do not, by themselves, encode a separate lesion region in the released files.

That distinction matters because:

- classification and detection labels are boxes,
- the released segmentation target is the leaf region,
- lesion-level segmentation and lesion-based severity require a lesion mask, which was handled later and separately.

## Where lesion masks came from

The project explicitly documents a mask-repair step for a later corpus version:

- `CropGuard_Final/data_pipeline/scripts/repair_masks_v9.py`

That script describes two concrete problems it fixed:

1. Lesion masks that were 1-pixel polygon outlines instead of filled regions. The fix was a dilate / flood-fill-outside / union / erode sequence.
2. Leaf masks that were healthy-tissue-only, leaving lesion areas as holes or outside the leaf. The fix was hole-filling on the leaf mask and union with the filled lesion region.

It also reports stats such as:

- lesion area inflation after filling,
- how many leaf masks changed,
- an assertion that lesion is inside leaf after repair.

So in this project, lesion masks were not trusted as raw annotations. They were programmatically inspected, repaired, and then used for the harder segmentation/severity metrics.

## The data-prep pipeline behind the masks and labels

The project did not just "run SAM and export." It prepared a larger pipeline for cleaning and exporting the corpus. The relevant modules are in:

- `ResourcesForCropGuard/data_prep/modules/`

Those modules include:

- dataset audit
- tomato relevance filtering
- class mapping / taxonomy normalization
- annotation validation
- annotation audit
- image quality checks
- duplicate detection, with MD5 exact match plus perceptual near-duplicate detection
- dataset splitting with group-aware stratification
- dataset statistics
- dataset exporter

The conceptual flow is:

1. audit every image and its annotations,
2. decide which classes belong in the tomato disease problem,
3. normalize equivalent disease names across sources into one ID space,
4. validate boxes: clip to image bounds, drop zero-area boxes, flag tiny or huge boxes,
5. check image quality,
6. detect exact and near-duplicates and assign group IDs,
7. split with leakage hygiene,
8. export to the final image/label/mask layout,
9. verify label/image correspondence and split integrity.

That is why the masks and labels in the released corpus should be thought of as the output of a cleaning and normalization pipeline, not just a raw SAM dump.

## What the data sources were

The project's data preparation materials describe a multi-source build, including PlantVillage-style classification images, Tomato-Village detection annotations, and other public tomato disease sources pulled through the Roboflow download helper.

Those were unified into:

- 11 classes in the released Kaggle corpus,
- a corpus on the order of 32,897 images in the Claude rebuild,
- splits of roughly 29,611 / 2,000 / 1,286 in that published corpus.

The project also kept explicit notes about coverage gaps, rare classes, and classes that were present in one source but not labeled in another.

## How SAM would actually fit if you reran annotation today

If you wanted to reproduce mask preparation with SAM from scratch using this repo's tooling, the intended shape of the work is roughly:

1. gather raw tomato images from the same kinds of public sources the project used,
2. run a SAM-based pass to propose masks, either:
   - from point/box prompts, or
   - from grounded detection boxes,
3. convert the masks into the project's expected formats:
   - leaf mask PNGs for the whole leaf region,
   - lesion masks only where you actually have lesion signal,
4. run the project's validation and cleaning modules on the result,
5. repair outlines and holes the way `repair_masks_v9.py` does,
6. split, verify, and export.

In other words, SAM is the mask proposer in this project's tooling, and the rest of the pipeline is what makes the result usable for training.

## Known limitations recorded in the project

The project is unusually explicit about what the masks do and do not support:

- the released corpus masks are leaf-only, so segmentation metrics on them are not the same thing as lesion segmentation metrics,
- there is no graded severity annotation in the corpus, so severity ends up weak unless a separate lesion mask is supplied,
- some classes are rare,
- some detection sources are missing detection labels for some classes,
- cross-split leakage was checked, but only to the extent of the methods used, so it is a measured lower bound rather than an absolute guarantee.

If you are writing a paper or report, those caveats belong next to any mask or severity claim.

## Files to point at internally

- SAM annotation guide: `ResourcesForCropGuard/data_prep/SAM_ANNOTATION_GUIDE.md`
- SAM Colab notebook: `ResourcesForCropGuard/data_prep/sam_annotation_colab.ipynb`
- Grounded SAM auto-label script: `ResourcesForCropGuard/data_prep/scripts/sam_auto_label.py`
- SAM batch annotator: `ResourcesForCropGuard/data_prep/scripts/sam_batch_annotator.py`
- Data-prep modules: `ResourcesForCropGuard/data_prep/modules/*.py`
- Dataset preparation report: `ResourcesForCropGuard/data_prep/DATASET_PREPARATION_REPORT.md`
- Claude corpus README: `ClaudeCropGuard/README.md`
- Roboflow download helper: `ClaudeCropGuard/download_roboflow.py`
- Mask repair script: `CropGuard_Final/data_pipeline/scripts/repair_masks_v9.py`
- Released corpus root, when present: `ClaudeCropGuard/datasetclaude/`

## Bottom line

The CropGuard masks were made through a SAM-aware annotation workflow plus a full data-prep, validation, and repair pipeline, not by manually annotating every image inside SAM.

The released corpus gives you binary leaf masks and YOLO box labels at 320x320. Lesion masks, where they matter, were produced later and repaired before being used for lesion/severity work.

If you reannotate, SAM is the mask-generation engine you plug into this existing pipeline; the pipeline is what turns raw SAM proposals into training-ready masks.
