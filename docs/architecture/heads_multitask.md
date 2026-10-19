# CropGuard Architecture: Multi-Task Heads

## 📌 Overview
CropGuard employs a specialized multi-head architecture that allows a single forward pass to perform four distinct diagnostic tasks. This holistic approach ensures that the findings of one task (e.g., classification) inform the result of another (e.g., severity).

## 🛠 Technical Specifications

### 1. Detection Head (`DecoupledDetHead`)
- **Design:** YOLOv8-style decoupled head.
- **Logic:** Separates the **Classification** (what is this object?) and **Bounding Box** (where is it?) branches to avoid conflicting gradients.
- **Output:** Bounding boxes for diseased regions using DFL (Distribution Focal Loss) encoding.

### 2. Classification Head (`ClassificationHead`)
- **Design:** SE $\rightarrow$ GAP $\rightarrow$ MLP.
- **Logic:** Uses **Squeeze-and-Excitation (SE)** blocks on all pyramid levels (P3, P4, P5).
- **Novelty:** By pooling from multiple levels, it captures both the *fine texture* of the lesion and the *global context* of the leaf.

### 3. Segmentation Head (`LightHamLiteDecoder`)
- **Design:** Hamburger-lite global context unit.
- **Logic:** Uses a **Hamburger block** (a lightweight attention mechanism) to refine the segmentation mask.
- **Output:** Pixel-level masks of the diseased areas, upsampled to the original image size.

### 4. Severity Head (`SeverityHead`)
- **Design:** Gated Multimodal Fusion.
- **Logic:** Fuses embeddings from the Detection, Classification, and Segmentation heads, along with the total lesion area ratio.
- **Output:** 
    - **Continuous Score:** A value between $0.0$ (Healthy) and $1.0$ (Severe).
    - **Ordinal Stage:** Classification into severity stages (e.g., Mild, Moderate, Severe).

## 🧠 Why This Design?
Traditional models usually perform these tasks in sequence (Detect $\rightarrow$ Crop $\rightarrow$ Classify). CropGuard's parallel architecture:
1. **Reduces Latency:** All results are generated in one pass.
2. **Increases Accuracy:** The Severity head uses the *actual* segmentation mask and classification confidence to calculate the score, mimicking how a plant pathologist works.

## 📂 File Reference
`CropGuard_Final/model/architecture/heads.py`
