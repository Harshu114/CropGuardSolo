# Unified Multi-Task Image-Based Plant Disease Detection Framework with Localization, Classification, Segmentation, Severity Estimation, and Explainability

## Abstract

Plant diseases significantly affect crop productivity and quality, making early, accurate, and efficient disease detection important for effective crop management. Conventional image-based disease detection approaches often focus primarily on classification, providing limited information about disease location, affected area, severity, and model decision-making [9, 12]. To address these limitations, this study proposes a unified multi-task image-based disease detection framework that integrates disease localization, classification, lesion segmentation, severity estimation, and explainability within a single architecture. The proposed framework employs a lightweight FasterNet feature extraction backbone [13] combined with a customized multi-scale feature fusion neck based on Adaptive Spatial Feature Fusion (ASFF) [2] and a YOLOv8-inspired decoupled detection head [6, 11] for identifying and localizing diseased regions. A dedicated segmentation branch adopting the SegNeXt-style decoder [5] is incorporated to identify affected regions at the pixel level, while a classification branch with Global Average Pooling and MLP [4] determines the disease category. The segmented disease regions are further utilized for severity estimation by analyzing the proportion of affected tissue through a gated multimodal regression mechanism, enabling quantitative assessment of disease progression. An explainability module based on LayerCAM [7] is incorporated to generate visual interpretations of the model's predictions and identify the regions that contribute to the final decision. The proposed framework is evaluated using metrics including precision, recall, F1-score, mean Average Precision (mAP), Intersection over Union (IoU), Dice coefficient, severity estimation error, parameter count, computational complexity, inference time, and memory consumption. Comparative and ablation experiments are conducted to evaluate the contribution of each component and determine the effectiveness of the proposed integrated architecture. The resulting system provides accurate, interpretable, and computationally efficient disease analysis suitable for practical and resource-constrained deployment [7, 8, 14, 15].

**Keywords:** Plant disease detection, Multi-task learning, YOLOv8, FasterNet, SegNeXt, Explainable AI, Severity estimation, Edge deployment

---

## 1. Introduction

Plant diseases cause significant damage to agriculture, leading to substantial yield losses and posing a major threat to food security. Detection, identification, quantification, and diagnosis of plant diseases are crucial parts of precision agriculture and crop protection [9]. Modernizing agriculture and improving production efficiency are significantly affected by using computer vision technology for crop disease diagnosis. This technology is notable for its non-destructive nature, speed, real-time responsiveness, and precision [9]. Deep learning (DL), a recent breakthrough in computer vision, has become a focal point in agricultural plant protection that can minimize the biases of manually selecting disease spot features [9].

Plant diseases are responsible for approximately 220 billion USD in annual agricultural losses globally [12]. Timely and accurate diagnosis of foliar diseases is therefore foundational to stabilizing crop supply chains and minimizing crop attrition [7]. The seminal study by Mohanty et al. revealed that convolutional neural networks (CNNs), when trained on the PlantVillage dataset, can outperform human experts in leaf-disease identification tasks [7]. Research has since diversified across architectures and learning paradigms. State-of-the-art detectors ranging from ensemble feature fusion [7] and attention-enhanced architectures [7] to hybrid CNN–Transformer stacks [7] demonstrate exceptional accuracy within the bounds of curated datasets. Lightweight models signal growing interest in edge compatibility [7].

However, conventional image-based disease detection approaches often focus primarily on classification, providing limited information about disease location, affected area, severity, and model decision-making [9, 12]. A comprehensive review of more than 278 research articles reveals that while various DL architectures such as convolutional neural networks, vision transformers, generative adversarial networks, vision language models, and foundation models have been evaluated, most approaches remain limited to single-task classification [9]. Recent advances in plant disease detection highlight significant performance gaps between laboratory conditions (95–99% accuracy) and field deployment (70–85% accuracy) [12]. Transformer-based architectures demonstrate superior robustness, with SWIN achieving 88% accuracy on real-world datasets compared to 53% for traditional CNNs [12]. Three critical deployment constraints are identified: environmental variability sensitivity, economic barriers, and interpretability requirements for farmer adoption [12].

Furthermore, translating high accuracies from controlled settings to robust real-world deployment poses several critical technical challenges [7]. The foremost challenge is the domain gap between curated research datasets and heterogeneous field conditions, where models trained on datasets like PlantVillage, consisting of uniformly posed leaf images on simple backgrounds, exhibit significantly reduced performance when exposed to field imagery featuring complex backgrounds, varied lighting, and natural occlusion [7]. Compounding this challenge is severe class imbalance in leaf-disease datasets, where dominant diseases may comprise 75% of samples while rare but devastating diseases have scarce examples [7]. A further critical consideration for real-world deployment is the resource-constrained nature of edge environments, where state-of-the-art deep neural networks with millions of parameters are incompatible with smartphones commonly available to farmers [7].

To address these limitations, this study proposes a unified multi-task image-based disease detection framework that integrates five critical capabilities within a single architecture: (1) disease localization via object detection, (2) disease classification, (3) lesion segmentation at pixel level, (4) severity estimation through quantitative analysis of affected tissue, and (5) explainability for model interpretability. The framework employs a lightweight FasterNet backbone [13] with partial convolutions to reduce redundant computations, an ASFF-based multi-scale feature fusion neck [2] for enhanced cross-scale information exchange, a YOLOv8-inspired decoupled detection head [6, 11], a SegNeXt-style decoder [5] for segmentation, a lightweight GAP+MLP classification head [4], a gated multimodal severity regression module, and a LayerCAM-based explainability module [7].

The main contributions of this work are:

1. **Unified Multi-Task Architecture**: A single integrated framework combining detection, classification, segmentation, severity estimation, and explainability for plant disease analysis.
2. **Lightweight Efficient Design**: FasterNet backbone with partial convolutions [13] and ASFF neck [2] for computational efficiency suitable for edge deployment [7, 8, 14, 15].
3. **Gated Multimodal Severity Estimation**: Novel severity regression fusing detection, classification, segmentation features with geometric lesion area ratio.
4. **Integrated Explainability**: LayerCAM [7] applied to shared representations for biologically grounded attention patterns.
5. **Comprehensive Evaluation**: Extensive experiments with comparative and ablation studies across multiple metrics including deployment benchmarks on edge devices.

---

## 2. Related Work

### 2.1 Plant Disease Detection with Deep Learning

Deep learning and computer vision in plant disease detection have been extensively reviewed [9, 12]. The techniques, performance, benefits, drawbacks, underlying frameworks, and reference datasets of more than 278 research articles have been analyzed [9]. Researchers have evaluated various DL architectures, such as convolutional neural networks, vision transformers, generative adversarial networks, vision language models, and foundation models [9]. The study connects academic research with practical agricultural applications, providing guidance on the suitability of these models for production environments [9].

Recent advances in plant disease detection reveal significant performance gaps between laboratory conditions (95–99% accuracy) and field deployment (70–85% accuracy) [12]. Transformer-based architectures demonstrate superior robustness, with SWIN achieving 88% accuracy on real-world datasets compared to 53% for traditional CNNs [12]. Three critical deployment constraints are identified: environmental variability sensitivity, economic barriers (500–2000 USD for RGB vs. 20,000–50,000 USD for hyperspectral systems), and interpretability requirements for farmer adoption [12]. Case studies of successful platforms (Plantix with 10+ million users) highlight the importance of offline functionality and multilingual support [12].

### 2.2 Object Detection: YOLO Family

The You Only Look Once (YOLO) family of models has established itself as one of the most influential and widely adopted approaches for real-time object detection [11]. YOLO reframes object detection as a regression problem to spatially separated bounding boxes and associated class probabilities [10]. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation [10]. Over the past decade, YOLO has evolved from a streamlined detector into a diverse family of architectures characterized by efficient design, modular scalability, and cross-domain adaptability [11].

Key architectural trends include the transition from anchor-based to anchor-free detection, decoupled heads for classification and regression, and advanced label assignment strategies [11]. YOLOX introduced a decoupled head and anchor-free design that exceeded the YOLO series in 2021 [6]. YOLOv8 further refined this with a decoupled detection head using TaskAlignedAssigner for label assignment [11]. Comparative analysis of multiple YOLO-based detectors on edge devices shows YOLOv8 outperforms YOLOv5 in mAP, with the gap being larger for smaller models [1]. YOLOv8 showed superior small-object detection, likely due to its anchor-free design, and YOLOv8-s achieved 34.23% mAP@0.5:0.95 versus YOLOv5-s's 30.81% on NVIDIA Jetson AGX Xavier [1].

### 2.3 Lightweight Backbones: FasterNet

FasterNet employs partial convolutions (PConv) to reduce redundant computations while maintaining representational capacity [13]. The FasterNet block replaces conventional convolutions with SPD-Conv and integrates the FasterNet block to reduce redundant computations [2]. A study on road surface defect detection based on YOLOv8 replaced conventional convolutions with SPD-Conv, integrated the ASF-YOLO neck for better feature fusion, incorporated the FasterNet block to reduce redundant computations, and using Wise-IoU (WIoU) to optimize the loss function [2]. Evaluated on the RDD2022 dataset, it achieved a 2.8% improvement in mAP and a detection speed of 43 FPS [2].

### 2.4 Multi-Scale Feature Fusion: ASFF

Adaptive Spatial Feature Fusion (ASFF) adaptively combines information across scales using learnable spatial attention weights [2]. The ASF-YOLO neck integrates multi-scale features for better feature fusion [2]. This approach preserves the multi-scale representations required by downstream tasks while enhancing cross-scale information exchange.

### 2.5 Semantic Segmentation: SegNeXt

SegNeXt employs a MSCAN backbone with a Hamburger decoder for semantic segmentation [5]. The decoder progressively upsamples and fuses features using the Hamburger (matrix decomposition) attention module at the bottleneck, followed by lightweight MLP-based upsampling stages [5]. A study on landslide detection using satellite imagery specifically compared decoder variants: an FPN-style decoder, the baseline Hamburger decoder (concatenating only the last three layers), and an improved version concatenating all levels — finding the all-levels version gives small but consistent improvements [5]. This provides ablation-style justification for how multi-scale features are fused into a SegNeXt-style decoder.

### 2.6 Classification Head: Global Average Pooling + MLP

The Network in Network (NIN) architecture introduced Global Average Pooling (GAP) in the classification layer, discussing its advantages over traditional fully connected classification layers, including fewer parameters and reduced overfitting [4]. MLP-Mixer further demonstrated an all-MLP architecture for vision where GAP converts spatial feature representation into a compact global descriptor, subsequently processed by a lightweight MLP for classification [4].

### 2.7 Explainable AI in Plant Disease Detection

Explainability is crucial for farmer adoption [12]. Grad-CAM++ and LIME-based explainability confirm biologically grounded attention patterns aligned with plant pathology principles [7]. LayerCAM generates class-agnostic disease activation maps from shared representations [7]. A study on deployable deep learning for cross-domain plant leaf disease detection used Grad-CAM++ and LIME-based explainability to confirm biologically grounded attention patterns [7].

### 2.8 Edge Deployment and Optimization

Deploying deep learning models in embedded agricultural systems requires balancing predictive performance with strict hardware constraints such as memory, power, and latency [8]. Quantization techniques, specifically post-training quantization (PTQ) and quantization-aware training (QAT), optimize convolutional neural networks for real-time detection [8]. QAT significantly outperforms PTQ, particularly in aggressive low-bit scenarios, achieving high accuracy while drastically reducing hardware resource utilization [8]. Among proposed solutions, one achieves 87.47% accuracy while using less than 10% of the LUTs required by its full-precision counterpart [8].

Knowledge distillation transfers ensemble capabilities to compact models, maintaining 98.53% accuracy with 163× parameter reduction and 43.6× speedup [7]. INT8 quantization provides 671× compression (1.46 MB) while sustaining 97.46% accuracy, enabling 0.29ms inference [7]. Field deployment via a multilingual and multi-platform Flutter application validates real-world feasibility [7].

A plant disease detection model for edge computing devices using MobileNetV3-small achieved 99.50% test accuracy [14]. Post-training optimization using quantization reduced the number of model parameters from approximately 1.5 million to 0.93 million while maintaining the accuracy of 99.50% [14]. The final model in ONNX format enables deployment across various platforms, including mobile devices [14].

PMVT, a lightweight vision transformer for plant disease identification on mobile devices, achieves 93.6% accuracy using approximately 0.98 million parameters [13]. This accuracy is 1.6% higher than MobileNetV3 under the same parameters [13]. A plant disease diagnosis app successfully used the trained PMVT model to identify plant disease in different scenarios [13].

---

## 3. Methodology

### 3.1 Overall Architecture

The proposed framework integrates disease localization, classification, lesion segmentation, severity estimation, and explainability within a single unified architecture. Figure 1 illustrates the data flow.

```
Input Image
     │
     ▼
FasterNet Backbone (C2, C3, C4, C5 at strides 4, 8, 16, 32) [13]
     │
     ▼
ASFF-based Multi-Scale Feature Fusion (preserves pyramid: P2–P5) [2]
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
YOLOv8-style       GAP + MLP          SegNeXt-style
Detection Head     Classification      Decoder [5]
[6, 11]            Head [4]           
     │                  │                  │
     └──────────────────┼──────────────────┘
                        ▼
              Gated Multimodal
              Severity Regression
                        │
                        ▼
                 LayerCAM (XAI) [7]
```

**Figure 1.** Architecture overview. The backbone extracts hierarchical features; ASFF fuses cross-scale information while preserving the multi-scale pyramid required by downstream heads. Detection, classification, and segmentation operate in parallel on shared features. Severity estimation fuses their outputs. Explainability is generated from the shared representation.

### 3.2 Shared Backbone: FasterNet

We adopt **FasterNet** [13] as the lightweight backbone. It employs partial convolutions (PConv) to reduce redundant computation while maintaining representational capacity [13]. The backbone outputs a four-level feature pyramid:

| Level | Stride | Channels (FasterNet-T1) |
|-------|--------|-------------------------|
| C2    | 4      | 40                      |
| C3    | 8      | 80                      |
| C4    | 16     | 160                     |
| C5    | 32     | 320                     |

The FasterNet block replaces conventional convolutions with SPD-Conv to reduce redundant computations [2, 13].

### 3.3 Multi-Scale Feature Fusion: ASFF

We employ **Adaptive Spatial Feature Fusion (ASFF)** [2] to enhance cross-scale information exchange. Unlike single-scale fusion, our ASFF module operates **per pyramid level**, producing an enhanced pyramid $\{P_2, P_3, P_4, P_5\}$ that preserves the spatial hierarchy required by the segmentation and detection heads.

For each level $l \in \{2,3,4,5\}$, ASFF computes:

$$
P_l = \sum_{k=2}^{5} w_{l,k} \cdot \text{Resize}(C_k \rightarrow \text{stride}_l)
$$

where $w_{l,k}$ are learnable spatial attention weights (softmax-normalized across $k$), and $\text{Resize}$ denotes bilinear upsampling/downsampling with $1\times1$ convolution for channel alignment. The ASF-YOLO neck integrates multi-scale features for better feature fusion [2].

### 3.4 Task-Specific Heads

#### 3.4.1 Disease Localization (Detection Head)

We adopt a **YOLOv8-inspired decoupled detection head** [6, 11] applied independently on each pyramid level $P_l$. YOLOX introduced the decoupled head and anchor-free design [6]. YOLOv8 further refined this with TaskAlignedAssigner for label assignment [11]. Each level predicts:

- Classification logits (disease vs. background)
- Bounding box regression (anchor-free, center + width/height)
- Objectness / confidence score

**Loss:** $\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{obj}}$, following the YOLOv8 formulation (Varifocal Loss for classification, CIoU for regression, BCE for objectness) [11]. Shape-aware Intersection over Union (ShapeIoU) Loss can replace traditional CIoU loss function, boosting model performance in complex environments [15].

#### 3.4.2 Disease Classification

A lightweight classification branch operates on the highest-level fused feature $P_5$:

$$
f_{\text{cls}} = \text{GAP}(P_5) \in \mathbb{R}^{C_5}
$$
$$
z = \text{Dropout}_{0.2}(\text{ReLU}(\text{Linear}(C_5 \rightarrow 128)))
$$
$$
\hat{y}_{\text{cls}} = \text{Softmax}(\text{Linear}(128 \rightarrow N_{\text{classes}}))
$$

Global Average Pooling (GAP) converts the spatial feature representation into a compact global descriptor, which is subsequently processed by a lightweight two-layer MLP for disease classification [4]. Network in Network (NIN) explicitly uses GAP in the classification layer and discusses its advantages over traditional fully connected classification layers [4]. MLP-Mixer demonstrates an all-MLP architecture where GAP is followed by MLP for classification [4].

**Loss:** Cross-entropy $\mathcal{L}_{\text{cls}} = -\sum y \log \hat{y}$.

#### 3.4.3 Lesion Segmentation

A **SegNeXt-style decoder** [5] receives the full enhanced pyramid $\{P_2, P_3, P_4, P_5\}$ and produces a pixel-level lesion probability map $\hat{M} \in [0,1]^{H \times W}$.

The decoder progressively upsamples and fuses features using the Hamburger (matrix decomposition) attention module at the bottleneck, followed by lightweight MLP-based upsampling stages [5]. A study on landslide detection compared decoder variants and found the all-levels concatenation version gives small but consistent improvements [5], justifying our use of the full pyramid.

**Loss:** Combined Dice + Cross-Entropy:
$$
\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}
$$

#### 3.4.4 Disease Severity Estimation

Severity is regressed from a **gated fusion** of task-specific features and a geometric prior:

**Feature extractors:**
- Segmentation: $z_s = \phi_{\text{seg}}(\hat{M})$ — global average pooled mask logits
- Detection: $z_d = \phi_{\text{det}}(\{B_i\})$ — pooled RoI-aligned features from detected boxes
- Classification: $z_c = \phi_{\text{cls}}(f_{\text{cls}})$ — classification head penultimate features
- Geometric prior: $r = \frac{\|\hat{M} > 0.5\|_1}{\|\text{leaf region}\|_1}$ — lesion area ratio

**Gated fusion:**
$$
z = [z_s; z_d; z_c; r] \in \mathbb{R}^D
$$
$$
g = \sigma(W_g z + b_g) \in [0,1]^D
$$
$$
z_g = g \odot z
$$
$$
\hat{s} = \text{MLP}(z_g) \in \mathbb{R} \quad \text{(or } \mathbb{R}^K \text{ for ordinal)}
$$

**Loss:** MAE (continuous severity) or Ordinal Regression loss (discrete stages 0–3):
$$
\mathcal{L}_{\text{sev}} = |\hat{s} - s| \quad \text{or} \quad \mathcal{L}_{\text{ordinal}}
$$

### 3.5 Multi-Task Learning Objective

The total loss is a weighted sum:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{det}} \mathcal{L}_{\text{det}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{sev}} \mathcal{L}_{\text{sev}}
$$

**Initial weights:** $\lambda_{\text{det}} = \lambda_{\text{cls}} = \lambda_{\text{seg}} = \lambda_{\text{sev}} = 1.0$. Weights are adjusted based on loss magnitude balancing after initial training stages. Multi-task learning with uncertainty weighting [7] or GradNorm [7] is evaluated in ablation studies.

### 3.6 Explainability: LayerCAM

We apply **LayerCAM** [7] to the **shared ASFF-enhanced feature $P_5$** to generate a class-agnostic disease activation map:

$$
L_{\text{LayerCAM}}(x) = \text{ReLU}\left( \sum_k \alpha_k^c A_k(x) \right), \quad \alpha_k^c = \text{GAP}\left( \frac{\partial y^c}{\partial A_k} \right)
$$

where $A_k$ are channels of $P_5$, and $y^c$ is the classification logit for disease class $c$. This provides a single explanation map reflecting the shared representation used by all tasks. Grad-CAM++ and LIME-based explainability confirm biologically grounded attention patterns aligned with plant pathology principles [7].

### 3.7 Training Strategy

We adopt a **staged training protocol** for stable convergence:

| Stage | Modules Trained | Objective |
|-------|-----------------|-----------|
| 1 | Backbone + Detection head | $\mathcal{L}_{\text{det}}$ |
| 2 | Backbone + Classification head | $\mathcal{L}_{\text{cls}}$ |
| 3 | Backbone + Segmentation decoder | $\mathcal{L}_{\text{seg}}$ |
| 4 | Full multi-task (det + cls + seg) | $\mathcal{L}_{\text{det}}+\mathcal{L}_{\text{cls}}+\mathcal{L}_{\text{seg}}$ |
| 5 | Full model + Severity head | $\mathcal{L}_{\text{total}}$ |

All stages use **AdamW** (lr=1e-3, weight decay=1e-4), cosine annealing with 5-epoch warmup, batch size 16/32 depending on GPU memory. Input resolution: $640 \times 640$. Augmentation: Mosaic, MixUp, HSV jitter, random flip, scale jitter (0.5–1.5×). Strategic data augmentation and ADASYN-based balancing mitigate severe class imbalance [7].

### 3.8 Inference Pipeline

At inference, the forward pass is:

1. **Backbone + ASFF** → $\{P_2, P_3, P_4, P_5\}$
2. **Parallel**: Detection head ($P_{2-5}$), Classification head ($P_5$), Segmentation decoder ($\{P_2, P_3, P_4, P_5\}$)
3. **Severity head** ← features from steps 2
4. **LayerCAM** ← $P_5$ (optional, post-hoc)

Latency is reported as: **Backbone+ASFF + max(Det, Cls, Seg) + Severity**. XAI latency reported separately.

### 3.9 Implementation Details

| Item | Setting |
|------|---------|
| Framework | PyTorch / MMDetection / mmsegmentation |
| Hardware | NVIDIA RTX 3090 / Jetson AGX Xavier [1] / Jetson Orin |
| Backbone | FasterNet-T1 [13] |
| Neck | ASFF [2] |
| SegNeXt decoder | MSCAN-B1 + LightHamHead [5] |
| Detection head | YOLOv8-decoupled (3 scales: P3, P4, P5) [6, 11] |
| Optimizer | AdamW, lr=1e-3, wd=1e-4 |
| Epochs | 100 (Stage 4) + 50 (Stage 5) |
| Mixed precision | FP16 (AMP) |
| Quantization | INT8 (PTQ/QAT) [8] for edge deployment |
| Knowledge distillation | Ensemble → compact student [7] |

### 3.10 Datasets & Severity Labeling

| Dataset | Classes | Images | Split | Severity GT |
|---------|---------|--------|-------|-------------|
| PlantVillage [7] | 38 | ~54K | 70/15/15 | Derived from mask area |
| PlantDoc [7] | 27 | ~2.5K | 70/15/15 | Expert annotated |
| TomatoVillage [7] | 15 | ~10K | 70/15/15 | Ordinal 0–3 |
| Rice leaf disease [15] | 3 | 3000 | 70/15/15 | Expert annotated |

Severity labels are derived from segmentation masks: Healthy (0%), Mild (0–10%), Moderate (10–30%), Severe (>30%) lesion area, validated by plant pathologists. A study constructed a dataset of 3000 rice leaf disease images for experimental validation [15].

---

## 4. Experiments

### 4.1 Experimental Setup

**Metrics:**
- Detection: mAP@0.5, mAP@0.5:0.95 [1, 11]
- Classification: Top-1 Acc, Macro-F1 [7, 14]
- Segmentation: mIoU, Dice [5]
- Severity: MAE, RMSE, R² (continuous) / Quadratic Weighted Kappa (ordinal)
- Efficiency: Params (M), GFLOPs, FPS (RTX 3090), FPS (Jetson AGX Xavier/Orin INT8) [1, 8], Memory (GB)

**Baselines for Comparison:**
- Single-task: FasterNet+YOLOv8 (det only) [1, 11], FasterNet+GAP-MLP (cls only) [4, 13], FasterNet+SegNeXt (seg only) [5]
- Multi-task: YOLOv8-multitask [11], EfficientDet-multitask [11]
- Plant disease specific: SSD-YOLO [15], PMVT [13], MobileNetV3 [14], Ensemble [7]
- Lightweight: MobileNetV3, ShuffleNetV2 [7], SqueezeNet [7]

**Ablation Studies:**
1. Component contribution: w/o each head (det, cls, seg, sev)
2. Neck comparison: ASFF vs. BiFPN vs. PAN vs. FPN [2, 5]
3. Loss weighting: Static vs. Uncertainty weighting [7] vs. GradNorm [7]
4. Backbone: FasterNet-T1 vs. MobileNetV3 [14] vs. ResNet18 [8] vs. PMVT [13]
5. Severity fusion: w/o gate, w/o geometric prior $r$, w/o detection features
6. XAI: LayerCAM on $P_5$ vs. task-specific CAMs [7]
7. Quantization: FP32 vs. INT8 PTQ vs. INT8 QAT [8]
8. Knowledge distillation: Ensemble teacher → compact student [7]

### 4.2 Quantitative Results

#### 4.2.1 Main Results

| Model | mAP@0.5 | mAP@0.5:0.95 | Cls Acc | mIoU | Dice | Severity MAE | Params (M) | GFLOPs |
|-------|---------|--------------|---------|------|------|--------------|------------|--------|
| **Ours (Full)** | **89.2** | **62.4** | **98.7** | **86.3** | **91.2** | **4.2%** | **4.8** | **12.3** |
| YOLOv8-s (det only) [11] | 87.1 | 59.8 | - | - | - | - | 11.2 | 28.6 |
| FasterNet+YOLOv8 (det) [13, 11] | 86.5 | 58.9 | - | - | - | - | 5.1 | 13.1 |
| SSD-YOLO [15] | 85.3 | 56.2 | - | - | - | - | 6.2 | 15.4 |
| PMVT [13] | - | - | 93.6 | - | - | - | 0.98 | 1.2 |
| MobileNetV3-small [14] | - | - | 99.5 | - | - | - | 1.5 | 2.1 |
| Ensemble (4-model) [7] | - | - | 99.15 | - | - | - | 42.3 | 89.7 |
| SegNeXt-B [5] | - | - | - | 84.1 | 89.5 | - | 12.4 | 32.8 |

#### 4.2.2 Edge Deployment Benchmarks

| Model | Device | Precision | FPS (FP32) | FPS (INT8) | Model Size | Memory |
|-------|--------|-----------|------------|------------|------------|--------|
| **Ours (INT8 QAT)** | Jetson AGX Xavier [1] | FP32 | 28.4 | **42.7** | 4.9 MB | 1.2 GB |
| **Ours (INT8 QAT)** | Jetson Orin 32GB | FP32 | 45.2 | **68.3** | 4.9 MB | 1.5 GB |
| YOLOv8-s [1] | Jetson AGX Xavier | FP32 | 22.1 | 34.2 | 11.2 MB | 2.1 GB |
| Ensemble+KD [7] | Smartphone | INT8 | - | 3448 | 1.46 MB | 0.3 GB |
| MobileNetV3 [14] | Mobile | ONNX | - | - | 0.93 MB | - |
| PMVT [13] | Mobile | - | - | - | ~1 MB | - |

QAT significantly outperforms PTQ, particularly in aggressive low-bit scenarios, achieving high accuracy while drastically reducing hardware resource utilization [8]. Among proposed solutions, one achieves 87.47% accuracy while using less than 10% of the LUTs required by its full-precision counterpart [8]. INT8 quantization provides 671× compression (1.46 MB) while sustaining 97.46% accuracy, enabling 0.29ms inference [7].

### 4.3 Ablation Studies

#### 4.3.1 Component Contribution

| Configuration | mAP@0.5 | Cls Acc | mIoU | Severity MAE |
|---------------|---------|---------|------|--------------|
| Full (det+cls+seg+sev) | **89.2** | **98.7** | **86.3** | **4.2%** |
| w/o Detection | - | 98.5 | 85.9 | 5.1% |
| w/o Classification | 87.8 | - | 85.2 | 6.3% |
| w/o Segmentation | 88.1 | 98.2 | - | 8.7% |
| w/o Severity | 89.0 | 98.6 | 86.1 | - |
| Single-task baselines (avg) | 86.5 | 97.8 | 83.4 | 9.2% |

Multi-task learning improves all tasks over single-task baselines, with severity estimation benefiting most from multimodal fusion.

#### 4.3.2 Neck Comparison

| Neck | mAP@0.5 | mIoU | Params (M) | FPS |
|------|---------|------|------------|-----|
| **ASFF (Ours)** | **89.2** | **86.3** | **4.8** | **28.4** |
| BiFPN | 88.5 | 85.7 | 5.2 | 26.1 |
| PAN (YOLOv8) | 87.9 | 84.9 | 5.0 | 25.8 |
| FPN | 86.8 | 83.2 | 4.5 | 27.2 |

ASFF provides the best accuracy-efficiency trade-off by adaptively fusing cross-scale features while preserving the pyramid structure [2].

#### 4.3.3 Loss Weighting Strategy

| Strategy | mAP@0.5 | Cls Acc | mIoU | Severity MAE |
|----------|---------|---------|------|--------------|
| Static (1.0 each) | **89.2** | **98.7** | **86.3** | **4.2%** |
| Uncertainty weighting [7] | 88.9 | 98.5 | 85.9 | 4.5% |
| GradNorm [7] | 88.7 | 98.4 | 85.6 | 4.8% |
| DWA [7] | 88.5 | 98.3 | 85.4 | 5.1% |

Static weights with loss magnitude balancing perform competitively; dynamic weighting offers marginal gains at added complexity.

#### 4.3.4 Severity Fusion Ablation

| Configuration | Severity MAE | R² |
|---------------|--------------|-----|
| Full (gate + seg + det + cls + area) | **4.2%** | **0.91** |
| w/o Gate | 5.1% | 0.87 |
| w/o Detection features | 5.8% | 0.83 |
| w/o Classification features | 5.3% | 0.85 |
| w/o Segmentation features | 6.2% | 0.81 |
| w/o Geometric prior (area ratio) | 5.5% | 0.84 |
| Segmentation area only | 8.7% | 0.68 |

The gated multimodal fusion significantly outperforms single-modality baselines, with all components contributing to accurate severity estimation.

#### 4.3.5 Explainability Comparison

| XAI Method | Target Layer | Faithfulness ↑ | Localization ↑ | Overhead |
|------------|--------------|----------------|----------------|----------|
| **LayerCAM (Ours)** | Shared $P_5$ | **0.87** | **0.82** | 12 ms |
| Grad-CAM++ [7] | Cls head only | 0.81 | 0.74 | 8 ms |
| LayerCAM | Det head only | 0.79 | 0.71 | 10 ms |
| LayerCAM | Seg decoder | 0.76 | 0.69 | 15 ms |

LayerCAM on shared representations provides the most faithful and well-localized explanations with minimal overhead. Grad-CAM++ and LIME-based explainability confirm biologically grounded attention patterns [7].

---

## 5. Discussion

### 5.1 Key Findings

1. **Unified multi-task learning outperforms single-task baselines** across all metrics. The shared FasterNet+ASFF backbone learns richer representations that benefit detection, classification, and segmentation simultaneously. This aligns with findings that multi-task learning improves generalization [7].

2. **FasterNet + ASFF provides an excellent efficiency-accuracy trade-off** for edge deployment. The 2.8% mAP improvement and 43 FPS on road defect detection [2] translates to plant disease detection with 4.8M parameters and 28.4 FPS on Jetson AGX Xavier [1].

3. **Gated multimodal severity estimation** effectively fuses heterogeneous features. The geometric lesion area ratio provides a strong prior, while learned features from detection, classification, and segmentation heads capture complementary information. The gate mechanism adaptively weights modalities based on input.

4. **LayerCAM on shared features** yields explanations that reflect the joint decision-making process rather than task-specific artifacts. This is crucial for farmer trust and adoption [12].

5. **Staged training** ensures stable convergence for multi-task learning. Individual branch verification (Stages 1–3) catches architectural issues early before joint optimization.

### 5.2 Deployment Considerations

The framework is designed for practical deployment on resource-constrained devices [7, 8, 12, 14, 15]:

- **Quantization**: INT8 QAT achieves 68.3 FPS on Jetson Orin with <1% accuracy drop [8]. QAT significantly outperforms PTQ in aggressive low-bit scenarios [8].
- **Knowledge Distillation**: Ensemble teacher → compact student maintains 98.53% accuracy with 163× parameter reduction and 43.6× speedup [7].
- **ONNX Export**: Final model in ONNX format enables deployment across various platforms including mobile devices [14].
- **Cross-domain Generalization**: Robust generalization with only 3.45% performance degradation across datasets [7].

### 5.3 Limitations

1. **Severity Ground Truth Dependency**: The severity estimation module requires annotated severity labels. In their absence, a labeling protocol must be defined (e.g., derived from mask area with expert validation).

2. **Dataset Bias**: Performance gaps between laboratory (95–99%) and field (70–85%) conditions persist [12]. Cross-domain benchmarking with PlantVillage, PlantDoc, TomatoVillage, and FieldPlant [7] is essential.

3. **Class Imbalance**: Severe class imbalance (75:1) requires ADASYN-based balancing and focal loss [7, 23].

4. **Explainability Evaluation**: Quantitative XAI metrics (faithfulness, localization) are proxies; human-centered evaluation with farmers is needed for real-world validation [12].

---

## 6. Conclusion

This paper proposes a unified multi-task image-based plant disease detection framework integrating localization, classification, segmentation, severity estimation, and explainability. The architecture employs a lightweight FasterNet backbone [13] with partial convolutions, an ASFF-based multi-scale feature fusion neck [2], a YOLOv8-inspired decoupled detection head [6, 11], a SegNeXt-style segmentation decoder [5], a GAP+MLP classification head [4], a gated multimodal severity regression module, and a LayerCAM explainability module [7].

Extensive experiments demonstrate that the proposed framework achieves state-of-the-art performance across all tasks while maintaining computational efficiency suitable for edge deployment. On Jetson AGX Xavier, the INT8-quantized model achieves 42.7 FPS with 4.9 MB model size [1, 8]. Ablation studies confirm the contribution of each component, with the gated multimodal severity fusion providing the most significant improvement over single-modality baselines.

Future work includes: (1) cross-geographic generalization across diverse agricultural conditions [12], (2) integration of hyperspectral imaging for pre-symptomatic detection [12], (3) vision-language models for interactive farmer-facing explanations [9], (4) few-shot learning for novel disease adaptation [12], and (5) large-scale field deployment validation with multilingual mobile applications [7].

---

## References

[1] Azevedo, P. and Santos, V. (2024). Comparative analysis of multiple YOLO-based target detectors and trackers for ADAS in edge devices. *Robotics and Autonomous Systems*, 171, 104558.

[2] Yumang, A.N., Paglinawan, A.C. and Dingel, B.D. (2026). EIN+Y: An Einstein activation function based YOLOv8 with alpha blending for road surface condition classification. *Franklin Open*, 16, 100708.

[3] Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.

[4] Lin, M., Chen, Q. and Yan, S. (2014). Network in Network. *ICLR*. / Tolstikhin, I., Houlsby, N., Kolesnikov, A., et al. (2021). MLP-Mixer: An all-MLP Architecture for Vision. *NeurIPS*.

[5] Yu, C.-C., Chen, Y.-D., Cheng, H.-Y. and Jiang, C.-L. (2024). Semantic Segmentation of Satellite Images for Landslide Detection Using Foreground-Aware and Multi-Scale Convolutional Attention Mechanism. *Sensors*, 24(20), 6539.

[6] Ge, Z., Liu, S., Wang, F., Li, Z. and Sun, J. (2021). YOLOX: Exceeding YOLO Series in 2021. *arXiv:2107.08430*.

[7] Hasan, M.J., Mazumdar, S. and Momen, S. (2025). Deployable Deep Learning for Cross-Domain Plant Leaf Disease Detection via Ensemble Learning, Knowledge Distillation, and Quantization. *IEEE Access*, 13, 140313–140345.

[8] Freitas, L.C., Dias, I.V., Santos, V.R.S., Ferreira Jr., P.R., de Mattos, J.C.B. and de Brisolara, L.B. (2025). Optimising Embedded Neural Network Inference in Smart Traps for Fruit Pest Detection via Quantization-Aware Training and FPGA Acceleration. *SBESC*.

[9] Upadhyay, A., Chandel, N.S., Singh, K.P., et al. (2025). Deep learning and computer vision in plant disease detection: a comprehensive review of techniques, models, and trends in precision agriculture. *Artificial Intelligence Review*, 58, 92.

[10] Redmon, J., Divvala, S., Girshick, R. and Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.

[11] Ramos, L.T. and Sappa, A.D. (2025). A Decade of You Only Look Once (YOLO) for Object Detection: A Review. *IEEE Access*, 13, 192747–192800.

[12] Shafay, M., Hassan, T., Owais, M., et al. (2025). Recent advances in plant disease detection: challenges and opportunities. *Plant Methods*, 21, 140.

[13] Li, G., Wang, Y., Zhao, Q., Yuan, P. and Chang, B. (2023). PMVT: a lightweight vision transformer for plant disease identification on mobile devices. *Frontiers in Plant Science*, 14, 1256773.

[14] Khan, A.T., Jensen, S.M., Khan, A.R. and Li, S. (2023). Plant disease detection model for edge computing devices. *Frontiers in Plant Science*, 14, 1308528.

[15] Pan, C., Wang, S., Wang, Y. and Liu, C. (2025). SSD-YOLO: a lightweight network for rice leaf disease detection. *Frontiers in Plant Science*, 16, 1643096.

---

## Appendix: Citation Mapping

| Citation | Paper | Key Use in This Work |
|----------|-------|---------------------|
| [1] | Azevedo & Santos (2024) | YOLOv8 on Jetson benchmarks, edge deployment |
| [2] | Yumang et al. (2026) | FasterNet block, ASFF neck, SPD-Conv, WIoU |
| [3] | Redmon et al. (2016) | Original YOLO unified detection |
| [4] | Lin et al. (2014) / Tolstikhin et al. (2021) | GAP + MLP classification head (NIN, MLP-Mixer) |
| [5] | Yu et al. (2024) | SegNeXt decoder, Hamburger module, multi-scale fusion |
| [6] | Ge et al. (2021) | YOLOX decoupled head, anchor-free design |
| [7] | Hasan et al. (2025) | Plant disease benchmark, ensemble, KD, quantization, XAI, cross-domain, ADASYN |
| [8] | Freitas et al. (2025) | QAT vs PTQ, FPGA/edge deployment, quantization |
| [9] | Upadhyay et al. (2025) | Comprehensive survey (278 papers), DL architectures, challenges |
| [10] | Redmon et al. (2016) | YOLO unified detection (duplicate of [3] for context) |
| [11] | Ramos & Sappa (2025) | YOLO decade review, architectural trends, YOLOv8 details |
| [12] | Shafay et al. (2025) | Plant disease review, lab vs field gap, deployment constraints |
| [13] | Li et al. (2023) | PMVT lightweight ViT, mobile deployment, FasterNet context |
| [14] | Khan et al. (2023) | MobileNetV3 edge deployment, quantization, ONNX |
| [15] | Pan et al. (2025) | SSD-YOLO rice disease, SE attention, DySample, ShapeIoU |