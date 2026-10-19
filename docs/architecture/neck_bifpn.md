# CropGuard Architecture: BiFPN-lite Neck

## 📌 Overview
The **BiFPN-lite** (Bidirectional Feature Pyramid Network) acts as the "information highway" of the model. It fuses the multi-scale feature maps (C3, C4, C5) from the backbone to ensure that the heads receive a rich, balanced set of features.

## 🛠 Technical Specifications
- **Input:** Feature maps from FasterNet-T0 with channels `[80, 160, 320]`.
- **Uniform Width:** All features are projected to a uniform width (default: 96) using $1 \times 1$ convolutions to standardize fusion.
- **Fusion Logic:** 
    - **Top-Down Path:** High-level semantic information is downsampled and fused with lower-level maps.
    - **Bottom-Up Path:** Fine-grained spatial information is upsampled and fused with higher-level maps.
- **Learnable Weights:** Unlike standard FPNs, BiFPN-lite uses **softmax-normalized learnable weights**. This allows the model to decide *which scale* is more important for a specific image.

## 🧠 Why BiFPN-lite?
Standard feature pyramids treat all scales equally. However, some tomato diseases are identified by tiny spots (requiring C3) and others by overall leaf discoloration (requiring C5). BiFPN-lite enables:
1. **Cross-Scale Synergy:** It allows the model to "cross-reference" textures and semantics.
2. **Parameter Efficiency:** The "lite" version uses depthwise-separable convolutions to keep the parameter count minimal.
3. **Dynamic Weighting:** The learnable fusion weights allow the network to adapt to different disease types automatically.

## 📂 File Reference
`CropGuard_Final/model/architecture/neck.py`
