# CropGuard Architecture: FasterNet-T0 Backbone

## 📌 Overview
The backbone is the foundation of the CropGuard system, responsible for extracting multi-scale feature representations from the input images. To maintain a lightweight profile suitable for edge deployment, we utilize **FasterNet-T0**.

## 🛠 Technical Specifications
- **Model:** `fasternet_t0` (via `timm` library).
- **Configuration:** `features_only=True`.
- **Output Indices:** `(1, 2, 3)`.
- **Feature Maps:**
    - **C3:** Stride 8 $\rightarrow$ 80 channels.
    - **C4:** Stride 16 $\rightarrow$ 160 channels.
    - **C5:** Stride 32 $\rightarrow$ 320 channels.

## 🧠 Why FasterNet-T0?
FasterNet is designed to reduce the redundancy of convolutions while maintaining high throughput. In the context of crop disease detection:
1. **Efficiency:** It provides a significant parameter reduction compared to standard ResNet or CSP-DarkNet backbones.
2. **Multi-Scale Support:** By extracting C3, C4, and C5, the model captures both fine-grained lesion textures (C3) and global leaf context (C5).
3. **Latency:** Optimized for faster inference on server-side CPUs and potential mobile integration.

## 📂 File Reference
`CropGuard_Final/model/architecture/backbone.py`
