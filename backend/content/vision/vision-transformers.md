---
title: "Vision Transformers (ViT)"
slug: vision-transformers
summary: "How Transformers conquered computer vision — from ViT's patch embeddings to DeiT's distillation and Swin Transformer's shifted windows."
tags: ["ViT", "vision-transformer", "self-attention", "DeiT", "Swin-Transformer", "computer-vision", "deep-learning"]
visibility: public
level: intermediate
---

# Vision Transformers (ViT)

## From NLP to Vision

The Transformer architecture — originally designed for language — has fundamentally changed computer vision. The key question was: **can self-attention replace convolutions?**

| Aspect | CNNs | Vision Transformers |
|--------|------|-------------------|
| Inductive bias | Local connectivity, translation equivariance | Minimal (learns from data) |
| Receptive field | Grows with depth | Global from the first layer |
| Data efficiency | Better with small datasets | Needs large datasets (or distillation) |
| Scalability | Diminishing returns at scale | Continues improving with more data/compute |

---

## Vision Transformer (ViT, 2021)

**Paper:** Dosovitskiy et al., "An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale"

### Core Idea

Treat an image as a sequence of patches — just like tokens in NLP.

### Architecture

```
Input Image (224×224×3)
    ↓
Split into patches (16×16) → 196 patches
    ↓
Flatten each patch → 16×16×3 = 768-d vector
    ↓
Linear projection → Patch embeddings (196 × D)
    ↓
Prepend [CLS] token → (197 × D)
    ↓
Add positional embeddings → (197 × D)
    ↓
Transformer Encoder (L layers)
    ↓
[CLS] token output → MLP head → Classification
```

### Patch Embedding

The image $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$ is divided into $N = \frac{HW}{P^2}$ patches, where $P$ is the patch size:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \ldots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

Where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding projection and $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are learned positional embeddings.

### Self-Attention in Vision

Each Transformer layer applies multi-head self-attention (MSA) and MLP:

$$\mathbf{z}'_l = \text{MSA}(\text{LN}(\mathbf{z}_{l-1})) + \mathbf{z}_{l-1}$$
$$\mathbf{z}_l = \text{MLP}(\text{LN}(\mathbf{z}'_l)) + \mathbf{z}'_l$$

Self-attention computes relationships between all pairs of patches:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Cost:** Self-attention has $O(N^2)$ complexity — quadratic in the number of patches. For a 224×224 image with 16×16 patches: $N = 196$, which is manageable. Higher resolution or smaller patches dramatically increase cost.

### ViT Models

| Model | Layers | Hidden Dim | Heads | Params | ImageNet Top-1 |
|-------|--------|-----------|-------|--------|---------------|
| ViT-B/16 | 12 | 768 | 12 | 86M | 77.9% (ImageNet-1k) |
| ViT-L/16 | 24 | 1024 | 16 | 307M | 76.5% (ImageNet-1k) |
| ViT-H/14 | 32 | 1280 | 16 | 632M | 88.6% (JFT pre-train) |

**Critical finding:** ViT underperforms CNNs when trained on ImageNet-1k alone (1.3M images), but excels when pre-trained on large datasets (JFT-300M, 300M images). Transformers lack the inductive biases of CNNs, so they need more data to learn spatial structure.

---

## DeiT: Data-Efficient Image Transformers (2021)

**Paper:** Touvron et al., "Training data-efficient image transformers & distillation through attention"

DeiT showed that ViT can match CNNs on ImageNet-1k **without** massive pre-training datasets.

### Key Innovations

1. **Knowledge distillation token:** A special token (like [CLS]) that learns from a CNN teacher
2. **Strong data augmentation:** RandAugment, Mixup, CutMix, Random Erasing
3. **Regularization:** Stochastic depth, repeated augmentation

### Distillation Architecture

```
Input patches + [CLS] token + [DIST] token
    ↓
Transformer Encoder
    ↓
[CLS] → Classification loss (with true labels)
[DIST] → Distillation loss (with teacher predictions)
```

The distillation loss uses hard labels from the teacher:

$$\mathcal{L} = \frac{1}{2} \mathcal{L}_{\text{CE}}(y, \psi(\mathbf{z}_{\text{cls}})) + \frac{1}{2} \mathcal{L}_{\text{CE}}(y_t, \psi(\mathbf{z}_{\text{dist}}))$$

Where $y_t = \text{argmax}(Z_t)$ is the hard decision of the teacher (a RegNet CNN).

### Results

| Model | Params | ImageNet Top-1 | Pre-training Data |
|-------|--------|---------------|-------------------|
| ViT-B/16 | 86M | 77.9% | ImageNet-1k only |
| DeiT-B | 86M | 81.8% | ImageNet-1k only |
| DeiT-B (distilled) | 87M | 83.4% | ImageNet-1k only |

DeiT closed the data gap — making ViT practical without hundreds of millions of training images.

---

## Swin Transformer (2021)

**Paper:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"

Addressed ViT's two main limitations: **quadratic complexity** and **single-scale features**.

### Hierarchical Architecture

Unlike ViT (which maintains a single resolution), Swin Transformer produces multi-scale feature maps — like a CNN:

```
Stage 1: 56×56, C=96     (4×4 patch merging from 224×224)
    ↓ Patch merging (2× downsample)
Stage 2: 28×28, C=192
    ↓ Patch merging
Stage 3: 14×14, C=384
    ↓ Patch merging
Stage 4: 7×7, C=768
```

This hierarchy makes Swin Transformer a drop-in replacement for CNN backbones in detection and segmentation.

### Window-Based Self-Attention

Instead of global self-attention ($O(N^2)$), Swin computes attention within **local windows** of $M \times M$ patches:

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2 hwC$$

vs. global:

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2 C$$

With $M = 7$ and $hw = 56 \times 56 = 3136$, window attention is dramatically cheaper.

### Shifted Window Mechanism

**Problem:** Window attention limits cross-window communication.

**Solution:** Alternate between regular and shifted window partitions:

```
Layer l:   Regular windows        Layer l+1: Shifted windows
┌───┬───┐                        ┌──┬────┬──┐
│ W1│ W2│                        │  │    │  │
├───┼───┤         →              ├──┼────┼──┤
│ W3│ W4│                        │  │    │  │
└───┴───┘                        └──┴────┴──┘
```

Shifting by $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ pixels creates new windows that overlap boundaries of the previous layer's windows, enabling information flow across the entire feature map.

### Relative Position Bias

Instead of absolute positional embeddings, Swin uses a relative position bias $B$ in the attention computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + B\right) V$$

Where $B \in \mathbb{R}^{M^2 \times M^2}$ is a learnable bias indexed by relative position. This generalizes better to different image sizes.

### Swin Transformer Models

| Model | Params | ImageNet Top-1 | FLOPs |
|-------|--------|---------------|-------|
| Swin-T | 29M | 81.3% | 4.5G |
| Swin-S | 50M | 83.0% | 8.7G |
| Swin-B | 88M | 83.5% | 15.4G |
| Swin-L | 197M | 86.3% | 34.5G |

### Impact on Downstream Tasks

Swin Transformer became the dominant backbone for detection and segmentation:

| Task | Model | AP/mIoU |
|------|-------|---------|
| Object Detection (COCO) | Swin-L + Cascade Mask R-CNN | 58.7 AP |
| Semantic Segmentation (ADE20K) | Swin-L + UPerNet | 53.5 mIoU |
| Instance Segmentation (COCO) | Swin-L + Cascade Mask R-CNN | 51.1 AP |

---

## ViT vs CNN: When to Use What

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Small dataset (<10K images) | CNN (ResNet, EfficientNet) | Better inductive bias |
| Medium dataset (10K–1M) | DeiT or Swin Transformer | Distillation/efficient attention |
| Large dataset (>1M) | ViT or Swin | Scales better with data |
| Dense prediction (detection, segmentation) | Swin Transformer | Hierarchical multi-scale features |
| Classification only | ViT or DeiT | Simpler, effective |
| Edge deployment | EfficientNet or MobileViT | Parameter efficient |

---

## Beyond Swin: Recent Developments

| Model | Year | Key Innovation |
|-------|------|---------------|
| BEiT | 2022 | BERT-style pre-training for ViT (masked image modeling) |
| MAE | 2022 | Masked autoencoders — mask 75% of patches, reconstruct |
| DINOv2 | 2023 | Self-supervised ViT features rivaling supervised |
| EVA-02 | 2023 | Scaling ViT to 4.4B params with masked image modeling |
| SigLIP | 2023 | Sigmoid loss for vision-language pre-training |

The trend is clear: **self-supervised pre-training + Transformer architectures** is the dominant paradigm in modern computer vision.

---

## Key Takeaways

1. **ViT** proved that pure Transformers can match CNNs in vision — treating images as sequences of patches
2. **Self-attention** provides global receptive fields from layer 1, but at $O(N^2)$ cost
3. **DeiT** made ViT practical on ImageNet-scale data through distillation and strong augmentation
4. **Swin Transformer** solved the efficiency and multi-scale problems with windowed attention and hierarchical features
5. **Shifted windows** enable cross-window communication while maintaining linear complexity
6. **Hierarchical ViTs** (Swin) serve as drop-in CNN backbone replacements for detection and segmentation
7. The field is moving toward **self-supervised ViT pre-training** (MAE, DINOv2) as the foundation for all vision tasks

## References

- Dosovitskiy, A., et al. (2021). An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Touvron, H., et al. (2021). Training data-efficient image transformers & distillation through attention. *ICML 2021*. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Liu, Z., et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *TMLR 2024*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
