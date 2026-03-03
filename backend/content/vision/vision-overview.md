---
title: "Computer Vision: Overview"
slug: vision-overview
summary: "From CNNs to Vision Transformers — key architectures, tasks, and techniques in modern computer vision."
tags: ["CNN", "ViT", "object-detection", "segmentation", "computer-vision", "ResNet", "YOLO"]
visibility: public
---

# Computer Vision: Overview

## What is Computer Vision?

**Computer Vision (CV)** gives machines the ability to interpret and understand visual data — images, video, and 3D scenes. It powers autonomous vehicles, medical imaging, face recognition, content moderation, and AR/VR.

**Core tasks:**

| Task | Description | Example Output |
|------|-------------|---------------|
| Image Classification | Classify entire image | "cat" |
| Object Detection | Localize + classify objects | Bounding boxes + labels |
| Semantic Segmentation | Label every pixel | Per-pixel class map |
| Instance Segmentation | Detect + segment each object | Per-instance masks |
| Depth Estimation | Estimate 3D depth | Depth map |
| Image Generation | Synthesize images | New images |
| Optical Flow | Estimate motion between frames | Flow vectors |

---

## Convolutional Neural Networks (CNNs)

### Core Operation: Convolution

For input feature map $X$ and filter $K$:

$$Y[i,j] = \sum_{m,n} X[i+m, j+n] \cdot K[m,n]$$

Key properties:
- **Local receptive field:** Each output depends on a small input region
- **Weight sharing:** Same filter applied across all spatial positions
- **Translation equivariance:** Features detected regardless of position

### Key Architectural Components

**Pooling:** Reduce spatial dimensions:
- Max pooling: $y = \max(x_1, x_2, x_3, x_4)$ — dominant feature
- Average pooling: $y = \text{mean}(x_1, x_2, x_3, x_4)$

**Batch Normalization:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Stabilizes training, allows higher learning rates.

**Skip Connections (ResNet):**
$$F(x) + x \quad \text{(residual connection)}$$

Enables training of very deep networks (100+ layers) by allowing gradient flow.

### CNN Architecture Evolution

| Model | Year | Params | Top-1 (ImageNet) | Innovation |
|-------|------|--------|-----------------|-----------|
| LeNet-5 | 1998 | 60K | — | First practical CNN |
| AlexNet | 2012 | 60M | 63.3% | Deep CNN, GPU training, ReLU |
| VGG-16 | 2014 | 138M | 74.4% | Deep with 3×3 convs |
| GoogLeNet | 2014 | 7M | 74.8% | Inception modules |
| ResNet-50 | 2015 | 25M | 76.0% | Residual connections |
| ResNet-152 | 2015 | 60M | 77.8% | Deeper residual |
| SENet | 2017 | 28M | 82.7% | Squeeze-and-excitation |
| EfficientNet-B7 | 2019 | 66M | 84.4% | Compound scaling |
| ConvNeXt | 2022 | 88M | 85.8% | Modernized CNN |

---

## Vision Transformer (ViT)

**Paper:** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

### Architecture

**Patch Embedding:**
1. Split image into $N = \frac{H \times W}{P^2}$ patches of size $P \times P$ (typically $P=16$)
2. Flatten each patch: $\mathbb{R}^{P \times P \times C} \to \mathbb{R}^{P^2 C}$
3. Linear projection to embedding dim $D$: $\mathbf{x}_p = W_E \cdot x_{\text{patch}}$

**Input sequence:**
$$[\mathbf{x}_{\text{cls}}; \mathbf{x}_1 + \mathbf{e}_1; \mathbf{x}_2 + \mathbf{e}_2; \ldots; \mathbf{x}_N + \mathbf{e}_N]$$

Where $\mathbf{x}_{\text{cls}}$ is a learnable classification token and $\mathbf{e}_i$ are position embeddings.

**Then:** Standard Transformer encoder (Multi-head self-attention + FFN + LayerNorm).

**Classification:** Use $\mathbf{x}_{\text{cls}}$ output → linear head.

### ViT Variants

| Model | Layers | Hidden | Heads | Params |
|-------|--------|--------|-------|--------|
| ViT-S/16 | 12 | 384 | 6 | 22M |
| ViT-B/16 | 12 | 768 | 12 | 86M |
| ViT-L/16 | 24 | 1024 | 16 | 307M |
| ViT-H/14 | 32 | 1280 | 16 | 632M |

**ViT vs CNN:**
| Aspect | CNN | ViT |
|--------|-----|-----|
| Inductive bias | Strong (locality, translation equiv.) | Weak |
| Data efficiency | Good (small data) | Poor (needs pre-training) |
| Scalability | Limited | Excellent |
| Global context | Limited (deep layers only) | All layers |
| State-of-art | ConvNeXt competitive | ViT + DeiT dominant |

---

## Object Detection

### Two-Stage Detectors (Slow, Accurate)

**Faster R-CNN:**
1. **Backbone** (ResNet): Extract feature map
2. **Region Proposal Network (RPN)**: Propose ~2000 candidate boxes
3. **RoI Align**: Crop and resize features for each proposal
4. **Detection head**: Classify + refine each proposal

**Anchor boxes:** Predefined aspect ratios/scales at each feature map position.

### One-Stage Detectors (Fast)

**YOLO (You Only Look Once):**
- Divide image into $S \times S$ grid
- Each cell predicts $B$ bounding boxes + confidence + class
- Output: $S \times S \times (B \times 5 + C)$ tensor

**YOLO Loss:**
$$\mathcal{L} = \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{obj}} + \mathcal{L}_{\text{cls}}$$

**YOLOv8 (Ultralytics, 2023):** Anchor-free, best speed-accuracy tradeoff for real-time.

**DETR (DEtection TRansformer, 2020):**
- Transformer + Hungarian matching — no NMS, no anchors
- Elegant but slow to train

### Detection Metrics

**IoU (Intersection over Union):**
$$\text{IoU} = \frac{|\text{box}_1 \cap \text{box}_2|}{|\text{box}_1 \cup \text{box}_2|}$$

**mAP (mean Average Precision):** Average AP across IoU thresholds and classes.

---

## Segmentation

### Semantic Segmentation

Assigns a class to **every pixel** (no instances — all cars are "car").

**FCN (Fully Convolutional Network):** Replace FC layers with convolutions; use upsampling.

**UNet:** Encoder-decoder with skip connections — dominant in medical imaging.

### Instance Segmentation

Detects each object AND segments it individually.

**Mask R-CNN (He et al., 2017):**
- Extends Faster R-CNN with a mask head
- For each detected object: predict a binary segmentation mask

**SAM (Segment Anything Model, Meta 2023):**
- Promptable segmentation: click, box, or text → high-quality mask
- ViT-H backbone + mask decoder
- Zero-shot generalization across any object

---

## Self-Supervised Vision Pre-training

### MAE (Masked Autoencoders, 2021)

Mask 75% of image patches, reconstruct from 25%:
- Highly efficient: only process visible patches in encoder
- Strong representations for fine-tuning
- Similar to BERT's MLM for vision

### DINO / DINOv2 (Self-Distillation)

Train ViT without labels using self-distillation:
- Student processes masked/cropped views
- Teacher processes global view (teacher = EMA of student)
- Features emerge that support segmentation, depth, etc.

---

## Key Takeaways

1. **CNNs** dominated vision from 2012-2020 with inductive biases (locality, translation equivariance)
2. **ViT** shows pure transformers match/exceed CNNs at scale with sufficient data
3. **ResNet skip connections** solved vanishing gradients — enabled very deep networks
4. **YOLO** is the practical choice for real-time detection; DETR for end-to-end elegance
5. **SAM** achieves promptable zero-shot segmentation — a new paradigm
6. **Self-supervised pretraining** (MAE, DINO) produces strong visual features without labels

## References

- Krizhevsky et al. (2012) — AlexNet
- He et al. (2015) — Deep Residual Learning (ResNet)
- Dosovitskiy et al. (2020) — ViT: An Image is Worth 16×16 Words
- Carion et al. (2020) — DETR
- He et al. (2022) — Masked Autoencoders Are Scalable Vision Learners (MAE)
- Kirillov et al. (2023) — Segment Anything (SAM)
