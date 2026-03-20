---
title: "Semantic & Instance Segmentation"
slug: semantic-segmentation
summary: "From pixel-level classification to panoptic segmentation — covering FCN, U-Net, DeepLab, Mask R-CNN, and modern unified approaches."
tags: ["segmentation", "FCN", "U-Net", "DeepLab", "Mask-R-CNN", "panoptic", "computer-vision", "deep-learning"]
visibility: public
level: intermediate
---

# Semantic & Instance Segmentation

## Segmentation Tasks

Segmentation assigns labels at a finer granularity than bounding boxes. There are three main variants:

| Task | What It Does | Output | Example |
|------|-------------|--------|---------|
| Semantic Segmentation | Label every pixel with a class | $H \times W$ class map | "All road pixels are class 1" |
| Instance Segmentation | Separate individual object instances | Per-instance masks | "Car A vs Car B" |
| Panoptic Segmentation | Combine both: stuff (sky, road) + things (cars, people) | Unified pixel map | Complete scene understanding |

**Formally:** Given an image $I \in \mathbb{R}^{H \times W \times 3}$, semantic segmentation produces a label map $L \in \{1, \ldots, C\}^{H \times W}$ where $C$ is the number of classes.

---

## Fully Convolutional Networks (FCN, 2015)

**Paper:** Long, Shelhamer, & Darrell, "Fully Convolutional Networks for Semantic Segmentation"

The first deep learning approach to semantic segmentation. Key insight: replace all fully connected layers with convolutional layers to produce spatial output maps.

### Architecture

```
Input Image (H×W×3)
    ↓
Classification CNN (e.g., VGG-16) with FC layers → Conv layers
    ↓
Coarse prediction map (H/32 × W/32 × C)
    ↓
Learned upsampling (transposed convolutions)
    ↓
Dense prediction (H × W × C)
```

### Skip Connections for Multi-Scale

FCN combines predictions from multiple stages to recover spatial detail:

| Variant | Combines | Stride | Result |
|---------|----------|--------|--------|
| FCN-32s | Pool5 only | 32× upsample | Coarse |
| FCN-16s | Pool5 + Pool4 | 16× upsample | Better |
| FCN-8s | Pool5 + Pool4 + Pool3 | 8× upsample | Best |

### Transposed Convolution

Also called deconvolution or fractionally-strided convolution. Upsamples feature maps by inserting zeros between input values and applying a learned convolution:

$$O = (I - 1) \times S - 2P + K$$

Where $I$ = input size, $S$ = stride, $P$ = padding, $K$ = kernel size.

**Limitation:** FCN produces relatively coarse boundaries due to repeated downsampling.

---

## U-Net (2015)

**Paper:** Ronneberger, Fischer, & Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation"

Designed for biomedical segmentation with limited training data. The symmetric encoder-decoder architecture with skip connections became one of the most influential segmentation designs.

### Architecture

```
Encoder (contracting)         Decoder (expanding)
    Input (572×572)
    ↓ Conv 3×3 × 2 (64)  ────────→  Concat + Conv × 2 (64) → Output
    ↓ MaxPool 2×2
    ↓ Conv 3×3 × 2 (128) ────────→  Concat + Conv × 2 (128)
    ↓ MaxPool 2×2
    ↓ Conv 3×3 × 2 (256) ────────→  Concat + Conv × 2 (256)
    ↓ MaxPool 2×2
    ↓ Conv 3×3 × 2 (512) ────────→  Concat + Conv × 2 (512)
    ↓ MaxPool 2×2
    └→ Conv 3×3 × 2 (1024) → Bottleneck
```

### Key Design Principles

1. **Symmetric encoder-decoder:** Encoder captures context, decoder enables precise localization
2. **Skip connections via concatenation:** Encoder features concatenated (not added) with decoder features — preserves full spatial detail
3. **No fully connected layers:** Fully convolutional, works on any input size
4. **Overlap-tile strategy:** Enables seamless segmentation of arbitrarily large images

### Why U-Net Works with Limited Data

- Skip connections preserve low-level features, reducing the amount of information the decoder must learn
- Heavy data augmentation (elastic deformations, rotations)
- Weighted cross-entropy loss that emphasizes boundaries between touching objects:

$$w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \exp\left(-\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2}\right)$$

Where $d_1$ and $d_2$ are distances to the two nearest cell boundaries.

**Impact:** U-Net remains the dominant architecture for medical image segmentation and has inspired countless variants (Attention U-Net, U-Net++, nnU-Net).

---

## DeepLab Family (2015–2021)

### The Receptive Field Problem

Standard CNNs with pooling/striding lose spatial resolution. To segment accurately, we need both:
- **Large receptive fields** for global context
- **High resolution** for precise boundaries

### Atrous (Dilated) Convolution

DeepLab's core innovation. A standard convolution with gaps (holes) between filter weights:

$$y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]$$

Where $r$ is the dilation rate. This expands the receptive field without increasing parameters or reducing resolution.

| Dilation Rate | Effective Kernel Size | Receptive Field |
|--------------|----------------------|-----------------|
| $r = 1$ | 3×3 (standard) | 3×3 |
| $r = 2$ | 3×3 with gaps | 5×5 |
| $r = 4$ | 3×3 with gaps | 9×9 |

### Atrous Spatial Pyramid Pooling (ASPP)

Apply atrous convolutions at multiple rates in parallel, then combine:

```
Feature Map
    ├── 1×1 Conv (rate=1)
    ├── 3×3 Atrous Conv (rate=6)
    ├── 3×3 Atrous Conv (rate=12)
    ├── 3×3 Atrous Conv (rate=18)
    └── Global Average Pooling → 1×1 Conv → Upsample
         ↓
    Concatenate → 1×1 Conv → Predictions
```

This captures objects at multiple scales without multiple forward passes.

### DeepLab Evolution

| Version | Year | Key Innovation | mIoU (VOC 2012) |
|---------|------|---------------|-----------------|
| DeepLabv1 | 2015 | Atrous convolution + CRF post-processing | 71.6% |
| DeepLabv2 | 2017 | ASPP multi-scale module | 79.7% |
| DeepLabv3 | 2017 | Improved ASPP + batch norm | 85.7% |
| DeepLabv3+ | 2018 | Encoder-decoder with atrous separable conv | 89.0% |

### Conditional Random Fields (CRF)

Early DeepLab versions used CRF as post-processing to sharpen boundaries. The CRF energy function:

$$E(\mathbf{x}) = \sum_i \theta_i(x_i) + \sum_{i<j} \theta_{ij}(x_i, x_j)$$

Where the pairwise term encourages nearby pixels with similar colors to share labels. DeepLabv3+ largely eliminated the need for CRF through better architecture.

---

## Mask R-CNN (2017)

**Paper:** He, Gkioxari, Dollár, & Girshick, "Mask R-CNN"

Extended Faster R-CNN with a parallel mask prediction branch for **instance segmentation**.

### Architecture

```
Input Image
    ↓
Backbone (ResNet-50-FPN)
    ↓
Region Proposal Network → Proposals
    ↓
RoI Align → Fixed-size features (14×14)
    ↓
Three parallel heads:
    ├── Classification → Class label
    ├── Box Regression → Bounding box
    └── Mask Branch → 28×28 binary mask per class
```

### Key Design Choices

1. **RoI Align (not RoI Pool):** Bilinear interpolation instead of quantization — critical for pixel-level accuracy
2. **Per-class binary masks:** Predict a separate $m \times m$ mask for each class — decouples classification from segmentation
3. **Multi-task loss:**

$$\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$$

Where $\mathcal{L}_{\text{mask}}$ is per-pixel binary cross-entropy, applied only for the ground-truth class.

### Performance

| Metric | Value |
|--------|-------|
| AP (instance seg, COCO) | 37.1 |
| AP (object detection, COCO) | 39.8 |
| Speed | ~5 FPS (ResNet-50-FPN) |

**Extensions:** Mask R-CNN also supports keypoint detection by replacing the mask branch with a keypoint heatmap head.

---

## Panoptic Segmentation

**Paper:** Kirillov et al. (2019), "Panoptic Segmentation"

Unifies semantic and instance segmentation into a single task. Every pixel gets both a class label and an instance ID.

### Stuff vs Things

| Category | Examples | Approach |
|----------|----------|----------|
| **Stuff** (uncountable) | Sky, road, grass, water | Semantic segmentation |
| **Things** (countable) | Cars, people, animals | Instance segmentation |

### Panoptic Quality (PQ)

The standard metric for panoptic segmentation:

$$\text{PQ} = \underbrace{\frac{\sum_{(p,g) \in TP} \text{IoU}(p,g)}{|TP|}}_{\text{Segmentation Quality (SQ)}} \times \underbrace{\frac{|TP|}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}}_{\text{Recognition Quality (RQ)}}$$

PQ decomposes into segmentation quality (how well matched segments align) and recognition quality (how well segments are detected).

### Modern Approaches

| Method | Year | Architecture | PQ (COCO) |
|--------|------|-------------|-----------|
| Panoptic FPN | 2019 | FPN + semantic head + Mask R-CNN | 40.9 |
| MaskFormer | 2021 | Transformer decoder + mask classification | 46.5 |
| Mask2Former | 2022 | Masked attention + multi-scale | 57.8 |
| OneFormer | 2023 | Task-conditioned joint training | 58.0 |

**Mask2Former** unified all three segmentation tasks (semantic, instance, panoptic) with a single architecture using masked cross-attention in a Transformer decoder.

---

## Loss Functions for Segmentation

| Loss | Formula | Best For |
|------|---------|----------|
| Cross-Entropy | $-\sum_c y_c \log(\hat{y}_c)$ | Balanced classes |
| Weighted CE | $-\sum_c w_c y_c \log(\hat{y}_c)$ | Class imbalance |
| Dice Loss | $1 - \frac{2|P \cap G|}{|P| + |G|}$ | Small objects, medical |
| Focal Loss | $-\alpha(1-\hat{y})^\gamma \log(\hat{y})$ | Extreme class imbalance |
| Lovász-Softmax | Lovász extension of IoU | Directly optimizes IoU |

In practice, combining Dice loss with cross-entropy often works best:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \mathcal{L}_{\text{Dice}}$$

---

## Key Takeaways

1. **FCN** introduced end-to-end trainable dense prediction by replacing FC layers with convolutions
2. **U-Net's** symmetric encoder-decoder with skip connections remains the gold standard for medical segmentation
3. **Atrous convolutions** expand receptive fields without losing resolution — central to the DeepLab family
4. **ASPP** captures multi-scale context by applying parallel atrous convolutions at different rates
5. **Mask R-CNN** extends detection to instance segmentation with a simple mask head and RoI Align
6. **Panoptic segmentation** unifies stuff (semantic) and things (instance) into a single coherent output
7. **Mask2Former** represents the state of the art — a single Transformer architecture handles all segmentation tasks

## References

- Long, J., Shelhamer, E., & Darrell, T. (2015). Fully Convolutional Networks for Semantic Segmentation. *CVPR 2015*. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
- Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *TPAMI*. [arXiv:1606.00915](https://arxiv.org/abs/1606.00915)
- Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. *ECCV 2018*. [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV 2017*. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic Segmentation. *CVPR 2019*. [arXiv:1801.00868](https://arxiv.org/abs/1801.00868)
- Cheng, B., Misra, I., Schwing, A. G., Kirillov, A., & Girshick, R. (2022). Masked-attention Mask Transformer for Universal Image Segmentation. *CVPR 2022*. [arXiv:2112.01527](https://arxiv.org/abs/2112.01527)
