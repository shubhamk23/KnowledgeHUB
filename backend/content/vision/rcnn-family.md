---
title: "R-CNN Family: Region-Based Detection"
slug: rcnn-family
summary: "Deep dive into the R-CNN family — from R-CNN to Faster R-CNN — covering region proposals, RoI pooling, and the Region Proposal Network."
tags: ["R-CNN", "Faster-R-CNN", "object-detection", "RPN", "RoI-pooling", "computer-vision", "deep-learning"]
visibility: public
level: intermediate
---

# R-CNN Family: Region-Based Detection

## Overview

The R-CNN family pioneered deep learning-based object detection through a two-stage approach: first propose candidate regions, then classify each region. Each generation dramatically improved speed while maintaining accuracy.

| Model | Year | Test Speed | mAP (VOC 2007) | Key Innovation |
|-------|------|-----------|-----------------|----------------|
| R-CNN | 2014 | 47s/image | 58.5% | CNN features for detection |
| Fast R-CNN | 2015 | 0.32s/image | 66.9% | Shared feature computation |
| Faster R-CNN | 2015 | 0.2s/image | 69.9% | Region Proposal Network |

---

## R-CNN (2014)

**Paper:** Girshick et al., "Rich feature hierarchies for accurate object detection and semantic segmentation"

### Pipeline

```
Input Image
    ↓
Selective Search → ~2000 region proposals
    ↓
For EACH proposal:
    Warp to 227×227 → CNN (AlexNet) → Feature vector (4096-d)
    ↓
    SVM classifier → Class label
    ↓
    Bounding box regressor → Refined coordinates
```

### Selective Search

A classical (non-learned) algorithm for generating region proposals:
1. Start with pixel-level superpixels
2. Iteratively merge similar regions based on color, texture, size, and fill
3. Output: ~2000 candidate bounding boxes per image

### Training Pipeline

Three separate stages:
1. **Fine-tune CNN** on detection data (ImageNet pre-trained AlexNet)
2. **Train SVMs** — one per class — on CNN features
3. **Train bounding box regressors** — linear regression on CNN features

### Bounding Box Regression

Learn to refine proposal coordinates toward ground truth:

$$\hat{G}_x = P_w \cdot d_x(P) + P_x$$
$$\hat{G}_y = P_h \cdot d_y(P) + P_y$$
$$\hat{G}_w = P_w \cdot \exp(d_w(P))$$
$$\hat{G}_h = P_h \cdot \exp(d_h(P))$$

Where $P$ is the proposal and $d_*(P)$ are learned transformations.

### Limitations
- **Very slow:** CNN runs 2000 times per image (once per proposal)
- **Multi-stage training:** CNN, SVM, and regressor trained separately
- **Storage intensive:** Features cached to disk for SVM training

---

## Fast R-CNN (2015)

**Paper:** Girshick, "Fast R-CNN"

**Key insight:** Share CNN computation across all proposals by computing features once for the entire image.

### Pipeline

```
Input Image
    ↓
CNN Backbone → Feature Map (shared for all proposals)
    ↓
Selective Search → ~2000 region proposals
    ↓
For EACH proposal:
    RoI Pooling → Fixed-size feature (7×7)
    ↓
    FC layers → Two heads:
                  ├── Softmax classifier → Class
                  └── Bounding box regressor → Refined box
```

### RoI Pooling

**Problem:** Proposals have different sizes, but FC layers need fixed-size input.

**Solution:** Divide each proposal's feature map region into a fixed grid (e.g., 7×7) and max-pool each cell:

1. Project proposal coordinates onto the feature map
2. Divide the projected region into $H \times W$ bins (e.g., 7×7 = 49 bins)
3. Apply max pooling within each bin
4. Output: Fixed $H \times W \times C$ tensor

**Quantization issue:** The projection involves rounding, which introduces spatial misalignment (especially for small objects). Addressed by RoI Align in Mask R-CNN.

### Multi-Task Loss

Fast R-CNN trains classification and regression jointly:

$$\mathcal{L} = \mathcal{L}_{\text{cls}}(p, u) + \lambda [u \geq 1] \cdot \mathcal{L}_{\text{loc}}(t^u, v)$$

Where:
- $\mathcal{L}_{\text{cls}}$ = cross-entropy loss for classification
- $\mathcal{L}_{\text{loc}}$ = smooth L1 loss for box regression
- $[u \geq 1]$ = only regress for non-background classes
- $\lambda$ = balance weight (typically 1.0)

**Smooth L1 Loss:**

$$\text{smooth}_{L_1}(x) = \begin{cases} 0.5x^2 & \text{if } |x| < 1 \\ |x| - 0.5 & \text{otherwise} \end{cases}$$

Less sensitive to outliers than L2 loss, more stable training than L1 loss.

### Improvements over R-CNN

| Aspect | R-CNN | Fast R-CNN |
|--------|-------|------------|
| Feature computation | Per-proposal (2000×) | Once per image |
| Training | 3 separate stages | Single-stage, end-to-end |
| Classifier | SVMs | Softmax (within network) |
| Speed (test) | 47s/image | 0.32s/image |
| Speed (training) | 84 hours | 9.5 hours |

### Remaining Bottleneck

Selective Search takes ~2 seconds per image — it's now the speed bottleneck.

---

## Faster R-CNN (2015)

**Paper:** Ren, He, Girshick, & Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"

**Key innovation:** Replace Selective Search with a **learned Region Proposal Network (RPN)** that shares features with the detector.

### Architecture

```
Input Image
    ↓
Backbone CNN (e.g., ResNet-50 + FPN)
    ↓
Feature Map (shared)
    ↓
┌──────────────────────┐
│  Region Proposal     │
│  Network (RPN)       │
│  → ~300 proposals    │
└──────────────────────┘
    ↓
RoI Align → Fixed-size features
    ↓
Detection Head:
    ├── Classification → Class + confidence
    └── Box Regression → Refined coordinates
```

### Region Proposal Network (RPN)

The RPN is a small fully-convolutional network that slides over the feature map:

```
Feature Map
    ↓
3×3 Conv (256 channels) → Intermediate feature
    ↓
Two sibling 1×1 conv heads:
    ├── cls layer → 2k scores (object / not-object)
    └── reg layer → 4k coordinates (Δx, Δy, Δw, Δh)
```

At each spatial location, $k$ anchor boxes are placed (typically $k = 9$: 3 scales × 3 aspect ratios).

**RPN Training:**

An anchor is labeled positive if:
- IoU with any ground truth ≥ 0.7, OR
- Highest IoU with a ground truth box

An anchor is labeled negative if:
- IoU with all ground truths < 0.3

Anchors between 0.3 and 0.7 IoU are ignored during training.

**RPN Loss:**

$$\mathcal{L}_{\text{RPN}} = \frac{1}{N_{\text{cls}}} \sum_i \mathcal{L}_{\text{cls}}(p_i, p_i^*) + \lambda \frac{1}{N_{\text{reg}}} \sum_i p_i^* \cdot \mathcal{L}_{\text{reg}}(t_i, t_i^*)$$

### RoI Align (from Mask R-CNN)

RoI Pooling's quantization causes misalignment. **RoI Align** fixes this:

1. Compute exact (floating-point) coordinates on the feature map
2. Sample at regular points within each bin using **bilinear interpolation**
3. No rounding at any step

**Impact:** +1-2 AP improvement, especially for small objects and segmentation masks.

### Training Strategy

**4-Step Alternating Training:**
1. Train RPN with ImageNet-pretrained backbone
2. Train Fast R-CNN using RPN proposals (separate backbone)
3. Fix shared backbone, fine-tune RPN
4. Fix shared backbone, fine-tune Fast R-CNN

**Modern approach:** Joint end-to-end training (both RPN and detection head together).

### Implementation Details

| Component | Details |
|-----------|---------|
| Backbone | ResNet-50 or ResNet-101 + FPN |
| Anchors | 3 scales × 3 ratios = 9 per location |
| RPN proposals | ~300 (after NMS from ~6000 pre-NMS) |
| RoI pool size | 7×7 |
| NMS threshold | 0.7 for RPN, 0.5 for detection |
| Mini-batch | 256 anchors (128 pos + 128 neg) |

---

## Cascade R-CNN (2018)

**Paper:** Cai & Vasconcelos, "Cascade R-CNN: Delving into High Quality Object Detection"

**Problem:** A single IoU threshold is a poor choice:
- Low threshold (0.5) → many false positives
- High threshold (0.7) → too few positive training samples

**Solution:** Use a sequence of detectors trained at increasing IoU thresholds:

```
RPN → Head₁ (IoU=0.5) → Head₂ (IoU=0.6) → Head₃ (IoU=0.7)
        ↓ boxes              ↓ boxes              ↓ final
```

Each stage refines the proposals from the previous stage, progressively improving localization quality.

**Result:** +2-4 AP improvement over Faster R-CNN, especially at high IoU thresholds.

---

## Speed vs Accuracy Comparison

| Model | Backbone | AP (COCO) | FPS | Year |
|-------|----------|-----------|-----|------|
| Faster R-CNN | ResNet-50-FPN | 36.4 | 15 | 2015 |
| Faster R-CNN | ResNet-101-FPN | 38.5 | 10 | 2015 |
| Cascade R-CNN | ResNet-50-FPN | 40.3 | 12 | 2018 |
| Cascade R-CNN | ResNet-101-FPN | 42.1 | 8 | 2018 |

---

## Key Takeaways

1. **R-CNN** proved CNN features work for detection but was impractically slow (47s/image)
2. **Fast R-CNN** shared computation across proposals — 150× speedup with better accuracy
3. **Faster R-CNN** replaced hand-crafted proposals with a learned RPN — the first fully end-to-end detector
4. **RoI Align** eliminated quantization artifacts, improving spatial precision
5. **Cascade R-CNN** showed that multi-stage refinement at increasing IoU thresholds yields better high-quality detections
6. **Faster R-CNN + FPN** remains the dominant baseline for two-stage detection
7. The R-CNN family's ideas (proposals, shared features, multi-task loss) influenced virtually all subsequent detectors

## References

- Girshick, R., Donahue, J., Darrell, T., & Malik, J. (2014). Rich feature hierarchies for accurate object detection and semantic segmentation. *CVPR 2014*. [arXiv:1311.1524](https://arxiv.org/abs/1311.1524)
- Girshick, R. (2015). Fast R-CNN. *ICCV 2015*. [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS 2015*. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. *ICCV 2017*. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Cai, Z., & Vasconcelos, N. (2018). Cascade R-CNN: Delving into High Quality Object Detection. *CVPR 2018*. [arXiv:1712.00726](https://arxiv.org/abs/1712.00726)
