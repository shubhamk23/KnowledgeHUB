---
title: "YOLO: Real-Time Object Detection"
slug: yolo-series
summary: "Complete guide to the YOLO family — from YOLOv1's grid-based design to YOLOv8's anchor-free architecture — with speed-accuracy tradeoffs."
tags: ["YOLO", "object-detection", "real-time", "YOLOv8", "anchor-free", "computer-vision", "deep-learning"]
visibility: public
level: intermediate
---

# YOLO: Real-Time Object Detection

## The YOLO Philosophy

**YOLO (You Only Look Once)** redefined object detection by framing it as a single regression problem — predicting bounding boxes and class probabilities directly from the full image in one forward pass.

**Two-stage detectors** (Faster R-CNN): Propose regions → classify each region
**YOLO approach:** Look at the entire image once → predict everything simultaneously

This makes YOLO dramatically faster — enabling real-time detection at 30-150+ FPS.

---

## YOLOv1 (2016)

**Paper:** Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"

### Core Idea

Divide the image into an $S \times S$ grid. Each grid cell is responsible for detecting objects whose center falls within that cell.

Each cell predicts:
- $B$ bounding boxes, each with 5 values: $(x, y, w, h, \text{confidence})$
- $C$ class probabilities (shared across all boxes in the cell)

**Output tensor shape:** $S \times S \times (B \times 5 + C)$

With $S=7$, $B=2$, $C=20$ (PASCAL VOC): output is $7 \times 7 \times 30$

### Architecture

```
Input (448×448)
    ↓
24 Conv layers (inspired by GoogLeNet)
    ↓
2 FC layers
    ↓
Output: 7×7×30 tensor
```

### Confidence Score

$$\text{Confidence} = P(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}$$

At inference, class-specific confidence:
$$P(\text{Class}_i | \text{Object}) \times P(\text{Object}) \times \text{IoU} = P(\text{Class}_i) \times \text{IoU}$$

### YOLOv1 Loss Function

$$\mathcal{L} = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]$$
$$+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]$$
$$+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2$$
$$+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2$$
$$+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

Key design choices:
- **Square root of width/height:** Penalizes errors more for small objects
- $\lambda_{\text{coord}} = 5$: Upweight localization loss
- $\lambda_{\text{noobj}} = 0.5$: Downweight no-object confidence loss (most cells have no object)

### Performance

| Metric | Value |
|--------|-------|
| Speed | 45 FPS (155 FPS for Fast YOLO) |
| mAP (VOC 2007) | 63.4% |
| Real-time? | Yes |

**Limitations:**
- Each cell predicts only $B=2$ boxes → struggles with small, grouped objects
- Only one class per cell → can't detect two different objects in the same cell
- Coarse spatial resolution (7×7 grid)

---

## YOLOv2 / YOLO9000 (2017)

**Paper:** Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger"

### Key Improvements

| Improvement | Effect |
|------------|--------|
| Batch Normalization | +2% mAP, removed dropout |
| High-resolution classifier | Pretrain at 448×448, not 224 |
| Anchor boxes (from Faster R-CNN) | Better box priors via k-means clustering |
| Dimension clusters | $k=5$ anchors from k-means on training boxes |
| Direct location prediction | Constrain center to grid cell using sigmoid |
| Multi-scale training | Random input sizes {320, 352, ..., 608} |
| Passthrough layer | Concatenate high-res features |
| Darknet-19 backbone | 19 conv layers, faster than VGG |

**Result:** 78.6% mAP on VOC 2007 at 40 FPS.

### YOLO9000

Trained on both ImageNet (classification) and COCO (detection) using a **WordTree** hierarchy — a tree structure combining ImageNet and COCO labels. Could detect 9000+ categories.

---

## YOLOv3 (2018)

**Paper:** Redmon & Farhadi, "YOLOv3: An Incremental Improvement"

### Key Changes

1. **Darknet-53 backbone:** 53-layer network with residual connections
2. **Multi-scale predictions:** Detections at 3 scales (like FPN)
3. **Independent logistic classifiers:** Binary cross-entropy per class (supports multi-label)
4. **9 anchor boxes:** 3 per scale, chosen via k-means

### Multi-Scale Detection

```
Scale 1 (13×13): 3 anchors → Large objects
Scale 2 (26×26): 3 anchors → Medium objects
Scale 3 (52×52): 3 anchors → Small objects
```

Each scale predicts: $N \times N \times [3 \times (4 + 1 + 80)]$ for COCO (80 classes).

| Metric | Value |
|--------|-------|
| AP (COCO) | 33.0 |
| AP50 (COCO) | 57.9 |
| Speed | 51ms on Titan X (~20 FPS) |

---

## YOLOv4 (2020)

**Paper:** Bochkovskiy, Wang, & Liao, "YOLOv4: Optimal Speed and Accuracy of Object Detection"

Comprehensive survey and combination of modern techniques:

### Bag of Freebies (training tricks, no inference cost)

| Technique | Category |
|-----------|----------|
| CutMix, Mosaic augmentation | Data augmentation |
| DropBlock regularization | Regularization |
| CIoU loss | Better localization loss |
| Label smoothing | Regularization |
| Cosine annealing scheduler | Learning rate |

### Bag of Specials (inference cost, big accuracy gain)

| Technique | Category |
|-----------|----------|
| CSPNet backbone | Efficient feature extraction |
| SPP (Spatial Pyramid Pooling) | Multi-scale features |
| PANet (Path Aggregation Network) | Better FPN |
| Mish activation | Better gradient flow |
| SAM (Spatial Attention Module) | Attention |

### CIoU Loss

Standard IoU loss doesn't account for distance or aspect ratio when boxes don't overlap. CIoU adds:

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

Where:
- $\rho^2(b, b^{gt})$ = squared Euclidean distance between centers
- $c$ = diagonal of smallest enclosing box
- $v$ = aspect ratio consistency term
- $\alpha$ = trade-off parameter

**Result:** 43.5% AP on COCO at 65 FPS (Tesla V100).

---

## YOLOv5 (2020)

**Released by Ultralytics** (not an academic paper). PyTorch implementation with focus on engineering and usability.

### Key Features
- PyTorch native (vs. Darknet C framework)
- Auto-anchor optimization
- Mosaic augmentation
- Mixed precision training (FP16)
- Model export (ONNX, CoreML, TFLite, TensorRT)
- Extensive CLI and Python API

### Model Variants

| Model | Params | AP (COCO) | Speed (V100) |
|-------|--------|-----------|-------------|
| YOLOv5n | 1.9M | 28.0 | 6.3ms |
| YOLOv5s | 7.2M | 37.4 | 6.4ms |
| YOLOv5m | 21.2M | 45.4 | 8.2ms |
| YOLOv5l | 46.5M | 49.0 | 10.1ms |
| YOLOv5x | 86.7M | 50.7 | 12.1ms |

---

## YOLOv8 (2023)

**Released by Ultralytics.** State-of-the-art YOLO with major architectural changes.

### Architecture Changes

1. **Anchor-free detection:** No predefined anchor boxes — directly predicts object centers
2. **Decoupled head:** Separate branches for classification and regression
3. **C2f module:** Cross-Stage Partial with 2 convolutions (improved feature flow)
4. **Distribution Focal Loss (DFL):** Predicts bounding box as a distribution over discrete values

### Anchor-Free vs Anchor-Based

| Aspect | Anchor-Based (YOLOv5) | Anchor-Free (YOLOv8) |
|--------|----------------------|---------------------|
| Prior boxes | Yes (predefined) | No |
| Predictions | Offsets from anchors | Direct center + distance |
| Hyperparameters | Anchor sizes, ratios | Fewer to tune |
| Assignment | IoU-based | Task-Aligned Assignment |
| Performance | Good | Better (simpler, faster) |

### Task-Aligned Assignment (TAL)

YOLOv8 uses a dynamic label assignment strategy:

$$t = s^\alpha \times u^\beta$$

Where $s$ = classification score, $u$ = IoU, and $\alpha, \beta$ control the balance. This assigns positive samples based on both classification quality and localization quality.

### YOLOv8 Model Family

| Model | Params | AP (COCO) | Speed (A100) |
|-------|--------|-----------|-------------|
| YOLOv8n | 3.2M | 37.3 | 0.99ms |
| YOLOv8s | 11.2M | 44.9 | 1.20ms |
| YOLOv8m | 25.9M | 50.2 | 1.83ms |
| YOLOv8l | 43.7M | 52.9 | 2.39ms |
| YOLOv8x | 68.2M | 53.9 | 3.53ms |

### Multi-Task Support

YOLOv8 supports multiple vision tasks with the same architecture:

| Task | Output | Command |
|------|--------|---------|
| Detection | Bounding boxes + classes | `yolo detect` |
| Segmentation | Instance masks | `yolo segment` |
| Classification | Image label | `yolo classify` |
| Pose estimation | Keypoints | `yolo pose` |

---

## YOLO Evolution Timeline

| Version | Year | Key Innovation | Speed-Accuracy |
|---------|------|---------------|----------------|
| YOLOv1 | 2016 | Single-shot grid detection | 45 FPS, 63.4% mAP |
| YOLOv2 | 2017 | Anchor boxes, multi-scale training | 40 FPS, 78.6% mAP |
| YOLOv3 | 2018 | FPN-style multi-scale, Darknet-53 | 20 FPS, 33.0 AP |
| YOLOv4 | 2020 | Bag of freebies/specials, CSPNet | 65 FPS, 43.5 AP |
| YOLOv5 | 2020 | PyTorch, engineering focus | Variable |
| YOLOv8 | 2023 | Anchor-free, decoupled head, DFL | 1-4ms, 37-54 AP |

---

## Key Takeaways

1. **YOLO's core insight:** Treat detection as a single regression problem — one forward pass for all predictions
2. **Anchor boxes** (v2-v5) improved accuracy but added complexity; **anchor-free** (v8) is simpler and better
3. **Multi-scale detection** (v3+) is essential for detecting objects of different sizes
4. **Training tricks** (mosaic augmentation, CIoU loss, label smoothing) provide free accuracy gains
5. **YOLOv8** is the current best choice for real-time detection — anchor-free, multi-task, excellent tooling
6. **Speed-accuracy tradeoff:** Choose model size (n/s/m/l/x) based on deployment constraints
7. **Ultralytics ecosystem** provides the most practical YOLO implementation with export to all major inference frameworks

## References

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR 2016*. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)
- Redmon, J., & Farhadi, A. (2017). YOLO9000: Better, Faster, Stronger. *CVPR 2017*. [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)
- Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. [arXiv:1804.02767](https://arxiv.org/abs/1804.02767)
- Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal Speed and Accuracy of Object Detection. [arXiv:2004.10934](https://arxiv.org/abs/2004.10934)
- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. [GitHub](https://github.com/ultralytics/ultralytics)
- Zheng, Z., et al. (2020). Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression. *AAAI 2020*. [arXiv:1911.08287](https://arxiv.org/abs/1911.08287)
