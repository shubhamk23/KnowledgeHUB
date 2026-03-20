---
title: "Object Detection Fundamentals"
slug: object-detection-fundamentals
summary: "A beginner's guide to object detection — bounding boxes, IoU, anchor boxes, NMS, and evaluation metrics like mAP."
tags: ["object-detection", "bounding-box", "IoU", "NMS", "mAP", "anchor-boxes", "computer-vision"]
visibility: public
level: beginner
---

# Object Detection Fundamentals

## What is Object Detection?

**Object detection** combines two tasks:
1. **Localization:** Where are the objects? → Bounding boxes
2. **Classification:** What are they? → Class labels

Unlike image classification (one label per image), detection identifies **multiple objects** with their locations.

**Output format per detection:**
$$[\text{class}, \text{confidence}, x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$

---

## Bounding Boxes

### Representation Formats

There are two common ways to represent bounding boxes:

**Corner format:** $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$
- Top-left and bottom-right corners
- Used in PASCAL VOC, COCO, most frameworks

**Center format:** $(x_c, y_c, w, h)$
- Center coordinates + width and height
- Used internally by YOLO-style detectors

**Conversion:**
$$x_c = \frac{x_{\min} + x_{\max}}{2}, \quad y_c = \frac{y_{\min} + y_{\max}}{2}$$
$$w = x_{\max} - x_{\min}, \quad h = y_{\max} - y_{\min}$$

### Coordinate Systems

| Dataset/Format | Coordinates | Origin |
|---------------|------------|--------|
| PASCAL VOC | $(x_{\min}, y_{\min}, x_{\max}, y_{\max})$ absolute | Top-left |
| COCO | $(x, y, w, h)$ absolute | Top-left |
| YOLO | $(x_c, y_c, w, h)$ normalized [0,1] | Top-left |

---

## Intersection over Union (IoU)

**IoU** (also called Jaccard Index) measures how well a predicted box matches the ground truth:

$$\text{IoU} = \frac{|\text{Pred} \cap \text{GT}|}{|\text{Pred} \cup \text{GT}|} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

| IoU Value | Meaning |
|-----------|---------|
| 0.0 | No overlap |
| 0.5 | Moderate overlap (common threshold) |
| 0.75 | Strong overlap (strict threshold) |
| 1.0 | Perfect match |

**Standard thresholds:**
- **IoU ≥ 0.5:** PASCAL VOC metric (AP50)
- **IoU ≥ 0.75:** Strict metric (AP75)
- **IoU averaged 0.5:0.95:** COCO primary metric (AP)

### Computing IoU

```
Intersection coordinates:
  x1 = max(pred_x1, gt_x1)
  y1 = max(pred_y1, gt_y1)
  x2 = min(pred_x2, gt_x2)
  y2 = min(pred_y2, gt_y2)

  intersection = max(0, x2 - x1) × max(0, y2 - y1)
  union = area_pred + area_gt - intersection
  IoU = intersection / union
```

---

## Anchor Boxes

### The Problem

Objects come in many shapes and sizes. How does the model know what shapes to look for?

### The Solution: Predefined Templates

**Anchor boxes** (also called prior boxes or default boxes) are predefined bounding box templates placed at each position in the feature map.

**Example anchor set at one position:**
```
┌─────────────┐  ┌───────┐  ┌──┐
│             │  │       │  │  │
│  1:2 wide   │  │ 1:1   │  │  │ 2:1 tall
│             │  │ square │  │  │
└─────────────┘  └───────┘  │  │
                             │  │
                             └──┘
```

At each feature map location, the model places $k$ anchors with different aspect ratios and scales. For each anchor, it predicts:
1. **Offset adjustments** $(Δx, Δy, Δw, Δh)$ to refine the box position
2. **Objectness score** — probability that the anchor contains an object
3. **Class probabilities** — which class the object belongs to

### Anchor Design Choices

| Detector | Aspect Ratios | Scales | Anchors/Position |
|----------|--------------|--------|------------------|
| Faster R-CNN | {1:2, 1:1, 2:1} | {128², 256², 512²} | 9 |
| SSD | {1:2, 1:1, 2:1, 1:3, 3:1} | Multi-scale | 6 |
| RetinaNet | {1:2, 1:1, 2:1} | {2⁰, 2^(1/3), 2^(2/3)} | 9 |

**Modern trend:** Anchor-free detectors (FCOS, CenterNet, YOLOv8) eliminate anchors entirely by predicting object centers directly.

---

## Non-Maximum Suppression (NMS)

### The Problem

Detectors produce many overlapping predictions for the same object. We need to keep only the best one.

### How NMS Works

1. Sort all detections by confidence score (descending)
2. Take the highest-confidence detection → keep it
3. Remove all remaining detections that have IoU > threshold with the kept detection
4. Repeat from step 2 until no detections remain

**NMS threshold:** Typically 0.5 — detections overlapping more than 50% with a higher-scored detection are suppressed.

### NMS Variants

| Variant | How It Works | Advantage |
|---------|-------------|-----------|
| Standard NMS | Hard removal above IoU threshold | Simple, fast |
| Soft-NMS | Decay score instead of removing | Better for overlapping objects |
| DIoU-NMS | Uses distance-based IoU | More robust suppression |
| Matrix NMS | Parallel computation | Faster (used in YOLACT) |

**Soft-NMS** reduces the score instead of eliminating detections:
$$s_i = s_i \cdot e^{-\text{IoU}^2 / \sigma}$$

This helps when objects genuinely overlap (e.g., a crowd of people).

---

## Feature Pyramid Networks (FPN)

### The Multi-Scale Problem

Small objects need high-resolution features (early layers), while large objects need semantic features (deep layers). How do we detect both?

### FPN Architecture

FPN creates a multi-scale feature pyramid with both strong semantics and high resolution:

```
Bottom-up          Top-down (FPN)
(backbone)         (with lateral connections)

C5 (7×7)    ────→  P5 (7×7)      ← Large objects
    ↓         ↗
C4 (14×14)  ────→  P4 (14×14)    ← Medium objects
    ↓         ↗
C3 (28×28)  ────→  P3 (28×28)    ← Small objects
    ↓         ↗
C2 (56×56)  ────→  P2 (56×56)    ← Very small objects
```

Each level uses:
- **Lateral connection:** 1×1 conv to match channel dimensions
- **Top-down pathway:** 2× upsampling + element-wise addition
- **3×3 conv:** Reduce aliasing after merging

**Impact:** FPN improved small object detection by 8+ AP points and became standard in all modern detectors.

---

## Evaluation Metrics

### Precision and Recall

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{Correct detections}}{\text{All detections}}$$

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{Correct detections}}{\text{All ground truths}}$$

A detection is a **True Positive (TP)** if:
- IoU with a ground truth box ≥ threshold
- The predicted class matches the ground truth class
- The ground truth box hasn't already been matched

### Average Precision (AP)

AP summarizes the precision-recall curve for a single class:

$$\text{AP} = \int_0^1 p(r) \, dr$$

In practice, computed using the 11-point or all-point interpolation method.

### Mean Average Precision (mAP)

mAP averages AP across all classes:

$$\text{mAP} = \frac{1}{C} \sum_{c=1}^{C} \text{AP}_c$$

### COCO Metrics

| Metric | IoU Threshold | Description |
|--------|--------------|-------------|
| AP | 0.50:0.05:0.95 | Primary metric (averaged over 10 thresholds) |
| AP50 | 0.50 | PASCAL VOC-style (lenient) |
| AP75 | 0.75 | Strict localization |
| AP_S | 0.50:0.95 | Small objects (area < 32²) |
| AP_M | 0.50:0.95 | Medium objects (32² < area < 96²) |
| AP_L | 0.50:0.95 | Large objects (area > 96²) |

---

## Two-Stage vs One-Stage Detectors

| Aspect | Two-Stage | One-Stage |
|--------|-----------|-----------|
| Examples | Faster R-CNN, Cascade R-CNN | YOLO, SSD, RetinaNet |
| Speed | Slower (~5-15 FPS) | Faster (~30-150 FPS) |
| Accuracy | Generally higher | Competitive (with focal loss) |
| Pipeline | Region proposal → classify | Direct prediction |
| Use case | When accuracy is critical | Real-time applications |

**Focal Loss** (Lin et al., 2017) closed the accuracy gap for one-stage detectors by addressing class imbalance:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

With $\gamma = 2$, easy examples (high $p_t$) contribute very little to the loss, letting the model focus on hard examples.

---

## Key Takeaways

1. **Object detection** = localization + classification — predicting bounding boxes and class labels for multiple objects
2. **IoU** is the fundamental metric for measuring box overlap — standard thresholds are 0.5 and 0.75
3. **Anchor boxes** provide shape priors that the model refines — though modern detectors are moving anchor-free
4. **NMS** removes duplicate detections — Soft-NMS is better for crowded scenes
5. **FPN** enables multi-scale detection by combining high-resolution and semantic features
6. **mAP** is the standard evaluation metric — COCO's AP (averaged over IoU 0.5:0.95) is the gold standard
7. **Two-stage detectors** prioritize accuracy; **one-stage detectors** prioritize speed — focal loss narrows the gap

## References

- Girshick, R. (2015). Fast R-CNN. *ICCV 2015*. [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)
- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. *NeurIPS 2015*. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)
- Lin, T.-Y., et al. (2017). Feature Pyramid Networks for Object Detection. *CVPR 2017*. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Bodla, N., Singh, B., Chellappa, R., & Davis, L. S. (2017). Soft-NMS — Improving Object Detection With One Line of Code. *ICCV 2017*. [arXiv:1704.04503](https://arxiv.org/abs/1704.04503)
