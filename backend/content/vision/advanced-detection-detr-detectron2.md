---
title: "DETR & Detectron2: Modern Detection"
slug: advanced-detection-detr-detectron2
summary: "End-to-end object detection with Transformers — DETR's bipartite matching, Deformable DETR's efficiency, and Detectron2's modular framework."
tags: ["DETR", "Detectron2", "transformer-detection", "bipartite-matching", "end-to-end", "computer-vision", "deep-learning"]
visibility: public
level: advanced
---

# DETR & Detectron2: Modern Detection

## The Problems with Traditional Detection

Traditional detectors (Faster R-CNN, YOLO) rely on hand-designed components:
- **Anchor boxes** require careful tuning of sizes and aspect ratios
- **Non-Maximum Suppression (NMS)** is a heuristic post-processing step
- **Region proposal networks** add architectural complexity

These components work well but introduce hyperparameters, heuristics, and design decisions that limit elegance and generality. **Can we eliminate all of them?**

---

## DETR: Detection Transformer (2020)

**Paper:** Carion et al., "End-to-End Object Detection with Transformers"

DETR was the first fully end-to-end detector — no anchors, no NMS, no hand-crafted components. It reformulates detection as a **direct set prediction problem**.

### Architecture

```
Input Image (800×1333)
    ↓
CNN Backbone (ResNet-50) → Feature map (H/32 × W/32 × 2048)
    ↓
1×1 Conv → (H/32 × W/32 × 256) + Positional encoding
    ↓
Transformer Encoder (6 layers)
    ↓
N learned object queries → Transformer Decoder (6 layers)
    ↓
N parallel prediction heads:
    ├── Class prediction (including ∅ = "no object")
    └── Box prediction (normalized center + size)
```

### Object Queries

DETR uses $N$ **learned object queries** (typically $N = 100$) — each query is a learnable embedding that attends to the encoded image features through cross-attention in the decoder. Each query produces one prediction.

The queries learn to specialize: some attend to large objects, others to small ones; some to specific spatial regions.

### Bipartite Matching Loss

The core innovation. Instead of matching predictions to ground truth using IoU thresholds (like Faster R-CNN), DETR finds the **optimal one-to-one assignment** using the Hungarian algorithm.

Given $N$ predictions and $M$ ground truth objects ($M \leq N$):

**Step 1:** Compute the cost matrix between all predictions and ground truths:

$$\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\mathbb{1}_{\{c_i \neq \varnothing\}} \hat{p}_{\sigma(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$

**Step 2:** Find the permutation $\hat{\sigma}$ that minimizes total cost:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

This is solved efficiently by the Hungarian algorithm in $O(N^3)$.

**Step 3:** Compute the training loss using the optimal assignment:

$$\mathcal{L}_{\text{Hungarian}} = \sum_{i=1}^{N} \left[-\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)})\right]$$

The box loss combines L1 loss and generalized IoU (GIoU):

$$\mathcal{L}_{\text{box}} = \lambda_{\text{iou}} \mathcal{L}_{\text{GIoU}} + \lambda_{\text{L1}} \| b_i - \hat{b}_{\sigma(i)} \|_1$$

### Why Bipartite Matching Matters

| Traditional Detection | DETR |
|----------------------|------|
| Multiple predictions per object → NMS needed | One-to-one assignment → no NMS |
| Anchor design required | No anchors |
| IoU threshold tuning | Optimal global matching |
| Post-processing heuristics | Clean end-to-end training |

### DETR Performance

| Model | Backbone | AP (COCO) | FPS | Params |
|-------|----------|-----------|-----|--------|
| DETR | ResNet-50 | 42.0 | 28 | 41M |
| DETR | ResNet-101 | 43.5 | 20 | 60M |
| DETR-DC5 | ResNet-50 | 43.3 | 12 | 41M |

### DETR Limitations

1. **Slow convergence:** 500 epochs to converge (vs. ~36 for Faster R-CNN) — the attention in the decoder takes very long to learn where to attend
2. **Poor small object detection:** Global attention struggles with small objects in high-resolution feature maps
3. **Fixed number of queries:** $N$ must be set larger than the maximum objects in any image

---

## Deformable DETR (2021)

**Paper:** Zhu et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection"

Addressed DETR's convergence and small object issues with **deformable attention**.

### Deformable Attention

Instead of attending to all spatial locations (global attention), each query attends to a small set of **learned sampling points** around a reference point:

$$\text{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \left[\sum_{k=1}^{K} A_{mqk} \cdot x(p_q + \Delta p_{mqk})\right]$$

Where:
- $p_q$ = reference point for query $q$
- $\Delta p_{mqk}$ = learned sampling offsets (like deformable convolutions)
- $A_{mqk}$ = attention weights (sum to 1)
- $K$ = number of sampling points (typically 4)
- $M$ = number of attention heads

**Complexity:** $O(NK)$ instead of $O(N \cdot HW)$ — linear in spatial size.

### Multi-Scale Deformable Attention

Deformable DETR operates on multi-scale features from FPN:

```
ResNet-50 → FPN → {P3, P4, P5, P6}
    ↓
Deformable Transformer Encoder (6 layers)
    ↓
Deformable Transformer Decoder (6 layers)
    ↓
Predictions
```

Each query samples from all scale levels simultaneously, enabling natural multi-scale detection.

### Results

| Model | Epochs | AP (COCO) | FPS |
|-------|--------|-----------|-----|
| DETR | 500 | 42.0 | 28 |
| Deformable DETR | 50 | 43.8 | 19 |
| + iterative box refinement | 50 | 46.2 | 19 |
| + two-stage | 50 | 46.9 | 16 |

**10× faster convergence** with better accuracy, especially on small objects.

---

## DINO: DETR with Improved deNoising anchOr boxes (2023)

**Paper:** Zhang et al., "DINO: DETR with Improved deNoising anchOr boxes for End-to-End Object Detection"

The current state-of-the-art in the DETR family.

### Key Innovations

1. **Contrastive denoising training:** Add noise to ground truth boxes and train the model to denoise them — provides direct supervision for the decoder
2. **Mixed query selection:** Initialize decoder queries using both learned embeddings and top encoder features
3. **Look-forward-twice:** Each decoder layer refines boxes by looking at the next layer's gradients

### Performance

| Model | Backbone | Epochs | AP (COCO) |
|-------|----------|--------|-----------|
| DINO-4scale | ResNet-50 | 12 | 49.0 |
| DINO-4scale | ResNet-50 | 36 | 50.9 |
| DINO-5scale | Swin-L | 36 | 63.3 |

DINO with Swin-L achieved **63.3 AP** on COCO — the highest single-model result, surpassing all previous detectors.

---

## Co-DETR (2023)

**Paper:** Zong et al., "DETRs with Collaborative Hybrid Assignments Training"

Co-DETR uses auxiliary detection heads (one-to-many assignment) during training alongside DETR's one-to-one matching, then discards the auxiliary heads at inference.

| Model | Backbone | AP (COCO) |
|-------|----------|-----------|
| Co-DETR | Swin-L | 65.6 |
| Co-DETR | ViT-L (LSJ) | 66.0 |

This pushed COCO AP beyond 65 for the first time.

---

## Detectron2 Framework

**Developed by Facebook AI Research (FAIR)**. Detectron2 is a modular, extensible framework for building detection and segmentation models.

### Architecture

```
Detectron2 Engine
├── Config System (YAML + Python)
├── Data Pipeline (DatasetMapper, augmentation)
├── Model Zoo (pretrained model library)
└── Modular Components:
    ├── Backbone (ResNet, RegNet, ViT, Swin)
    ├── Proposal Generator (RPN, anchor-free)
    ├── ROI Heads (box, mask, keypoint)
    └── Meta-Architecture:
        ├── GeneralizedRCNN (Faster/Mask R-CNN)
        ├── RetinaNet (one-stage)
        ├── FCOS (anchor-free)
        └── PanopticFPN (panoptic)
```

### Supported Tasks

| Task | Model | Key Metric |
|------|-------|-----------|
| Object Detection | Faster R-CNN, RetinaNet, FCOS | AP (COCO) |
| Instance Segmentation | Mask R-CNN, PointRend | AP (mask) |
| Panoptic Segmentation | Panoptic FPN | PQ |
| Keypoint Detection | Mask R-CNN (keypoint head) | AP (keypoint) |
| Semantic Segmentation | DeepLabv3+, Mask2Former | mIoU |

### Model Zoo Highlights

| Model | Backbone | AP (COCO) | Inference (ms) |
|-------|----------|-----------|---------------|
| Faster R-CNN | R50-FPN | 37.9 | 38 |
| Faster R-CNN | R101-FPN | 42.0 | 52 |
| Mask R-CNN | R50-FPN | 38.6 | 46 |
| Cascade R-CNN | R50-FPN | 44.3 | 56 |
| DETR | R50 | 42.0 | 36 |

### When to Use Detectron2

| Scenario | Use Detectron2? |
|----------|----------------|
| Research prototyping | Yes — modular, easy to modify |
| Production deployment | Maybe — consider ONNX/TensorRT export |
| Custom architecture experiments | Yes — plug-and-play components |
| Simple detection (just want predictions) | No — use Ultralytics YOLOv8 instead |
| Edge deployment | No — use lighter frameworks |

---

## DETR Family Evolution

| Model | Year | Key Innovation | AP (COCO) | Convergence |
|-------|------|---------------|-----------|-------------|
| DETR | 2020 | Bipartite matching, no NMS | 42.0 | 500 epochs |
| Deformable DETR | 2021 | Deformable attention, multi-scale | 46.9 | 50 epochs |
| DAB-DETR | 2022 | Dynamic anchor boxes as queries | 48.7 | 50 epochs |
| DN-DETR | 2022 | Denoising training | 48.6 | 50 epochs |
| DINO | 2023 | Denoising + mixed queries | 63.3 | 36 epochs |
| Co-DETR | 2023 | Collaborative hybrid training | 66.0 | 36 epochs |

### The Convergence Problem — Solved

DETR's original 500-epoch training was its biggest practical limitation. The progression from 500 → 50 → 36 epochs came through:
1. **Deformable attention** — focused attention on relevant locations
2. **Denoising training** — direct box supervision accelerates decoder learning
3. **Better query initialization** — encoder features bootstrap decoder queries

---

## Practical Comparison: Modern Detectors

| Detector | Type | AP (COCO) | Speed | Best For |
|----------|------|-----------|-------|----------|
| YOLOv8-X | One-stage, anchor-free | 53.9 | 3.5ms (A100) | Real-time production |
| DINO (R50) | DETR-style | 50.9 | ~50ms | Research, high accuracy |
| Co-DETR (ViT-L) | DETR-style | 66.0 | ~100ms | Maximum accuracy |
| Faster R-CNN (R50-FPN) | Two-stage | 37.9 | 38ms | Well-understood baseline |
| Cascade R-CNN (R101-FPN) | Multi-stage | 42.1 | 75ms | High-quality detections |

---

## Key Takeaways

1. **DETR** eliminated anchors, NMS, and hand-designed components — reformulating detection as set prediction with bipartite matching
2. **The Hungarian algorithm** finds optimal one-to-one assignment between predictions and ground truth, removing the need for NMS
3. **Deformable attention** solved DETR's convergence and efficiency issues by attending to learned sampling points instead of all locations
4. **DINO and Co-DETR** represent the current state of the art, achieving 63-66 AP on COCO through denoising training and hybrid assignments
5. **Detectron2** provides a modular framework for building and experimenting with detection/segmentation models
6. For **real-time applications**, YOLO-family detectors remain the best choice; DETR-family excels when accuracy is the priority
7. The trend in detection is toward **simpler, more unified architectures** — Transformers are replacing the complex multi-component pipelines of traditional detectors

## References

- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. *ECCV 2020*. [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2021). Deformable DETR: Deformable Transformers for End-to-End Object Detection. *ICLR 2021*. [arXiv:2010.04159](https://arxiv.org/abs/2010.04159)
- Zhang, H., et al. (2023). DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. *ICLR 2023*. [arXiv:2203.03605](https://arxiv.org/abs/2203.03605)
- Zong, Z., et al. (2023). DETRs with Collaborative Hybrid Assignments Training. *ICCV 2023*. [arXiv:2211.12860](https://arxiv.org/abs/2211.12860)
- Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019). Detectron2. [GitHub](https://github.com/facebookresearch/detectron2)
- Liu, S., et al. (2022). DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR. *ICLR 2022*. [arXiv:2201.12329](https://arxiv.org/abs/2201.12329)
