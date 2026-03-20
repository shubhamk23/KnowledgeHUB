---
title: "Self-Supervised Learning in Vision"
slug: self-supervised-vision
summary: "How models learn visual representations without labels — contrastive learning (SimCLR, MoCo), masked image modeling (MAE, BEiT), and self-distillation (DINO, DINOv2)."
tags: ["self-supervised", "SimCLR", "MoCo", "MAE", "DINO", "contrastive-learning", "computer-vision", "deep-learning"]
visibility: public
level: advanced
---

# Self-Supervised Learning in Vision

## The Labeling Bottleneck

Supervised learning requires massive labeled datasets — ImageNet's 1.28M labels took years of human annotation. Self-supervised learning (SSL) learns visual representations from **unlabeled images** by solving pretext tasks that require understanding visual structure.

| Paradigm | Labels Needed | Data Scale | Representation Quality |
|----------|--------------|-----------|----------------------|
| Supervised (ImageNet) | 1.28M labeled | Fixed, curated | Good (task-specific) |
| Self-supervised | None | Unlimited, uncurated | Excellent (general) |
| Foundation models (SSL + scale) | None for pre-training | Billions of images | State-of-the-art |

**Why SSL works:** By forcing models to solve hard tasks on raw images — predicting missing patches, distinguishing transformations, maintaining consistency across views — the model must learn rich, transferable representations.

---

## Contrastive Learning

### Core Idea

Learn representations where **similar pairs** (different views of the same image) are close and **dissimilar pairs** (different images) are far apart in embedding space.

### SimCLR (2020)

**Paper:** Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"

The simplest and most influential contrastive framework.

#### Architecture

```
Image x
    ├── Augmentation t' → x'_i ─┐
    └── Augmentation t" → x'_j ─┤
                                 ↓
                     Encoder f(·) (ResNet-50)
                                 ↓
                     Projection head g(·) (MLP)
                                 ↓
                     z_i, z_j ∈ ℝ^128
                                 ↓
                     NT-Xent Loss (maximize agreement)
```

Two random augmentations of the same image form a **positive pair**. All other images in the batch form **negative pairs**.

#### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

For a positive pair $(i, j)$ in a batch of $N$ images ($2N$ augmented views):

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

Where $\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$ is cosine similarity and $\tau$ is the temperature (typically 0.1).

#### Critical Design Choices

| Component | Finding |
|-----------|---------|
| Augmentations | Random crop + color jitter is essential. Color jitter alone: 39.2% → combined: 64.0% linear probe |
| Projection head | Non-linear MLP head is crucial. Without it: 50.2% → with MLP: 64.0% |
| Batch size | Larger is better (4096 optimal). Provides more negatives per positive |
| Temperature $\tau$ | 0.1 works best. Too high → too uniform; too low → too peaked |

**Result:** 76.5% top-1 on ImageNet linear evaluation with ResNet-50 (4096 batch size, 800 epochs).

**Limitation:** Requires very large batches (4096+) for enough negatives — needs massive GPU memory.

---

### MoCo: Momentum Contrast (2020)

**Paper:** He et al., "Momentum Contrast for Unsupervised Visual Representation Learning"

Solved SimCLR's large-batch requirement with a **momentum-updated queue** of negatives.

#### Architecture

```
Query: x_q → Encoder f_q → q        (gradient-updated)
Key:   x_k → Encoder f_k → k        (momentum-updated, no gradient)
                 ↓
Queue: [k₁, k₂, ..., k_K]           (FIFO, K=65536)
                 ↓
InfoNCE Loss: pull q toward k⁺, push away from queue
```

#### Momentum Update

The key encoder $f_k$ is updated as an exponential moving average of the query encoder $f_q$:

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

With momentum $m = 0.999$. This provides a slowly-evolving, consistent encoder for the key representations stored in the queue.

#### InfoNCE Loss

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{i=0}^{K-1} \exp(q \cdot k_i^- / \tau)}$$

**Key advantage:** The queue provides $K = 65536$ negatives regardless of batch size. MoCo achieves strong results with batch sizes as small as 256.

### MoCo v2 and v3

| Version | Year | Key Changes | ImageNet Linear |
|---------|------|------------|-----------------|
| MoCo v1 | 2020 | Queue + momentum encoder | 60.6% |
| MoCo v2 | 2020 | + MLP head + augmentation (from SimCLR) | 71.1% |
| MoCo v3 | 2021 | Applied to ViT, stability tricks | 76.7% (ViT-B) |

---

### Contrastive Learning Without Negatives

A key problem with contrastive methods is **representation collapse** — without negatives, the model can trivially map all inputs to the same point. Several methods solved this without explicit negatives:

| Method | Year | How It Avoids Collapse |
|--------|------|----------------------|
| BYOL | 2020 | Asymmetric architecture + momentum encoder |
| SimSiam | 2021 | Stop-gradient operation |
| Barlow Twins | 2021 | Cross-correlation matrix redundancy reduction |
| VICReg | 2022 | Variance + invariance + covariance regularization |

#### BYOL (Bootstrap Your Own Latent)

$$\mathcal{L}_{\text{BYOL}} = \| \bar{q}_\theta(z_\theta) - \bar{z}'_\xi \|_2^2$$

Where $q_\theta$ is a predictor network on the online branch and $\bar{z}'_\xi$ is from the momentum-updated target. The stop-gradient on the target branch, combined with the momentum update, prevents collapse.

**Result:** 74.3% linear probe (ResNet-50, 300 epochs) — without any negatives.

---

## Masked Image Modeling (MIM)

### Core Idea

Inspired by BERT in NLP: **mask parts of the input and train the model to reconstruct them**. This forces the model to understand visual structure, context, and object composition.

### BEiT (2022)

**Paper:** Bao et al., "BEiT: BERT Pre-Training of Image Transformers"

#### Approach

1. Train a discrete VAE (dVAE) to tokenize image patches into visual tokens
2. Mask 40% of patches in the ViT input
3. Train ViT to predict the visual token IDs of masked patches

$$\mathcal{L}_{\text{BEiT}} = -\sum_{i \in \mathcal{M}} \log p(z_i | \tilde{x})$$

Where $\mathcal{M}$ is the set of masked positions and $z_i$ is the visual token from the dVAE.

### MAE: Masked Autoencoders (2022)

**Paper:** He et al., "Masked Autoencoders Are Scalable Vision Learners"

The simplest and most efficient MIM approach. Key insight: **mask 75% of patches** and reconstruct raw pixels.

#### Architecture

```
Input: 224×224 image → 196 patches (16×16)
    ↓
Mask 75% → Keep 49 visible patches (random)
    ↓
ViT Encoder (only on visible patches — very efficient!)
    ↓
Insert mask tokens at masked positions
    ↓
Lightweight ViT Decoder (8 layers)
    ↓
Reconstruct pixel values for masked patches
```

#### Why 75% Masking?

| Mask Ratio | Linear Probe | Fine-tune |
|-----------|-------------|-----------|
| 25% | 57.6% | 82.5% |
| 50% | 65.0% | 83.3% |
| 75% | 67.8% | 83.6% |
| 85% | 65.2% | 83.4% |

At 75%, the task is hard enough to force semantic understanding (can't just interpolate from neighbors), but not so hard that reconstruction becomes impossible.

#### Reconstruction Loss

Simple per-pixel MSE on masked patches:

$$\mathcal{L}_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \| x_i - \hat{x}_i \|^2$$

Where patches are normalized (zero mean, unit variance) before computing the loss.

#### Efficiency

Because the encoder only processes 25% of patches, MAE is **3-4× faster** than standard ViT training:

| Method | Pre-training Time (ViT-L) | ImageNet Fine-tune |
|--------|--------------------------|-------------------|
| Supervised ViT-L | 1× | 82.6% |
| MAE ViT-L | 0.6× | 85.9% |
| MAE ViT-H | 1.2× | 87.8% |

### SimMIM (2022)

**Paper:** Xie et al., "SimMIM: A Simple Framework for Masked Image Modeling"

Simplified MIM further — no tokenizer, no separate decoder. Just mask patches, pass through the full backbone, and predict raw pixels with a linear head.

**Result:** Comparable to MAE with even simpler design. Works well with Swin Transformer (not just ViT).

---

## Self-Distillation: DINO Family

### DINO (2021)

**Paper:** Caron et al., "Emerging Properties in Self-Supervised Vision Transformers"

Self-distillation without labels — a student network learns to match the output of a momentum-updated teacher network across different views.

#### Architecture

```
Image x
    ├── Global crops (2×, 224×224) → Teacher (momentum ViT)
    └── Local crops (6×, 96×96) → Student (ViT)
                                      ↓
                          Cross-entropy between
                          student and teacher outputs
```

#### Loss

$$\mathcal{L} = \sum_{x \in \{x_1^g, x_2^g\}} \sum_{\substack{x' \in V \\ x' \neq x}} H(P_t(x), P_s(x'))$$

Where $P_t$ and $P_s$ are softmax-normalized outputs of teacher and student, and $V$ is the set of all views (2 global + 6 local).

**Centering and sharpening** prevent collapse:
- Teacher outputs are centered by subtracting a running mean
- Temperature sharpening: $\tau_t = 0.04$ (teacher, sharp), $\tau_s = 0.1$ (student, softer)

#### Emergent Properties

DINO-trained ViTs exhibit remarkable properties without any supervision:

1. **Scene segmentation emerges in attention maps:** Self-attention heads learn to attend to semantically meaningful regions — foreground objects, parts, boundaries
2. **k-NN classification without fine-tuning:** 77.3% top-1 on ImageNet using just k-nearest neighbors on frozen features
3. **Copy detection:** DINO features are highly robust to image transformations

---

### DINOv2 (2023)

**Paper:** Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"

Scaled up DINO with curated data, combined objectives, and larger models to create **universal visual features**.

#### Key Improvements over DINO

| Aspect | DINO | DINOv2 |
|--------|------|--------|
| Data | ImageNet (1.3M) | LVD-142M (curated, 142M images) |
| Objective | Self-distillation only | Self-distillation + masked image modeling |
| Models | ViT-S/B | ViT-S/B/L/g (1.1B params) |
| Training | 300 epochs | 625K iterations on curated data |

#### Combined Loss

$$\mathcal{L}_{\text{DINOv2}} = \mathcal{L}_{\text{DINO}} + \lambda \cdot \mathcal{L}_{\text{iBOT}}$$

Where $\mathcal{L}_{\text{iBOT}}$ is a masked image modeling loss that predicts teacher patch tokens for masked positions — combining the strengths of self-distillation and MIM.

#### Data Curation Pipeline

LVD-142M was automatically curated from a larger uncurated pool:
1. Start from ImageNet as the seed
2. Use retrieval to find similar images from a large uncurated web dataset
3. Apply deduplication (copy detection, near-duplicate removal)
4. Balance the distribution across visual concepts

#### Performance — Universal Features

DINOv2 features work across tasks **without any fine-tuning** (just a linear probe or k-NN):

| Task | Dataset | DINOv2 ViT-g | Previous Best SSL |
|------|---------|-------------|-------------------|
| Classification | ImageNet-1k | 86.5% (linear) | 82.1% |
| Segmentation | ADE20K | 49.0 mIoU (linear) | 39.3 |
| Depth Estimation | NYU-Depth | 0.344 RMSE (linear) | 0.382 |
| Retrieval | Oxford/Paris | 85.9 mAP | 78.2 |

**DINOv2 features are truly general-purpose** — a single frozen model works well across classification, segmentation, depth estimation, and retrieval.

---

## Contrastive vs Masked: Comparison

| Aspect | Contrastive (SimCLR, MoCo) | Masked (MAE, BEiT) | Self-Distillation (DINO) |
|--------|---------------------------|--------------------|-----------------------|
| Pretext task | Positive/negative discrimination | Reconstruct masked input | Match teacher across views |
| Augmentation dependence | Very high (critical) | Low (just masking) | High (multi-crop) |
| Negative samples needed | Yes (or collapse avoidance) | No | No |
| Architecture | Any (CNN or ViT) | Primarily ViT | Primarily ViT |
| Linear probe quality | Good | Good | Excellent |
| Dense prediction transfer | Moderate | Good | Excellent |
| Training efficiency | Moderate | High (encoder sees 25%) | Moderate |
| Semantic emergence | Moderate | Low | High (attention maps) |

### When to Use What

| Scenario | Recommendation |
|----------|---------------|
| General-purpose features | DINOv2 (best linear probe across tasks) |
| Pre-training large ViTs | MAE (most efficient) |
| CNN backbone pre-training | MoCo v2 or BYOL |
| Small dataset fine-tuning | MAE or DINOv2 pre-trained models |
| Dense prediction (segmentation, depth) | DINOv2 (best transfer) |
| Domain-specific pre-training | MAE (simplest, works on any domain) |

---

## The Big Picture: SSL Timeline

| Year | Method | Paradigm | Key Contribution |
|------|--------|----------|-----------------|
| 2020 | SimCLR | Contrastive | Simple framework, strong augmentation |
| 2020 | MoCo | Contrastive | Momentum queue, small batch training |
| 2020 | BYOL | Self-distillation | No negatives needed |
| 2021 | DINO | Self-distillation | Emergent segmentation in ViT |
| 2022 | BEiT | Masked modeling | Visual tokens + masked prediction |
| 2022 | MAE | Masked modeling | 75% masking, pixel reconstruction |
| 2023 | DINOv2 | Hybrid | Universal visual features at scale |

The field has converged on **combining self-distillation with masked modeling** (as in DINOv2) as the most effective approach for learning universal visual representations.

---

## Key Takeaways

1. **Self-supervised learning** eliminates the labeling bottleneck by learning from the structure of raw images
2. **Contrastive methods** (SimCLR, MoCo) learn by pulling positive pairs together and pushing negatives apart — augmentation quality is critical
3. **MoCo's momentum queue** enables contrastive learning with small batches, making SSL practical on limited hardware
4. **Masked image modeling** (MAE) is the vision equivalent of BERT — mask patches, reconstruct them, learn structure
5. **75% masking** in MAE forces semantic understanding and makes training 3-4× more efficient
6. **DINO's self-distillation** produces features with emergent segmentation properties — attention heads learn object boundaries without any labels
7. **DINOv2** represents the current frontier — combining self-distillation and MIM at scale to produce truly universal visual features that work across tasks without fine-tuning

## References

- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. *CVPR 2020*. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
- Grill, J.-B., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. *NeurIPS 2020*. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV 2021*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- Bao, H., Dong, L., & Wei, F. (2022). BEiT: BERT Pre-Training of Image Transformers. *ICLR 2022*. [arXiv:2106.08254](https://arxiv.org/abs/2106.08254)
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *TMLR 2024*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
