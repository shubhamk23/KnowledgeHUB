---
title: "Transfer Learning for Computer Vision"
slug: transfer-learning-cv
summary: "A practical guide to transfer learning — pretrained models, fine-tuning strategies, feature extraction, and domain adaptation for vision tasks."
tags: ["transfer-learning", "fine-tuning", "pretrained-models", "domain-adaptation", "computer-vision", "deep-learning"]
visibility: public
level: beginner
---

# Transfer Learning for Computer Vision

## Why Transfer Learning?

Training a deep CNN from scratch requires millions of labeled images and days of GPU time. **Transfer learning** reuses knowledge learned from one task (usually ImageNet classification) to accelerate learning on a new task.

| Approach | Training Data Needed | Training Time | Accuracy |
|----------|---------------------|---------------|----------|
| From scratch | Millions of images | Days | Baseline |
| Feature extraction | Hundreds of images | Minutes | Good |
| Fine-tuning | Thousands of images | Hours | Excellent |

**Key insight:** Early CNN layers learn universal features (edges, textures, colors) that transfer well across tasks. Later layers learn task-specific features.

---

## How Transfer Learning Works

### Feature Hierarchy in CNNs

```
Layer 1-2:  Edges, corners, color gradients     ← Universal (transfers well)
Layer 3-5:  Textures, patterns, simple shapes    ← Mostly universal
Layer 6-10: Object parts (eyes, wheels, windows) ← Somewhat task-specific
Layer 11+:  Full objects, scene categories        ← Task-specific
```

The deeper you go, the more task-specific features become. Transfer learning leverages this hierarchy by keeping universal features and adapting task-specific ones.

### Two Main Strategies

**1. Feature Extraction (Freeze backbone)**

Use the pretrained network as a fixed feature extractor:

```
Pretrained CNN (frozen) → Feature vector → New classifier head
```

- Replace the final classification layer with a new one matching your number of classes
- Freeze all backbone weights — only train the new head
- Fast training, works with very small datasets (even 100 images per class)

**2. Fine-Tuning (Unfreeze some layers)**

Update some or all of the pretrained weights on your new data:

```
Pretrained CNN (partially unfrozen) → New classifier head
```

- Start with a pretrained backbone
- Replace the final layers
- Gradually unfreeze layers from top to bottom
- Use a smaller learning rate than training from scratch

---

## Fine-Tuning Strategies

### Learning Rate Selection

The most important hyperparameter in fine-tuning:

| Strategy | Learning Rate | When to Use |
|----------|--------------|-------------|
| Feature extraction | 1e-3 to 1e-2 (head only) | Very small dataset, similar domain |
| Conservative fine-tune | 1e-5 to 1e-4 (all layers) | Small dataset, similar domain |
| Discriminative LR | 1e-5 (early) → 1e-3 (late) | Medium dataset |
| Full fine-tune | 1e-4 to 1e-3 | Large dataset, different domain |

### Discriminative Learning Rates

Use different learning rates for different layer groups — lower rates for early (universal) layers, higher rates for later (task-specific) layers:

```
Layer group 1 (early):  lr = 1e-5   ← Preserve universal features
Layer group 2 (middle): lr = 1e-4   ← Gentle adaptation
Layer group 3 (late):   lr = 1e-3   ← More aggressive adaptation
New head:               lr = 1e-2   ← Learn from scratch
```

### Gradual Unfreezing

A popular approach that reduces the risk of catastrophic forgetting:

1. **Epoch 1-3:** Train only the new head (backbone frozen)
2. **Epoch 4-6:** Unfreeze the last backbone block, train with small LR
3. **Epoch 7-10:** Unfreeze more blocks progressively
4. **Epoch 11+:** All layers unfrozen with discriminative LR

This stabilizes training — the head learns a reasonable mapping before backbone weights are adjusted.

---

## Choosing a Pretrained Model

### Popular Backbones for Transfer Learning

| Model | Params | ImageNet Top-1 | Best For |
|-------|--------|---------------|----------|
| ResNet-50 | 25.6M | 76.0% | General purpose, well-understood |
| EfficientNet-B0 | 5.3M | 77.3% | Mobile/edge deployment |
| EfficientNet-B4 | 19M | 82.9% | Good accuracy-efficiency balance |
| ConvNeXt-T | 29M | 82.1% | Modern CNN, Transformer-like performance |
| ViT-B/16 | 86M | 81.8% | Large datasets, research |
| Swin-T | 29M | 81.3% | Dense prediction (detection, segmentation) |
| DINOv2 ViT-B | 86M | 84.5% | Self-supervised, universal features |

### Decision Guide

```
Dataset size < 1K images?
  → Feature extraction with ResNet-50 or EfficientNet

Dataset size 1K–10K images?
  → Fine-tune EfficientNet or ConvNeXt with gradual unfreezing

Dataset size 10K–100K images?
  → Full fine-tune with discriminative LR

Dataset size > 100K images?
  → Consider training from scratch (or fine-tune large ViT)

Domain very different from ImageNet (medical, satellite)?
  → Use domain-specific pretrained models if available
  → Otherwise, fine-tune with aggressive augmentation
```

---

## Domain Adaptation

When your target domain is very different from the source domain (ImageNet):

### Domain Gap Examples

| Source (ImageNet) | Target Domain | Gap |
|------------------|---------------|-----|
| Natural photos | Medical X-rays | Very large |
| Natural photos | Satellite imagery | Large |
| Natural photos | Art/paintings | Medium |
| Natural photos | Product photos | Small |

### Strategies for Large Domain Gaps

1. **Intermediate pre-training:** Fine-tune on a related intermediate dataset before the target task
2. **Domain-specific models:** Use models pre-trained on domain data (e.g., CheXNet for chest X-rays, SatMAE for satellite)
3. **Self-supervised pre-training:** Pre-train with MAE or DINO on unlabeled target domain data, then fine-tune
4. **Strong augmentation:** More aggressive augmentation helps bridge domain gaps

### Unsupervised Domain Adaptation (UDA)

When you have labeled source data but only unlabeled target data:

$$\mathcal{L} = \mathcal{L}_{\text{task}}(\text{source}) + \lambda \cdot \mathcal{L}_{\text{domain}}(\text{source, target})$$

The domain loss encourages the model to learn domain-invariant features. Common approaches:
- **Domain adversarial training:** A domain classifier tries to distinguish source from target; the feature extractor tries to fool it
- **Self-training:** Use confident predictions on target data as pseudo-labels
- **Style transfer:** Transform source images to look like target domain

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Learning rate too high | Start with 1e-5, increase gradually |
| Forgetting pretrained features | Freeze backbone first, unfreeze gradually |
| Not enough augmentation | Use RandAugment, Mixup, CutMix |
| Wrong input preprocessing | Match the pretrained model's normalization (ImageNet mean/std) |
| Too few epochs | Fine-tuning often needs 20-50 epochs with cosine LR |
| Overfitting on small data | Use dropout, weight decay, early stopping |

### Input Preprocessing

Always match the preprocessing used during pre-training:

```
ImageNet normalization:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]

Input size: Match the pretrained model's expected size
  ResNet: 224×224
  EfficientNet-B0: 224×224
  EfficientNet-B4: 380×380
  ViT-B/16: 224×224 (can interpolate positional embeddings for other sizes)
```

---

## Practical Training Pipeline

A step-by-step approach for fine-tuning on a new vision task:

1. **Prepare data:** Split into train/val/test, apply augmentation to train set
2. **Load pretrained model:** Download weights from torchvision, timm, or Hugging Face
3. **Replace the head:** New linear layer matching your number of classes
4. **Phase 1 (5 epochs):** Train head only, LR = 1e-3, cosine schedule
5. **Phase 2 (15-30 epochs):** Unfreeze all, discriminative LR (1e-5 backbone, 1e-3 head)
6. **Evaluate:** Check validation accuracy, confusion matrix, per-class metrics
7. **Export:** Save to ONNX or TorchScript for deployment

### Key Hyperparameters

| Parameter | Recommended Value |
|-----------|--------------------|
| Optimizer | AdamW (weight_decay=0.01) |
| LR schedule | Cosine annealing with warmup |
| Batch size | 32-64 (larger if GPU memory allows) |
| Augmentation | RandAugment(n=2, m=9) + Mixup(α=0.2) |
| Label smoothing | 0.1 |
| Early stopping | Patience of 5-10 epochs |

---

## Key Takeaways

1. **Transfer learning** is the default approach for vision tasks — training from scratch is rarely needed
2. **Feature extraction** (frozen backbone) works well with very small datasets; **fine-tuning** is better with more data
3. **Discriminative learning rates** — lower for early layers, higher for later layers — prevent catastrophic forgetting
4. **Gradual unfreezing** stabilizes training by letting the new head learn before adjusting backbone weights
5. **Match preprocessing:** Always normalize inputs with the same mean/std used during pre-training
6. For large domain gaps, consider **domain-specific pretrained models** or **self-supervised pre-training** on target data
7. **Modern self-supervised models** (DINOv2, MAE) often transfer better than supervised ImageNet models

## References

- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How transferable are features in deep neural networks? *NeurIPS 2014*. [arXiv:1411.1792](https://arxiv.org/abs/1411.1792)
- Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL 2018*. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do Better ImageNet Models Transfer Better? *CVPR 2019*. [arXiv:1805.08974](https://arxiv.org/abs/1805.08974)
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR 2022*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *TMLR 2024*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
