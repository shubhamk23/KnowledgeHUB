---
title: "CLIP: Contrastive Language-Image Pre-training"
slug: clip
summary: "How OpenAI's CLIP learns visual representations from natural language supervision and enables zero-shot image classification."
tags: ["CLIP", "vision-language", "contrastive", "zero-shot", "multimodal", "embeddings"]
visibility: public
---

# CLIP: Contrastive Language-Image Pre-training

## Overview

**CLIP** (Radford et al., OpenAI, 2021) learns visual representations by training on **400 million image-text pairs** scraped from the internet using a contrastive objective. The resulting model can transfer to downstream vision tasks **zero-shot** — without any task-specific training.

**Key paper:** "Learning Transferable Visual Models From Natural Language Supervision"

---

## Architecture

```
Images ──→ [Image Encoder] ──→ image embeddings (normalized)
                                        ↕ cosine similarity matrix
Text   ──→ [Text Encoder]  ──→ text  embeddings (normalized)
```

**Image Encoder:** ViT (Vision Transformer) or ResNet
**Text Encoder:** Transformer (similar to GPT-2)

Both encoders project to the same embedding space of dimension $d$ (typically 512 or 1024).

### Linear Projection

After encoding, project to a shared embedding space:
- $\mathbf{I}_i = W_I \cdot \text{ImageEncoder}(x_i^I) / \|W_I \cdot \text{ImageEncoder}(x_i^I)\|_2$
- $\mathbf{T}_i = W_T \cdot \text{TextEncoder}(x_i^T) / \|W_T \cdot \text{TextEncoder}(x_i^T)\|_2$

Both $\mathbf{I}_i, \mathbf{T}_i \in \mathbb{R}^d$, unit-normalized.

---

## Training Objective

For a batch of $N$ (image, text) pairs, CLIP optimizes contrastive loss:

**Similarity matrix:**
$$S_{ij} = \mathbf{I}_i \cdot \mathbf{T}_j \cdot \exp(\tau)$$

Where $\tau$ is a learned temperature parameter.

**Symmetric cross-entropy loss:**

$$\mathcal{L}_I = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{S_{ii}}}{\sum_{j=1}^N e^{S_{ij}}}$$

$$\mathcal{L}_T = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{S_{ii}}}{\sum_{j=1}^N e^{S_{ji}}}$$

$$\mathcal{L} = \frac{\mathcal{L}_I + \mathcal{L}_T}{2}$$

**Intuition:** Each image $\mathbf{I}_i$ should have maximum similarity with its paired text $\mathbf{T}_i$ and minimum similarity with all other texts in the batch.

With $N = 32768$ (large batch), the model learns to distinguish 32,768 negatives per positive — a very challenging task that forces rich representations.

---

## Zero-Shot Image Classification

CLIP's most remarkable ability — classify images into arbitrary categories without any task-specific training:

### How it Works

1. **Convert labels to text templates:**
```python
labels = ["cat", "dog", "bird"]
templates = [f"a photo of a {label}" for label in labels]
# → ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
```

2. **Encode all templates:**
```python
text_embeddings = text_encoder(templates)  # [num_classes, d]
```

3. **Encode query image:**
```python
image_embedding = image_encoder(image)  # [d]
```

4. **Compute similarities and argmax:**
```python
similarities = image_embedding @ text_embeddings.T  # [num_classes]
predicted_class = similarities.argmax()
```

### Performance

On ImageNet (1000-class classification):
- CLIP ViT-L/14: **76.2% zero-shot** accuracy
- Context: ResNet-50 supervised = 76.1% (but ResNet was explicitly trained on ImageNet)

Zero-shot CLIP ≈ supervised ResNet-50 on ImageNet — remarkable generalization.

---

## Prompt Engineering for CLIP

The text template significantly affects zero-shot performance:

| Template | Accuracy |
|---------|---------|
| `label` | 63.3% |
| `a photo of {label}` | 76.2% |
| `a photo of a {label}, a type of food` (with context) | Higher for food tasks |

**Ensemble of prompts** (average 80 templates):
```python
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "a rendering of a {}",
    "a bad photo of a {}",
    "a close-up photo of a {}",
    # ... 75 more
]

# Average text embeddings across templates
text_emb = mean([text_encoder(t.format(label)) for t in templates])
```

Ensembling improves accuracy by ~3-5%.

---

## Applications

### 1. Image-Text Retrieval

Find images matching a text query (or vice versa):
```python
# Given text query: "a dog playing in snow"
text_emb = text_encoder("a dog playing in snow")

# Rank all images by similarity
similarities = text_emb @ all_image_embeddings.T
top_k_images = images[similarities.topk(k).indices]
```

### 2. Content Moderation

```python
labels = ["safe content", "explicit content", "violence", "hate symbols"]
# Classify any image without task-specific training
```

### 3. Visual Search

Product search, similar image retrieval — CLIP embeddings work as universal visual features.

### 4. Foundation for Generative Models

CLIP's text encoder is used in:
- **Stable Diffusion** — CLIP text encoder guides image generation
- **DALL-E 2** — CLIP prior maps text to image embedding space
- **MidJourney** — CLIP-guided generation

```python
# Stable Diffusion's text conditioning
text_embedding = clip_text_encoder(prompt)
# Used to condition the UNet denoiser at every step
image = unet_denoise(noise, conditioning=text_embedding)
```

---

## CLIP Variants & Successors

| Model | Innovation |
|-------|-----------|
| **OpenCLIP** | Open-source reimplementation of CLIP |
| **ALIGN** (Google) | 1.8B noisy image-text pairs, larger scale |
| **Florence** (Microsoft) | Hierarchical pretraining with multiple grains |
| **SigLIP** (Google) | Sigmoid loss instead of softmax; better for small batches |
| **BLIP-2** | Adds Q-Former to bridge CLIP image encoder with LLM |
| **LLaVA** | Connects CLIP encoder to LLaMA with projection layer |

### SigLIP Loss

Binary cross-entropy instead of contrastive softmax — no normalization across batch:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i,j} \log \sigma(z_{ij} \cdot y_{ij})$$

Where $y_{ij} = +1$ for matched pairs, $-1$ for unmatched.

**Advantages:** Works better with smaller batches; faster convergence.

---

## Limitations of CLIP

| Limitation | Details |
|------------|---------|
| **Fine-grained tasks** | Struggles with counting, spatial relationships |
| **OCR** | Poor at reading text in images |
| **OOD generalization** | Fails on unusual image-text combinations |
| **Bias** | Inherits dataset biases (internet text + images) |
| **Abstract concepts** | Better with concrete than abstract descriptions |

---

## Key Takeaways

1. **CLIP bridges vision and language** — trains on 400M image-text pairs with contrastive learning
2. **Zero-shot classification** without task-specific training — remarkable generalization
3. **Symmetric InfoNCE loss** with large batches forces rich semantic alignment
4. **Prompt engineering** significantly affects performance — templates and ensembling help
5. **Foundation for generative AI** — CLIP encoder is used in Stable Diffusion, DALL-E 2
6. **SigLIP** improves on CLIP with sigmoid loss, especially for smaller batches

## References

- Radford et al. (2021) — Learning Transferable Visual Models From Natural Language Supervision (CLIP)
- Zhai et al. (2023) — Sigmoid Loss for Language Image Pre-Training (SigLIP)
- Li et al. (2022) — BLIP-2
- Liu et al. (2023) — LLaVA: Visual Instruction Tuning
