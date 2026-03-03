---
title: "Multimodal AI: Introduction"
slug: multimodal-intro
summary: "How modern AI systems combine vision, language, and audio into unified models that understand and generate across modalities."
tags: ["multimodal", "vision-language", "LLaVA", "GPT-4V", "Gemini", "cross-modal"]
visibility: public
---

# Multimodal AI: Introduction

## What is Multimodal AI?

**Multimodal AI** systems process and reason over multiple types of data simultaneously — text, images, audio, video, and structured data. Unlike unimodal models (text-only GPT, vision-only ViT), multimodal models can:

- **Understand** cross-modal relationships (describe what's in an image)
- **Generate** cross-modal outputs (create an image from text)
- **Reason** across modalities (answer questions about charts)

**Why multimodal?** Human perception is inherently multimodal — we describe what we see, identify sounds, and interpret diagrams. Truly intelligent systems must handle this complexity.

---

## Core Modality Combinations

| Combination | Task | Example Models |
|-------------|------|---------------|
| Text + Image (input) | Visual QA, image captioning | LLaVA, GPT-4V, Gemini |
| Text → Image (output) | Image generation | DALL-E, Stable Diffusion |
| Text + Audio (input) | Speech understanding | Whisper + LLM |
| Text + Video | Video QA, captioning | Video-LLaMA, Gemini |
| Any → Any | Universal understanding | GPT-4o, Gemini Ultra |

---

## Vision-Language Models (VLMs)

The most developed multimodal paradigm: combine image encoder + language model.

### Architecture Pattern

```
Image → [Vision Encoder] → visual tokens
                                ↓
                    [Projection / Adapter]
                                ↓
Text  → [Tokenizer] → text tokens + visual tokens → [LLM Decoder] → Response
```

### LLaVA (Large Language and Vision Assistant)

**Paper:** "Visual Instruction Tuning" (Liu et al., 2023)

Architecture:
1. **Vision encoder:** CLIP ViT-L/14 (frozen)
2. **Projection:** Linear layer mapping visual features to LLM input space
3. **LLM:** LLaMA / Vicuna

Training stages:
1. **Stage 1 (pre-training):** Train only the projection layer on 595K image-text pairs (CC3M)
2. **Stage 2 (instruction tuning):** Fine-tune projection + LLM on 158K instruction-following data (GPT-4 generated)

**LLaVA 1.5:** MLP projection (vs single linear), higher resolution, stronger LLM → state-of-art in open-source VLMs

```python
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Process image + text
inputs = processor(text=prompt, images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
```

### BLIP-2 (Bootstrapping Language-Image Pre-training)

**Paper:** "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al., 2023)

Key innovation: **Q-Former** (Querying Transformer) bridges the modality gap:
1. Frozen image encoder → visual features
2. **Q-Former:** 32 learnable query tokens extract relevant info from visual features (cross-attention)
3. Query token output → frozen LLM input

**Why Q-Former?** Forces compression of visual information into a small, LLM-compatible representation. More efficient than raw visual tokens.

Training stages:
1. Train Q-Former with frozen vision encoder (image-text matching, generation, contrastive)
2. Connect to frozen LLM with a projection layer

### Flamingo (DeepMind)

**Architecture:**
- Interleaves visual information into LLM via gated cross-attention layers
- Image features injected at multiple LLM layers (vs only at input for LLaVA)
- Supports interleaved image-text sequences

```
[Text Token] → LLM Layer → [Gated X-Attn with Image Features] → LLM Layer → ...
```

**Training:** 80B image-text pairs, 185M video-text pairs — massive scale.
**Result:** Strong few-shot performance on VQA, captioning, classification.

---

## Frontier Multimodal Models

### GPT-4V / GPT-4o (OpenAI)

- **GPT-4V:** Vision understanding integrated into GPT-4
- **GPT-4o:** "Omni" — processes text, image, audio natively in one model
- Capabilities: OCR, diagram understanding, code from screenshots, chart analysis

### Gemini (Google)

- **Architecture:** Natively multimodal from pre-training (not post-hoc added vision)
- **Gemini Ultra:** First model to surpass GPT-4 on MMLU (massive multitask language understanding)
- Supports text, images, audio, video, code in any combination
- **1M context window** (Gemini 1.5 Pro) — entire movies or codebases

### Claude 3 (Anthropic)

- Haiku / Sonnet / Opus with vision capabilities
- Strong at document understanding, chart analysis, scientific diagrams
- Context: 200K tokens

---

## Pre-training Strategies

### Contrastive Pre-training (CLIP paradigm)

Match image and text representations — covered in detail in the CLIP note.

### Generative Pre-training

Train to generate text given images:

$$\mathcal{L} = -\sum_t \log p_\theta(w_t | w_{<t}, \text{image})$$

### Masked Image Modeling + Text

Extend BERT's MLM to joint image-text masking.

**FLAVA:** Jointly train image, text, and multimodal encoders with separate objectives.

---

## Datasets

| Dataset | Type | Size | Usage |
|---------|------|------|-------|
| LAION-5B | Image-text pairs | 5B | Pre-training |
| CC3M / CC12M | Image captions | 3M/12M | Pre-training |
| VQA v2 | Visual QA | 1.1M QA pairs | Fine-tuning |
| GQA | Compositional VQA | 22M QA pairs | Fine-tuning |
| LLaVA-Instruct | Instruction following | 158K | Instruction tuning |
| TextVQA | OCR + VQA | 45K | Evaluation |

---

## Evaluation Benchmarks

| Benchmark | Measures |
|-----------|---------|
| VQA v2 | Visual question answering |
| MMBENCH | Comprehensive VLM evaluation |
| MME | Perception + cognition tasks |
| MMMU | Massive Multidisciplinary Multimodal Understanding |
| TextVQA | OCR + reasoning |
| ScienceQA | Science multimodal reasoning |

---

## Key Takeaways

1. **VLMs = Vision Encoder + Adapter + LLM** — CLIP + projection + LLaMA is LLaVA
2. **Q-Former** (BLIP-2) elegantly bridges modalities through learned query tokens
3. **Natively multimodal pre-training** (Gemini) outperforms post-hoc integration
4. **Instruction tuning** is crucial — raw VLMs need fine-tuning to follow instructions
5. **GPT-4o** takes this further with native audio-image-text processing
6. **Open-source VLMs** (LLaVA 1.6, InternVL) are rapidly approaching closed-model performance

## References

- Liu et al. (2023) — Visual Instruction Tuning (LLaVA)
- Li et al. (2023) — BLIP-2
- Alayrac et al. (2022) — Flamingo
- Radford et al. (2021) — CLIP
- Team et al. (2023) — Gemini: A Family of Highly Capable Multimodal Models
