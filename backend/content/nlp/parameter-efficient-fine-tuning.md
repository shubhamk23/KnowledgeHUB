---
title: "Parameter-Efficient Fine-Tuning (PEFT)"
slug: parameter-efficient-fine-tuning
summary: "LoRA, QLoRA, adapters, and prefix tuning — how to fine-tune billion-parameter LLMs without updating all weights."
tags: ["LoRA", "QLoRA", "PEFT", "fine-tuning", "adapters", "LLM"]
visibility: public
---

# Parameter-Efficient Fine-Tuning (PEFT)

## Overview

Full fine-tuning of large language models (GPT-3: 175B params, LLaMA-2: 70B params) requires enormous GPU memory and compute. **Parameter-Efficient Fine-Tuning (PEFT)** methods adapt pre-trained models to downstream tasks by updating only a small fraction of parameters while keeping most weights frozen.

**Goal:** Match or approach full fine-tuning performance while training <1% of model parameters.

---

## Why PEFT?

| Approach | Trainable Params | GPU Memory | Performance |
|----------|-----------------|------------|-------------|
| Full fine-tuning | 100% | Very high (multiple A100s) | Baseline |
| PEFT (LoRA) | 0.1–1% | ~1 A100 or less | Near full FT |
| Prefix tuning | ~0.1% | Low | Good |
| Adapters | 0.5–5% | Moderate | Good |

**Additional benefits:**
- Multiple task-specific adapters can share one base model (storage efficient)
- Reduced catastrophic forgetting of pre-training knowledge
- Faster training and lower cost

---

## LoRA: Low-Rank Adaptation

**Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

### Core Idea

Weight matrices in Transformers have low intrinsic rank during adaptation. Instead of updating $W \in \mathbb{R}^{d \times d}$, decompose the update into two low-rank matrices:

$$W' = W_0 + \Delta W = W_0 + BA$$

Where:
- $W_0 \in \mathbb{R}^{d \times d}$ — frozen pre-trained weights
- $B \in \mathbb{R}^{d \times r}$ — trainable
- $A \in \mathbb{R}^{r \times d}$ — trainable
- $r \ll d$ — rank (typically 4, 8, or 16)

### Forward Pass

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

During inference, merge back: $W' = W_0 + BA$ (no overhead).

### Parameter Comparison

For $d = 4096$, $r = 8$:
- Full matrix: $4096^2 = 16.7M$ params
- LoRA: $2 \times 4096 \times 8 = 65K$ params (**256× reduction**)

### Initialization

- $A$ initialized with random Gaussian
- $B$ initialized to zero (so $\Delta W = 0$ at start, preserving pre-trained behavior)

### Where to Apply LoRA

Typically applied to attention weight matrices:
- Query ($W_Q$) and Value ($W_V$) projections (most impactful)
- Key ($W_K$) and Output ($W_O$) projections (optional)
- Feed-forward layers (sometimes)

```python
# HuggingFace PEFT example
from peft import get_peft_model, LoraConfig, TaskType

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                     # rank
    lora_alpha=32,           # scaling factor (alpha/r)
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622%
```

---

## QLoRA: Quantized LoRA

**Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)

Enables fine-tuning 65B parameter models on a single 48GB GPU.

### Key Innovations

**1. 4-bit NormalFloat (NF4) quantization:**
Quantizes base model weights to 4 bits using a data type optimized for normally distributed weights:

$$W_{\text{int4}} = \text{quantize}(W_{\text{fp16}})$$

During forward pass, weights are dequantized on-the-fly to compute activations.

**2. Double Quantization:**
Quantizes the quantization constants themselves, saving ~0.37 bits per parameter.

**3. Paged Optimizers:**
Uses NVIDIA unified memory to prevent OOM errors during gradient checkpointing.

### Memory Comparison (65B model)

| Method | GPU Memory Required |
|--------|-------------------|
| Full fine-tuning (fp16) | >780 GB |
| LoRA (fp16) | ~160 GB |
| QLoRA (4-bit) | **~48 GB** |

---

## Adapters

**Paper:** "Parameter-Efficient Transfer Learning for NLP" (Houlsby et al., 2019)

### Architecture

Insert small trainable modules inside each Transformer layer:

```
Original: Attention → LayerNorm → FFN → LayerNorm
Adapter:  Attention → Adapter → LayerNorm → FFN → Adapter → LayerNorm
```

Each adapter module:
1. Down-projection: $h_d = W_{\text{down}} x$ (reduces dimension from $d$ to $m$)
2. Non-linearity: $h = \text{ReLU}(h_d)$
3. Up-projection: $h_u = W_{\text{up}} h$ (restores dimension)
4. Residual connection: output = $x + h_u$

### Variants

- **Bottleneck adapters:** Original design, ~3.6% overhead
- **Parallel adapters:** Adapters in parallel with attention/FFN (faster)
- **AdapterFusion:** Learns to combine multiple adapters for different tasks

---

## Prefix Tuning

**Paper:** "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (Li & Liang, 2021)

### Core Idea

Prepend trainable continuous vectors (the "prefix") to the keys and values of every attention layer:

$$\text{Attention}(Q, [P_K; K], [P_V; V])$$

Where $P_K$ and $P_V$ are learnable prefix matrices prepended to K and V.

**Only the prefix parameters are trained** (~0.1% of model params). Base model remains frozen.

### vs Prompt Tuning

| Method | Level | Where Learned |
|--------|-------|---------------|
| Prefix Tuning | Every layer | K/V at each attention layer |
| Prompt Tuning | Input only | Soft tokens at input |
| In-context learning | None | No training, just examples |

---

## Prompt Tuning (Soft Prompts)

**Paper:** "The Power of Scale for Parameter-Efficient Prompt Tuning" (Lester et al., 2021)

### Core Idea

Learn continuous "soft prompt" vectors prepended to the input embeddings:

$$\text{Input} = [\theta_1, \ldots, \theta_k; e_1, \ldots, e_n]$$

Where $\theta_i$ are trainable soft tokens, $e_i$ are frozen input embeddings.

**Only the soft prompt embeddings are trained** — as few as 20–100 tokens.

**Key finding:** At scale (11B+ params), prompt tuning approaches full fine-tuning performance.

---

## IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

**Paper:** "(IA)³: Few-Shot Parameter-Efficient Fine-Tuning" (Liu et al., 2022)

Rescales activations in attention and FFN layers with learned vectors:

$$h' = l \odot h$$

Where $l$ is a learnable vector (same dimension as $h$). Only adds ~0.01% trainable parameters.

---

## Comparison Table

| Method | Trainable % | Memory | Inference Overhead | Best For |
|--------|------------|--------|-------------------|----------|
| Full FT | 100% | Very high | None (after merge) | Unlimited compute |
| LoRA | 0.1–1% | Low | None (merge at test) | General purpose |
| QLoRA | 0.1–1% | Very low | Dequantization | Consumer GPUs |
| Adapters | 0.5–5% | Moderate | Small overhead | Multi-task |
| Prefix Tuning | ~0.1% | Low | KV prefix overhead | Generation tasks |
| Prompt Tuning | ~0.01% | Very low | Small overhead | Large models |

---

## Practical Tips

```python
# LoRA rank selection
# Lower r (4-8): faster, less capacity, works well for most tasks
# Higher r (16-64): more capacity, approaches full FT, more memory

# Alpha scaling
# lora_alpha = 2*r is a common default (effective learning rate)
# lora_alpha = r is more conservative

# Merge LoRA weights for zero-overhead inference
merged_model = model.merge_and_unload()
```

---

## Key Takeaways

1. **LoRA** decomposes weight updates into low-rank matrices — dramatic parameter reduction with near-full-FT performance
2. **QLoRA** adds 4-bit quantization enabling billion-parameter fine-tuning on consumer hardware
3. **Adapters** insert small bottleneck modules — useful for multi-task settings
4. **Prefix/Prompt tuning** learn continuous prompt vectors — minimal compute but limited expressivity
5. **Rule of thumb:** Start with LoRA (r=8, target Q+V) for most tasks; use QLoRA if memory-constrained

## References

- Hu et al. (2021) — LoRA: Low-Rank Adaptation of Large Language Models
- Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs
- Houlsby et al. (2019) — Parameter-Efficient Transfer Learning for NLP
- Li & Liang (2021) — Prefix-Tuning: Optimizing Continuous Prompts for Generation
- Lester et al. (2021) — The Power of Scale for Parameter-Efficient Prompt Tuning
