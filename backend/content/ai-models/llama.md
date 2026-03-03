---
title: "LLaMA: Open Foundation Language Models"
slug: llama
summary: "Meta's LLaMA series — architecture innovations including RoPE, SwiGLU, RMSNorm, grouped-query attention, and how they enabled open-source LLM development."
tags: ["LLaMA", "open-source", "RoPE", "SwiGLU", "RMSNorm", "GQA", "Meta", "foundation-model"]
visibility: public
---

# LLaMA: Open Foundation Language Models

## Overview

**LLaMA** (Large Language Model Meta AI) is Meta's series of open-weight foundation models that democratized access to capable LLMs. Unlike GPT-3/4 (API-only, closed weights), LLaMA models can be downloaded, fine-tuned, and deployed locally.

**Impact:** Spawned an ecosystem of open-source LLMs — Alpaca, Vicuna, Mistral, WizardLM, and many others are built on LLaMA foundations.

---

## LLaMA 1 (February 2023)

**Paper:** "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al.)

### Key Design Philosophy

**Main thesis:** A 13B model trained on more tokens can outperform a 175B model trained on fewer tokens.

Chinchilla showed that compute-optimal training uses ~20 tokens/parameter. LLaMA instead trains on **much more data** than Chinchilla-optimal — optimizing for **inference efficiency** rather than training efficiency.

### Model Sizes

| Model | Parameters | Training Tokens | Context | Layers | Heads |
|-------|-----------|-----------------|---------|--------|-------|
| LLaMA-7B | 7B | 1T | 2048 | 32 | 32 |
| LLaMA-13B | 13B | 1T | 2048 | 40 | 40 |
| LLaMA-33B | 33B | 1.4T | 2048 | 60 | 52 |
| LLaMA-65B | 65B | 1.4T | 2048 | 80 | 64 |

### Architecture Modifications from Original Transformer

**1. Pre-normalization (RMSNorm)**

Instead of LayerNorm applied after residual, apply RMSNorm **before** the sub-layer (pre-norm):

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2 + \epsilon}} \cdot \gamma$$

Advantages: More stable training; removes mean-centering for efficiency.

**2. SwiGLU Activation**

Replaces ReLU in the FFN with SwiGLU (Shazeer, 2020):

$$\text{FFN}_{\text{SwiGLU}}(x) = (\text{SiLU}(xW_1) \odot xW_3) W_2$$

Uses three weight matrices instead of two — increased parameters offset by using $\frac{2}{3} \times 4d$ hidden dim.

**3. Rotary Positional Embeddings (RoPE)**

Replaces absolute positional embeddings with rotary encoding applied to Q and K:

$$\mathbf{q}_m^\top \mathbf{k}_n = (\mathbf{R}_m \mathbf{q})^\top (\mathbf{R}_n \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{n-m} \mathbf{k}$$

Where $\mathbf{R}_m$ is a rotation matrix at position $m$.

**Advantages over sinusoidal PE:**
- Naturally encodes relative position
- Can generalize to longer sequences
- Better performance on downstream tasks

---

## LLaMA 2 (July 2023)

**Paper:** "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al.)

### Architecture Updates

**1. Grouped Query Attention (GQA)**

Standard multi-head attention: $H$ query heads, $H$ key heads, $H$ value heads.

**GQA:** $H$ query heads, but only $G$ groups of K/V heads ($G < H$):

```
Multi-Head (MHA):  Q1K1V1, Q2K2V2, Q3K3V3, Q4K4V4  (4 KV heads)
Grouped-Query (GQA): Q1Q2→K1V1, Q3Q4→K2V2           (2 KV heads)
Multi-Query (MQA):   Q1Q2Q3Q4→K1V1                   (1 KV head)
```

**Why it matters:** KV cache memory = $2 \times \text{seq\_len} \times H \times d_k \times \text{layers}$. GQA reduces KV cache by $H/G$ — critical for long contexts and batched inference.

**2. Longer Context:** 4096 tokens (vs 2048 in LLaMA 1)

**3. More Training Data:** 2T tokens (vs 1T)

### Chat Models (Llama 2-Chat)

Fine-tuned with RLHF:
1. SFT on high-quality conversation data
2. Reward model trained on human preference comparisons
3. RLHF with PPO (iteratively refined, 5 versions)
4. **Ghost Attention (GAtt):** Ensures the model remembers system prompt throughout long conversations

**GAtt Training:**
- Prepend system prompt to every human turn in training
- Forces model to condition on it throughout the dialogue
- Enables "persona" and constraint adherence

---

## LLaMA 3 / 3.1 (April / July 2024)

### Key Improvements

**Tokenizer:** 128K vocabulary (vs 32K) — better multilingual support, shorter token sequences for code.

**Architecture:**
- GQA throughout all sizes (7B also gets GQA in LLaMA 3 vs only 70B in LLaMA 2)
- Larger 8K context (native), extendable to 128K+ with RoPE scaling

**Training:**
- 15T tokens (vs 2T) — massive increase
- Higher quality data curation
- 50% code data

### Model Performance (LLaMA 3.1 405B)

Competitive with GPT-4 Turbo on many benchmarks:
- MMLU: 88.6
- HumanEval (code): 89.0
- MATH: 73.8
- GSM8K: 96.8

---

## Architecture Summary

```python
class LlamaAttention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(dim, n_heads * head_dim)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim)  # GQA
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim)  # GQA
        self.o_proj = nn.Linear(n_heads * head_dim, dim)
        self.rotary_emb = RotaryEmbedding(head_dim)  # RoPE

class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(dim, intermediate_size)  # SwiGLU gate
        self.up_proj = nn.Linear(dim, intermediate_size)    # SwiGLU up
        self.down_proj = nn.Linear(intermediate_size, dim)  # SwiGLU down

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def forward(self, x):
        # Pre-norm
        h = x + self.self_attn(self.input_layernorm(x))  # RMSNorm before attn
        out = h + self.mlp(self.post_attention_layernorm(h))  # RMSNorm before FFN
        return out
```

---

## Ecosystem Built on LLaMA

| Model | Base | Innovation |
|-------|------|-----------|
| Alpaca | LLaMA 1-7B | Instruction tuning (52K examples, $600) |
| Vicuna | LLaMA 13B | ChatGPT conversation fine-tuning |
| Mistral 7B | — | Sliding Window Attention, better efficiency |
| WizardLM | LLaMA | Evol-Instruct data evolution |
| Llama-Guard | LLaMA | Safety classifier |
| Code Llama | LLaMA 2 | Code-focused fine-tuning |

---

## Running LLaMA Locally

```bash
# llama.cpp (CPU inference)
./llama-cli -m llama-3.1-8b-q4_k_m.gguf -p "What is quantum computing?"

# Ollama (easy local deployment)
ollama pull llama3.1
ollama run llama3.1 "Explain transformer architecture"
```

**Quantization for local use:**
- Q4_K_M: 4.5GB for 8B model, good quality
- Q8_0: 8.5GB for 8B model, near-lossless

---

## Key Takeaways

1. **LLaMA's insight:** Train smaller models on much more data → better inference efficiency
2. **Key architecture innovations:** RoPE (position), RMSNorm (norm), SwiGLU (FFN), GQA (KV cache)
3. **GQA reduces KV cache** memory significantly — critical for long contexts and batching
4. **LLaMA 2-Chat** used iterative RLHF (5 versions) + Ghost Attention for persona consistency
5. **LLaMA 3.1 405B** is competitive with frontier closed models on many benchmarks
6. **Open weights = ecosystem:** Alpaca, Vicuna, Mistral, Code Llama all build on LLaMA

## References

- Touvron et al. (2023) — LLaMA: Open and Efficient Foundation Language Models
- Touvron et al. (2023) — LLaMA 2: Open Foundation and Fine-Tuned Chat Models
- Meta AI (2024) — LLaMA 3: Meta's Most Advanced Open Source Model
- Ainslie et al. (2023) — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
