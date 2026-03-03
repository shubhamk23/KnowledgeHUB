---
title: "Transformers Overview"
slug: transformers-overview
summary: "A comprehensive guide to the Transformer architecture — the foundation of modern NLP and LLMs."
tags: ["transformers", "attention", "nlp", "deep-learning"]
visibility: public
---

# Transformers Overview

The Transformer architecture, introduced in the landmark paper *"Attention Is All You Need"* (Vaswani et al., 2017), revolutionized NLP and became the foundation for all modern large language models.

## The Core Idea: Self-Attention

Unlike RNNs that process sequences step-by-step, Transformers process **all tokens in parallel** using the self-attention mechanism.

### Scaled Dot-Product Attention

Given queries $Q$, keys $K$, and values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $d_k$ is the dimension of the key vectors. The $\sqrt{d_k}$ scaling prevents the dot products from growing too large.

## Multi-Head Attention

Instead of a single attention, we run $h$ attention heads in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

**Why multiple heads?** Each head can attend to different aspects of the input simultaneously — one head might focus on syntactic relationships while another captures semantic similarity.

## Encoder-Decoder Architecture

```
Input → [Embedding + Positional Encoding]
     → [Multi-Head Attention → Add & Norm → FFN → Add & Norm] × N  (Encoder)
     → [Masked MHA → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm] × N  (Decoder)
     → Linear → Softmax → Output Probabilities
```

## Positional Encoding

Since attention is permutation-invariant, positional information must be injected explicitly:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

## Feed-Forward Network

Each position is processed independently through a two-layer FFN:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

The inner dimension is typically $4 \times d_{model}$.

## Key Variants

| Model | Architecture | Key Innovation |
|-------|-------------|----------------|
| BERT | Encoder-only | Bidirectional pre-training (MLM) |
| GPT | Decoder-only | Autoregressive language modeling |
| T5 | Encoder-Decoder | Text-to-text transfer learning |
| BART | Encoder-Decoder | Denoising auto-encoder |

## Code Example: Self-Attention

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights
```

## Key Takeaways

- Transformers replaced recurrence with **parallelizable self-attention**
- The $\sqrt{d_k}$ scaling stabilizes gradients during training
- Multi-head attention captures **diverse relationship types** simultaneously
- Positional encodings allow the model to reason about **token order**
- The encoder-decoder split is only needed for seq2seq tasks; BERT uses encoder-only, GPT uses decoder-only
