---
title: "Attention Mechanism"
slug: attention
summary: "From the seq2seq bottleneck to multi-head self-attention — a deep dive into how attention lets models dynamically focus on relevant input."
tags: ["attention", "seq2seq", "self-attention", "transformer", "multi-head"]
visibility: public
---

# Attention Mechanism

## Overview

The attention mechanism addresses the **bottleneck problem** in sequence-to-sequence models — fixed-size context vectors struggle with lengthy sequences. Instead of compressing all input information into one vector, attention enables models to dynamically focus on relevant input portions during each decoding step.

> "Attention allows the model to focus on the relevant parts of the input sequence as needed."

**Foundational paper:** "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., ICLR 2015)

---

## The Classic Seq2Seq Bottleneck

The original seq2seq model (Cho et al., 2014) uses an encoder-decoder architecture:

**Encoder** reads input $x = (x_1, \ldots, x_T)$ into a fixed-length vector $c$:

$$h_t = f(x_t, h_{t-1})$$
$$c = q(h_1, \ldots, h_T)$$

**Decoder** predicts each output word given $c$ and previous words:

$$p(y) = \prod_{i=1}^{T} p(y_t \mid y_1, \ldots, y_{t-1}, c)$$

**Problem:** All source information is compressed into a single vector $c$. Performance degrades significantly on long sentences.

---

## Attention: The Fix

Instead of a single context vector, attention computes a **distinct context vector per output step** using all encoder hidden states.

### Bidirectional Encoder
Uses forward and backward RNNs. The annotation for word $j$ is:

$$h_j = [\overrightarrow{h}_j^T; \overleftarrow{h}_j^T]^T$$

### Context Vector

$$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$$

The alignment weights are computed via softmax over alignment scores:

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

The alignment score (Bahdanau et al.):

$$e_{ij} = a(s_{i-1}, h_j) = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$$

where $v_a$, $W_a$, $U_a$ are learnable weight matrices.

---

## Query, Key, Value Framework

Modern attention uses the **Q-K-V paradigm**:

> "You have a specific question (query). Books on shelves have titles (keys) suggesting content. You compare your question to titles to decide relevance, then retrieve information (value) from relevant books."

Three matrices trained during training multiply input embeddings to produce $K$, $V$, $Q$ vectors. In Transformers, these have dimension 64 while embeddings maintain dimension 512.

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The $\sqrt{d_k}$ scaling prevents dot products from growing too large, which would push softmax into regions of very small gradients.

### Step-by-Step Calculation

1. **Create vectors:** Generate $Q$, $K$, $V$ from input embeddings
2. **Score:** Compute dot product $q_1 \cdot k_1$, $q_1 \cdot k_2$, etc.
3. **Scale:** Divide by $\sqrt{d_k} = \sqrt{64} = 8$
4. **Normalize:** Apply softmax to get attention weights
5. **Weight values:** Multiply each value vector by its softmax score
6. **Sum:** Output is the weighted sum of value vectors

---

## Self-Attention

In self-attention, all $Q$, $K$, $V$ originate from the **same input sequence**. This allows each word to attend to all other words:

- Interaction strength between key-query pairs varies by content
- Captures short and long-range dependencies in parallel
- Unlike RNNs: no sequential bottleneck

**Example** — "The animal didn't cross the street because it was too tired."

Self-attention allows associating the pronoun "it" with "animal" through learned attention weights, demonstrating semantic relationship capture.

---

## Multi-Head Attention

Rather than a single attention function, apply attention $h$ times in parallel with different learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Why multiple heads?**
- Attend to different relationship types simultaneously
- First layer: word-pair interactions
- Second layer: pair-of-pairs interactions (broader receptive field)
- Provides redundancy; stacked layers with skip connections allow error recovery

**Decoder masking:** Self-attention layer only attends to earlier positions by masking future positions to $-\infty$ before softmax.

---

## Encoder vs Decoder Roles

| Role | Description |
|------|-------------|
| **Encoder** | Processes full input bidirectionally. Self-attention enables each word to weigh all others, capturing short and long-range dependencies |
| **Decoder** | Takes start token and encoder embeddings, outputs next-word probabilities. Uses masked self-attention (can't see future) + cross-attention to encoder |

**Cross-attention:** Queries from decoder, keys/values from encoder output.

**Word generation:** Decoder outputs probability distribution over vocabulary. Selection via greedy (argmax) or beam search (top-n).

---

## Ghost Attention (Llama 2)

**Purpose:** Helps models remember and adhere to initial instructions throughout extended conversations.

**Technique:**
1. Instruction appended to all synthetic dialogue user messages during training
2. During training, instruction retained only in first turn
3. Loss set to zero for tokens from earlier turns

**Context Distillation:** Extracts crucial details from context, highlighting them for better instruction adherence.

**Result:** Maintains instruction adherence across 20+ dialogue turns until maximum context length is reached.

---

## Linformer: Linear Complexity Attention

**Problem:** Standard self-attention has $O(n^2)$ time/space complexity — expensive for long sequences.

**Core innovation:** Self-attention can be approximated using low-rank matrices, reducing complexity to $O(n)$.

**Approach:**
1. Self-attention relationships don't require full-rank matrices
2. Introduce projection matrices $E_i$ and $F_i$ for key/value computation
3. Project high-dimensional K/V matrices to lower-dimensional space $(n \times k)$
4. Low-rank factorization approximates full-rank attention

$$\tilde{P} = \text{softmax}\left(\frac{Q (E_i K)^T}{\sqrt{d_k}}\right)$$

**Benefit:** Particularly valuable for long documents where standard Transformers are infeasible.

---

## Luong Attention Variants (2015)

Luong et al. proposed alternative alignment score functions:

| Variant | Formula |
|---------|---------|
| **Dot** | $\text{score}(s_t, h_s) = s_t^T h_s$ |
| **General** | $\text{score}(s_t, h_s) = s_t^T W_a h_s$ |
| **Concat** | $\text{score}(s_t, h_s) = v_a^T \tanh(W_a[s_t; h_s])$ |

**Local Attention:** Focuses on a subset of source positions per target word rather than all words, improving computational efficiency.

---

## Key Takeaways

1. **Bottleneck problem:** Fixed context vector in seq2seq fails on long sequences
2. **Dynamic context:** Attention computes distinct $c_i$ per output step by weighting all encoder states
3. **Q-K-V:** Flexible, content-dependent weighting via query-key-value mechanism
4. **Self-attention:** Parallel processing without recurrence; every word attends to every other
5. **Multi-head:** Multiple parallel attention layers capture diverse relationship types
6. **Modern extensions:** Ghost Attention (instruction adherence), Linformer (linear complexity)

## References

- Bahdanau et al. (2015) — Neural Machine Translation by Jointly Learning to Align and Translate
- Cho et al. (2014) — Learning Phrase Representations using RNN Encoder–Decoder
- Luong et al. (2015) — Effective Approaches to Attention-based Neural Machine Translation
- Vaswani et al. (2017) — Attention Is All You Need
- Meta AI (2023) — Llama 2: Open Foundation and Fine-Tuned Chat Models
- Wang et al. (2020) — Linformer: Self-Attention with Linear Complexity
