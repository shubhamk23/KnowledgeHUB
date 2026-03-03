---
title: "Transformer Architecture"
slug: transformers-architecture
summary: "A complete guide to the Transformer architecture — embeddings, positional encoding, encoder/decoder stacks, multi-head attention, and modern extensions like RoPE and KV cache."
tags: ["transformer", "attention", "positional-encoding", "BPE", "encoder", "decoder"]
visibility: public
---

# Transformer Architecture

## Overview

The Transformer (Vaswani et al., 2017) replaced recurrent networks with a fully attention-based architecture, enabling parallel processing and superior capture of long-range dependencies. It is the foundation of virtually every modern language model.

**Core idea:** Instead of processing text word-by-word (RNN), connect every word to every other word simultaneously via self-attention.

---

## Representation Learning Fundamentals

### One-Hot Encoding

Converts categorical variables (words) into binary vectors where exactly one position equals 1:

```
"find"  → [0, 1, 0]
"my"    → [0, 0, 1]
"files" → [1, 0, 0]
```

**Problem:** Creates false ordinal relationships; doesn't encode similarity.

### Dot Product

$$a \cdot b = \sum_{i} a_i b_i = \|a\| \|b\| \cos\theta$$

Key properties for attention:
- One-hot with itself: 1
- One-hot with different one-hot: 0
- Measures vector similarity

### Matrix Multiplication

Comprises a series of dot products. Matrix $A$ ($n \times m$) times $B$ ($m \times p$) → result ($n \times p$). The foundation of all linear transformations in Transformers.

---

## Embeddings

Projects one-hot vectors from $N$-dimensional space into lower-dimensional continuous space (typically 512–768 dimensions):

$$\text{embedded} = \text{one-hot}_{[1 \times N]} \times W_e^{[N \times d_{\text{model}}]}$$

**Benefits:**
- Reduces parameters from $N^2$ to $N \times d_{\text{model}}$
- Encodes semantic similarity (similar words → similar vectors)
- Generalizes across semantically related words

---

## Positional Encoding

Transformers process all tokens in parallel — they have no inherent notion of sequence order. Positional encoding injects position information.

### Sinusoidal (Absolute) Encoding

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Creates circular "wiggle" perturbations of increasing frequency. The model can attend to relative positions by linearity of sines.

**Limitation:** Doesn't generalize to sequence lengths longer than seen during training.

### Relative Positional Encoding

Encodes distance between tokens rather than absolute positions. Requires $2N - 1$ unique encodings for sequence length $N$. Better for tasks where relative order matters more than absolute position.

### Rotary Position Embeddings (RoPE)

Applies rotation matrices to embeddings based on position:

$$\text{RoPE}(x, pos) = R(pos) \cdot x$$

Where $R(pos)$ is a block-diagonal rotation matrix. Used in LLaMA, GPT-NeoX, and most modern LLMs.

**Advantages:**
- Captures both absolute and relative positions
- Efficient handling of long contexts (thousands of tokens)
- Relative position information preserved after dot-product attention

---

## Attention Mechanism

The core of every Transformer layer:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Each head attends to different representation subspaces simultaneously. In the original paper: $h = 8$ heads, $d_k = d_v = 64$, $d_{\text{model}} = 512$.

---

## Feed-Forward Network

Applied after attention in every layer:

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

Two linear transformations with a ReLU in between. Typical hidden dimension is $4 \times d_{\text{model}}$ (e.g., 2048 for $d_{\text{model}} = 512$).

**Role:** Creates multi-word features (W1), thresholds to binary presence (ReLU), applies transition probabilities (W2).

---

## Layer Normalization

Applied before attention and feed-forward blocks (Pre-LN in modern variants):

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

Normalizes across embedding dimensions. Improves training stability and gradient flow.

---

## Skip Connections

Residual connections add layer inputs to outputs:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Benefits:**
- Enables gradient flow through deep networks
- Prevents vanishing gradients
- Allows deep architectures (BERT: 12 layers, GPT-3: 96 layers)

---

## Encoder vs Decoder Architecture

### Encoder Stack

- Processes full input sequences bidirectionally
- Each layer: multi-head self-attention + feed-forward
- All positions attend to all others simultaneously
- Used in: BERT, RoBERTa (encoder-only models)

### Decoder Stack

- Generates output sequences autoregressively (token by token)
- **Masked self-attention:** prevents attending to future tokens (causal mask)
- **Cross-attention:** queries from decoder, keys/values from encoder output
- Used in: GPT series (decoder-only), T5 (encoder-decoder)

### Cross-Attention

Enables seq2seq tasks and multimodal fusion:
- Query: from decoder layer below
- Key/Value: from encoder stack output

---

## Tokenization

Converts raw text into discrete tokens. Modern Transformers use **subword tokenization**.

### Byte Pair Encoding (BPE)

Iteratively merges the most frequent byte pairs in the corpus:

```python
# Vocabulary starts character-level: {"l o w", "l o w e r", "n e w e s t", ...}
# Step 1: Count pairs: "e s" → 9, "e r" → 5, ...
# Step 2: Merge most frequent: "es" → new token
# Step 3: Repeat until target vocab size
```

**Benefits:** Handles rare and misspelled words efficiently via subword decomposition.

### WordPiece (BERT)

Similar to BPE but maximizes likelihood of training data rather than frequency-based merges. Uses `##` prefix for continuation tokens.

### SentencePiece

Language-agnostic; treats text as raw unicode stream. Used in T5, LLaMA, XLNet.

---

## Training Techniques

### Teacher Forcing

During training, decoder receives ground-truth previous tokens rather than its own generated predictions:

```
Target: "The cat sat on the mat"
Step 1: input=[<BOS>] → predict "The"
Step 2: input=[<BOS>, "The"] → predict "cat"  # uses ground truth, not prediction
```

**Benefit:** Faster convergence. **Problem:** Exposure bias — mismatch between training and inference.

### Scheduled Sampling

Gradually transitions from teacher forcing to model-generated tokens during training, reducing exposure bias.

### Label Smoothing

Instead of hard one-hot targets, use soft distributions (e.g., 0.9 for correct class, 0.1 / (V-1) for others). Regularizes the model and improves generalization.

---

## Inference Optimizations

### KV Cache

During autoregressive generation, cache the key and value matrices from previous steps:

```python
# Without cache: recompute K,V for all previous tokens at each step
# With KV cache: K,V are stored; only compute for new token
```

**Speedup:** Reduces computation from $O(n^2)$ to $O(n)$ per new token at inference. Critical for long sequences.

### Greedy Decoding

Select the highest-probability token at each step:

```python
next_token = argmax(softmax(logits))
```

Fast but suboptimal — can miss globally better sequences.

### Beam Search

Maintain top-$k$ partial sequences at each step:
- $k = 1$: greedy
- $k = 5$: typical for translation
- Produces better sequences but $k \times$ slower

---

## RNNs vs Transformers: Comparison

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| Processing | Sequential (word by word) | Parallel (all at once) |
| Time complexity | $O(n \cdot d^2)$ | $O(n^2 \cdot d)$ |
| Long-range deps | Poor (vanishing gradient) | Excellent (direct attention) |
| Training speed | Slow (sequential) | Fast (parallelizable) |
| Memory | $O(n)$ | $O(n^2)$ for attention |
| Inductive bias | Strong sequential | Weak (needs positional encoding) |

---

## Parameter Distribution

Typical Transformer layer (per layer):

| Component | Parameters |
|-----------|-----------|
| Self-attention (Q, K, V, O projections) | $4 d^2$ |
| Feed-forward (W1, W2) | $8 d^2$ |
| Layer norms | $4d$ |

Ratio: ~1:2 attention vs feed-forward. GPT-3 (175B params) has 96 layers.

---

## Transformers as Graph Neural Networks

Sentences function as **fully-connected word graphs**:
- Nodes = words
- Edges = connections between all word pairs
- Attention weights = edge strengths (learned from content)

Transformers are GNNs with fully-connected topology. This framing helps understand their expressiveness.

---

## Key Takeaways

1. **Parallel attention** replaces sequential recurrence — enables faster training and better long-range dependencies
2. **Positional encoding** (sinusoidal or RoPE) is required since attention has no inherent order
3. **Encoder-only** (BERT): good for classification/understanding. **Decoder-only** (GPT): good for generation. **Encoder-decoder** (T5): good for seq2seq
4. **KV cache** is essential for efficient autoregressive inference
5. **BPE/WordPiece** subword tokenization handles vocabulary gracefully
6. **Multi-head attention** captures diverse relationship types in parallel

## References

- Vaswani et al. (2017) — Attention Is All You Need
- Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers
- Su et al. (2021) — RoFormer: Enhanced Transformer with Rotary Position Embedding
- Sennrich et al. (2016) — Neural Machine Translation of Rare Words with Subword Units
- Scheduled Sampling — Bengio et al. (2015)
