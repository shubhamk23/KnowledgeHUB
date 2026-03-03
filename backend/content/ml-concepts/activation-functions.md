---
title: "Activation Functions"
slug: activation-functions
summary: "ReLU, GELU, SiLU, sigmoid, tanh and more — how non-linearities shape neural network learning."
tags: ["ReLU", "GELU", "sigmoid", "activation", "deep-learning", "SiLU"]
visibility: public
---

# Activation Functions

## Overview

Activation functions introduce **non-linearity** into neural networks. Without them, a deep network collapses into a linear model regardless of depth:

$$W_3(W_2(W_1 x)) = (W_3 W_2 W_1) x = W_{\text{eff}} x$$

A good activation function should be:
- Non-linear (to learn complex patterns)
- Differentiable (for backpropagation)
- Computationally efficient
- Not prone to vanishing/exploding gradients

---

## Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}} \in (0, 1)$$

**Derivative:**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Properties:**
- Output range: $(0, 1)$ — interpretable as probability
- Saturates at extremes → **vanishing gradient problem**
- Not zero-centered (all outputs positive → zig-zag gradient updates)

**Use when:** Output layer for binary classification, gating mechanisms (LSTM).

**Avoid:** Hidden layers of deep networks (use ReLU variants instead).

---

## Hyperbolic Tangent (tanh)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1 \in (-1, 1)$$

**Derivative:**
$$\tanh'(x) = 1 - \tanh^2(x)$$

**Properties:**
- Zero-centered (better than sigmoid for gradient flow)
- Still saturates → vanishing gradients in deep networks
- Range: $(-1, 1)$

**Use when:** RNNs (hidden states), output layer for bounded regression.

---

## ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

**Derivative:**
$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**Advantages:**
- No saturation for $x > 0$ → alleviates vanishing gradients
- Computationally trivial (thresholding operation)
- Induces sparsity (many neurons output 0)

**Dying ReLU Problem:**
If a neuron's input is always negative, gradient is always 0 → neuron never updates → "dead" neuron.

Causes: Large learning rate, bad weight initialization.

**Use when:** Default for most hidden layers in CNNs and MLPs.

---

## Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

Typically $\alpha = 0.01$ (small negative slope).

**Fixes:** Dying ReLU — always has non-zero gradient.

**PReLU (Parametric ReLU):** $\alpha$ is learned during training.

---

## ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

**Properties:**
- Smooth, differentiable everywhere
- Negative outputs push mean activations toward zero (helps batch normalization)
- More computationally expensive than ReLU

---

## GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Approximation used in practice:**
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

**Intuition:** Stochastically gates inputs based on their magnitude. For large $x$ → passes through; for large negative $x$ → suppressed. It's ReLU but smooth.

**Properties:**
- No dying units (non-zero gradient everywhere)
- Smooth — helps with optimization
- Standard in Transformers: **BERT, GPT-2/3/4, ViT**

---

## SiLU / Swish

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Also called:** Swish (Ramachandran et al., 2017). SiLU = Sigmoid Linear Unit.

**Properties:**
- Smooth, non-monotonic (slight dip below 0 for small negative $x$)
- Unbounded above (like ReLU), bounded below (unlike ReLU)
- Consistently outperforms ReLU in deep networks

**Used in:** EfficientNet, **LLaMA**, Gemini, many modern vision models.

---

## GLU Variants (Gated Linear Units)

$$\text{GLU}(x, W, V, b, c) = \sigma(xW + b) \odot (xV + c)$$

**SwiGLU** (Shazeer, 2020):
$$\text{SwiGLU}(x, W, V) = \text{SiLU}(xW) \odot (xV)$$

Used in the **FFN of LLaMA 2/3**, PaLM, Gemini — replaces the standard 2-layer FFN:

```python
# Standard FFN
h = ReLU(x @ W1 + b1) @ W2 + b2

# SwiGLU FFN (LLaMA style)
h = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
# Note: uses 3 weight matrices instead of 2
```

**Why SwiGLU?** Gating allows the network to selectively pass information, improving expressivity.

---

## Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$$

**Properties:**
- Output sums to 1 → probability distribution
- Sensitive to scale (use temperature $T$ for calibration)
- Numerically stable form: subtract $\max(x)$ before exp

**Temperature-scaled softmax:**
$$\text{softmax}(x_i / T): \quad T \to 0 \text{ (hard)} \quad T \to \infty \text{ (uniform)}$$

---

## Comparison Table

| Activation | Range | Saturates | Dying | Used In |
|-----------|-------|-----------|-------|---------|
| Sigmoid | (0,1) | Yes (both) | No | Binary output, LSTM gates |
| Tanh | (-1,1) | Yes (both) | No | RNNs |
| ReLU | [0,∞) | No (pos) | Yes | CNNs, MLPs |
| Leaky ReLU | (-∞,∞) | No | No | CNNs (alternative) |
| GELU | ~(-0.17,∞) | No | No | BERT, GPT |
| SiLU/Swish | ~(-0.28,∞) | No | No | LLaMA, EfficientNet |
| SwiGLU | — | No | No | LLaMA 2/3, PaLM |
| Softmax | (0,1)^K | Relative | No | Output layer (multiclass) |

---

## Choosing an Activation Function

```
Transformer LLM?    → SwiGLU (FFN), Softmax (attention, output)
CNN / ViT?          → GELU or ReLU
Simple MLP?         → ReLU or GELU
Binary output?      → Sigmoid
Multi-class output? → Softmax
RNN / LSTM gates?   → Sigmoid (gates), Tanh (states)
Self-supervised?    → GELU
```

---

## Key Takeaways

1. **ReLU** is simple and effective but suffers from dying neurons in deep nets
2. **GELU** has replaced ReLU as the default in Transformers (BERT, GPT, ViT)
3. **SiLU/Swish** is preferred in modern LLMs (LLaMA, Gemini)
4. **SwiGLU** adds gating to SiLU — best performing FFN activation in current LLMs
5. **Sigmoid/Tanh** remain useful at output layers and in gating mechanisms
6. **Temperature** in softmax controls entropy of the distribution (sampling strategies)

## References

- Hendrycks & Gimpel (2016) — GELU Activation Function
- Ramachandran et al. (2017) — Searching for Activation Functions (Swish)
- Shazeer (2020) — GLU Variants Improve Transformer
- Touvron et al. (2023) — LLaMA 2 (SwiGLU usage)
