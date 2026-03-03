---
title: "Token Sampling Strategies"
slug: token-sampling
summary: "Greedy, beam search, temperature, top-k, and top-p (nucleus) sampling — how LLMs decide what to generate next."
tags: ["sampling", "beam-search", "temperature", "top-k", "top-p", "greedy", "decoding"]
visibility: public
---

# Token Sampling Strategies

## Overview

After computing logits over the vocabulary, an LLM must decide **which token to generate next**. This decoding strategy profoundly affects output quality, diversity, and coherence.

For a vocabulary of size $|V|$, the model produces logits $\mathbf{z} \in \mathbb{R}^{|V|}$, then converts to probabilities:

$$p(w_i | w_{<t}) = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

---

## Greedy Decoding

At each step, select the token with **maximum probability**:

$$w_t = \arg\max_{w} p(w | w_{<t})$$

**Advantages:**
- Deterministic
- Fast (single forward pass per step)
- Good for factual, precise tasks

**Disadvantages:**
- Locally optimal ≠ globally optimal
- Can get stuck in repetitive loops
- Low diversity, no randomness

```python
# Greedy decoding
next_token = logits.argmax(dim=-1)
```

**Use when:** Code generation (exact syntax), factual QA, translation of short sequences.

---

## Beam Search

Maintains a **beam** of the $B$ most probable partial sequences at each step:

At step $t$, for each of $B$ beams, expand to all $|V|$ tokens → keep top $B$ by cumulative log-probability.

**Cumulative score:**
$$\text{score}(\mathbf{w}_{1:t}) = \sum_{i=1}^t \log p(w_i | w_{<i})$$

**Length penalty** (to avoid short sequences winning):
$$\text{score}_{\text{normalized}} = \frac{\text{score}(\mathbf{w}_{1:t})}{t^\alpha}$$

Typical $\alpha = 0.6$–$0.8$.

**Advantages:**
- Better than greedy (explores multiple paths)
- Produces high-likelihood outputs

**Disadvantages:**
- Computationally expensive ($B \times |V|$ per step)
- Favors generic, high-probability phrases ("safe" but boring outputs)
- Still deterministic → no diversity

```python
# HuggingFace beam search
outputs = model.generate(
    input_ids,
    num_beams=5,
    length_penalty=0.8,
    early_stopping=True
)
```

**Use when:** Machine translation, summarization (quality-focused).

### Diverse Beam Search

Penalizes beams that are too similar to each other, encouraging diversity within the beam.

---

## Temperature Sampling

Apply **temperature** $T$ to scale logits before sampling:

$$p_T(w_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

**Effect of temperature:**

| Temperature | Distribution | Behavior |
|-------------|-------------|----------|
| $T \to 0$ | Peaked (near greedy) | Deterministic, repetitive |
| $T = 1$ | Original softmax | Standard sampling |
| $T > 1$ | Flattened (more uniform) | More random, creative |
| $T \to \infty$ | Uniform | Random noise |

```python
# Temperature sampling
logits_scaled = logits / temperature
probs = torch.softmax(logits_scaled, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

**Intuition:** Temperature controls the "sharpness" of the distribution. Low T concentrates mass on the most likely tokens; high T spreads it across more tokens.

**Use when:**
- Creative writing: $T = 0.8$–$1.2$
- Factual tasks: $T = 0.1$–$0.5$
- Never use $T > 1.5$ in production (output degrades)

---

## Top-k Sampling

Sample only from the **top-k most probable tokens**:

1. Sort vocabulary by probability (descending)
2. Keep top $k$ tokens
3. Re-normalize probabilities
4. Sample from this restricted set

$$p_k(w_i) = \begin{cases} \frac{p(w_i)}{\sum_{j \in V_k} p(w_j)} & w_i \in V_k \\ 0 & \text{otherwise} \end{cases}$$

Where $V_k$ = top-k tokens by probability.

**Common values:** $k = 10$–$100$

**Problem:** Fixed $k$ is context-insensitive:
- When distribution is flat (uncertain): $k=50$ may still include many bad tokens
- When distribution is peaked (confident): $k=50$ includes many garbage tokens

```python
# Top-k filtering
top_k_probs, top_k_indices = torch.topk(probs, k=50)
next_token = top_k_indices[torch.multinomial(top_k_probs, 1)]
```

---

## Top-p (Nucleus) Sampling

Sample from the **smallest set of tokens whose cumulative probability exceeds $p$**:

1. Sort vocabulary by probability (descending)
2. Find smallest set $V_p$ such that $\sum_{w \in V_p} p(w) \geq p$
3. Re-normalize within $V_p$ and sample

$$V_p = \text{smallest set}: \sum_{w \in V_p} p(w) \geq p$$

**Typical values:** $p = 0.9$–$0.95$

**Why it's better than top-k:**
- **Flat distribution** (uncertain): nucleus expands to include more tokens
- **Peaked distribution** (confident): nucleus shrinks to just top few tokens
- Adapts dynamically to the distribution shape

```python
# Top-p (nucleus) sampling
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

# Remove tokens with cumulative probability above the threshold
sorted_indices_to_remove = cumulative_probs > 0.9
# Keep at least 1 token
sorted_indices_to_remove[..., 0] = 0
# Sample from filtered distribution
filtered_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.)
next_token = sorted_indices[torch.multinomial(filtered_probs, 1)]
```

---

## Min-p Sampling

**Newer alternative** to top-p (2023): Set a minimum probability threshold as a fraction of the maximum token probability:

$$p_{\min} = \alpha \cdot \max_i p(w_i)$$

Remove all tokens with $p(w_i) < p_{\min}$.

**Advantage:** Scales relative to the peak, better behavior at high temperatures.

---

## Typical Sampling

Sample tokens that are "typical" — not too surprising, not too expected — based on conditional entropy:

$$\mathcal{T}(\epsilon) = \{w : |{-\log p(w | w_{<t})} - H(p(\cdot | w_{<t}))| \leq \epsilon\}$$

Where $H$ is the entropy of the distribution.

**Use when:** Long-form generation where maintaining naturalness is important.

---

## Repetition Penalty

Penalize previously generated tokens to reduce repetition:

$$z'_i = \begin{cases} z_i / \text{penalty} & w_i \in \text{generated tokens} \\ z_i & \text{otherwise} \end{cases}$$

Typical penalty: $1.1$–$1.5$.

---

## Combined Strategy (Production Default)

Most production LLM APIs use a combination:

```python
def generate_token(logits, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1):
    # 1. Apply repetition penalty
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # 2. Apply temperature
    logits = logits / temperature

    # 3. Apply top-k
    logits = top_k_filter(logits, top_k)

    # 4. Apply top-p
    probs = torch.softmax(logits, dim=-1)
    probs = top_p_filter(probs, top_p)

    # 5. Sample
    return torch.multinomial(probs, num_samples=1)
```

---

## Strategy Comparison

| Strategy | Deterministic | Diversity | Quality | Best For |
|---------|--------------|-----------|---------|----------|
| Greedy | ✅ | ❌ | High (local) | Code, factual QA |
| Beam Search (B=5) | ✅ | Low | High (global) | Translation, summarization |
| Temperature only | ❌ | ✅ | Variable | Creative writing |
| Top-k | ❌ | Moderate | Good | General purpose |
| Top-p (nucleus) | ❌ | ✅ | Good | General purpose |
| Top-k + Top-p + T | ❌ | ✅✅ | Best | Production default |

---

## Key Takeaways

1. **Greedy** is fast and deterministic but often locally suboptimal
2. **Beam search** is better for translation/summarization but boring for creative tasks
3. **Temperature** scales all probabilities — lower T for precision, higher for creativity
4. **Top-k** restricts to k tokens but is context-insensitive
5. **Top-p (nucleus)** dynamically adjusts vocabulary size based on cumulative probability — generally preferred over top-k
6. **Production default:** Temperature (0.7–0.9) + Top-p (0.9) + Repetition penalty (1.1)
7. **Min-p and Typical sampling** are newer alternatives gaining adoption

## References

- Holtzman et al. (2019) — The Curious Case of Neural Text Degeneration (Nucleus Sampling)
- Fan et al. (2018) — Hierarchical Neural Story Generation (Top-k Sampling)
- Meister et al. (2023) — Locally Typical Sampling
- Nguyen (2023) — Min-P Sampling
