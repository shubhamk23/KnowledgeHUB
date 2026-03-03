---
title: "Large Language Models: Overview"
slug: llm-overview
summary: "How LLMs work, scaling laws, emergent abilities, and the landscape of foundation models from GPT to Gemini."
tags: ["LLM", "GPT", "scaling", "emergent-abilities", "few-shot", "foundation-model"]
visibility: public
---

# Large Language Models: Overview

## What is an LLM?

A **Large Language Model (LLM)** is a neural network (typically Transformer-based) trained on massive text corpora to model the probability distribution of language:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T p(x_t | x_1, \ldots, x_{t-1})$$

"Large" refers to both model size (billions of parameters) and training data scale (trillions of tokens).

**Training objective:** Minimize the negative log-likelihood (next-token prediction):

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log p_\theta(x_t | x_{<t})$$

---

## Scale: Parameter Counts

| Model | Parameters | Organization | Year |
|-------|-----------|--------------|------|
| GPT-2 | 1.5B | OpenAI | 2019 |
| GPT-3 | 175B | OpenAI | 2020 |
| PaLM | 540B | Google | 2022 |
| GPT-4 | ~1.8T (MoE est.) | OpenAI | 2023 |
| LLaMA 2 | 7B / 13B / 70B | Meta | 2023 |
| Gemini Ultra | ~1T+ (est.) | Google | 2023 |
| Claude 3 | Unknown | Anthropic | 2024 |
| LLaMA 3.1 | 8B / 70B / 405B | Meta | 2024 |

---

## Scaling Laws

**Kaplan et al. (2020)** discovered that loss scales as a power law with compute:

$$\mathcal{L}(C) \propto C^{-\alpha}$$

For a given compute budget $C$ (FLOPs), the optimal model size $N$ and training tokens $D$ scale as:

$$N \propto C^{0.73}, \quad D \propto C^{0.27}$$

**Chinchilla (Hoffmann et al., 2022)** revised this — previous models were significantly undertrained:

$$N_{\text{opt}} = D_{\text{opt}} \quad (\text{approx.})$$

**Chinchilla's rule:** For a model of $N$ parameters, train on approximately $20 \times N$ tokens.

| Model | Parameters | Training Tokens | Ratio |
|-------|-----------|-----------------|-------|
| GPT-3 | 175B | 300B | 1.7× — **undertrained** |
| Chinchilla | 70B | 1.4T | 20× — optimal |
| LLaMA 2 70B | 70B | 2T | 28× — overtrained (for smaller inference) |

---

## Pre-training Data

Modern LLMs train on diverse internet-scale data:

| Source | Example | Typical % |
|--------|---------|-----------|
| Common Crawl | Web text | 60-70% |
| Books | BookCorpus, Books3 | 10-20% |
| Wikipedia | All languages | 5-10% |
| Code | GitHub | 5-15% |
| Scientific | ArXiv, PubMed | 2-5% |
| Conversations | Reddit, forums | 2-5% |

**Data quality >> data quantity.** Careful filtering (language detection, deduplication, quality scoring) has large impact.

---

## Emergent Abilities

Abilities that appear **suddenly** at certain scales — not present in smaller models:

| Ability | Appears Around |
|---------|---------------|
| Few-shot learning | ~100B params |
| Chain-of-thought reasoning | ~100B params |
| Instruction following (without fine-tuning) | ~100B params |
| Multi-step arithmetic | ~100B params |
| Code generation | ~10B params |
| Theory of mind | ~100B params |

**Controversy:** Some argue emergent abilities are artifacts of discontinuous metrics (Wei et al.), not true emergence.

---

## Key Capabilities

### Few-Shot / In-Context Learning (ICL)

No gradient updates — learning from examples in the prompt:

```
Input: "Translate English to French: Sea → "
Output: "Mer"

Input: "Translate English to French: Moon → "
Output: "Lune"

Input: "Translate English to French: Star → "
Output: ???
```

**Why it works:** LLMs implicitly implement gradient-descent-like algorithms through attention (Akyürek et al., 2022).

### Chain-of-Thought (CoT)

Prompting the model to reason step-by-step:

```
Q: A store has 5 boxes with 12 items each. 3 boxes were sold. How many items remain?

Standard: 24  ← often wrong

CoT: "Start with 5 × 12 = 60 items total.
      3 boxes sold = 3 × 12 = 36 items.
      Remaining: 60 - 36 = 24 items." ← reasoning shown
```

Significant improvements on math, logic, and multi-step reasoning.

### Tool Use & Function Calling

LLMs can call external APIs and tools:

```json
{
  "function": "web_search",
  "arguments": {"query": "current weather in Paris"}
}
```

Foundation for AI agents.

---

## LLM Families

### Closed / Proprietary

| Model | Strengths |
|-------|-----------|
| GPT-4o (OpenAI) | Best overall, multimodal |
| Claude 3.5 Sonnet (Anthropic) | Reasoning, coding, safety |
| Gemini 1.5 Pro (Google) | 1M context, multimodal |

### Open-Weight

| Model | Parameters | Strengths |
|-------|-----------|-----------|
| LLaMA 3.1 (Meta) | 8B-405B | Strong open model |
| Mistral 7B | 7B | Efficient, Apache 2.0 |
| Falcon 180B | 180B | Large open model |
| Qwen (Alibaba) | 7B-72B | Strong multilingual |
| Phi-3 (Microsoft) | 3.8B | Small but capable |

---

## Mixture of Experts (MoE)

Route tokens to different "expert" sub-networks instead of activating all parameters:

$$\text{MoE}(x) = \sum_{i=1}^N G(x)_i \cdot E_i(x)$$

Where $G(x)$ is a router that selects top-$k$ experts (typically $k=2$) for each token.

**Advantage:** Total parameters = $N \times$ expert size, but only $k \times$ expert size is active per token → better performance at same compute.

**Examples:** Mixtral 8×7B (Mistral), GPT-4 (estimated), Switch Transformer (Google).

---

## Inference Optimization

| Technique | Speedup | Memory | Quality Loss |
|-----------|---------|--------|-------------|
| FP16 inference | 2× | -50% | None |
| INT8 quantization | 3-4× | -75% | Minimal |
| INT4/NF4 quantization | 6-8× | -87% | Small |
| Flash Attention | 2-4× | -50% | None |
| KV cache | Removes repeated computation | +memory | None |
| Speculative decoding | 2-3× | Minimal | None |
| Continuous batching | High throughput | Same | None |

---

## Key Takeaways

1. **LLMs are autoregressive** — trained to predict next token, generalize to complex tasks
2. **Scaling laws** predict performance from compute — Chinchilla optimal: 20 tokens/parameter
3. **Emergent abilities** appear at scale — few-shot learning, CoT, instruction following
4. **Open vs closed:** LLaMA/Mistral series are increasingly competitive with proprietary models
5. **MoE architectures** allow massive parameter counts with controlled compute
6. **Inference optimization** (quantization, Flash Attention) makes deployment feasible

## References

- Brown et al. (2020) — Language Models are Few-Shot Learners (GPT-3)
- Kaplan et al. (2020) — Scaling Laws for Neural Language Models
- Hoffmann et al. (2022) — Training Compute-Optimal LLMs (Chinchilla)
- Wei et al. (2022) — Emergent Abilities of Large Language Models
- Akyürek et al. (2022) — What Learning Algorithm is In-Context Learning?
