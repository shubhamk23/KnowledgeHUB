---
title: "DeepSeek: MLA, MoE & Pure RL Reasoning"
slug: deepseek
summary: "DeepSeek's V2/V3/R1 series — Multi-head Latent Attention (MLA), auxiliary-loss-free MoE load balancing, and GRPO-based reasoning without supervised fine-tuning."
tags: ["DeepSeek", "MLA", "MoE", "GRPO", "reasoning", "R1", "V3", "efficiency", "open-source"]
visibility: public
---

# DeepSeek: MLA, MoE & Pure RL Reasoning

## Overview

**DeepSeek** (DeepSeek AI, China) has produced some of the most technically innovative and cost-efficient LLMs in 2024-2025. Three major innovations define the series:

1. **MLA (Multi-head Latent Attention):** 28× KV cache compression enabling long-context inference at a fraction of the memory cost
2. **Auxiliary-loss-free MoE load balancing:** Dynamic bias adjustment replaces degrading auxiliary losses
3. **GRPO (Group Relative Policy Optimization):** Pure RL reasoning without any supervised fine-tuning — rivaling OpenAI's o1

**Cost efficiency:** DeepSeek-V3 (671B parameters) was trained in 2.788M H800 GPU hours — roughly $5.6M — while comparable closed-source models cost an estimated $100M+ to train.

---

## DeepSeek-V2 (May 2024)

**Paper:** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (arXiv:2405.04434)

### Multi-head Latent Attention (MLA)

**The problem:** Standard multi-head attention (MHA) requires caching $O(n \cdot H \cdot d_k)$ key-value pairs per token, where $n$ = sequence length, $H$ = heads, $d_k$ = head dimension. For long contexts this dominates memory.

**MLA solution:** Low-rank joint compression of keys and values into a latent vector $c_t^{KV}$:

**Compression:**
$$c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll H \cdot d_k$$

**Decompression:**
$$K_t^h = W^{UK} c_t^{KV}, \quad V_t^h = W^{UV} c_t^{KV}$$

Where:
- $h_t \in \mathbb{R}^d$ — input hidden state at position $t$
- $W^{DKV} \in \mathbb{R}^{d_c \times d}$ — down-projection (compression)
- $W^{UK}, W^{UV} \in \mathbb{R}^{d_h H_{\text{KV}} \times d_c}$ — up-projections (decompression)
- $d_c = 512$ (latent dim), $H = 128$ heads, $d_h = 128$ (head dim)

**Cache comparison:**

| Attention Type | Cache Per Token | DeepSeek-V2 |
|----------------|----------------|-------------|
| Multi-Head (MHA) | $H \times d_h \times 2 = 32{,}768$ | Baseline |
| Multi-Query (MQA) | $d_h \times 2 = 256$ | Loses quality |
| Grouped-Query (GQA, 8 groups) | $4{,}096$ | Moderate |
| **MLA** | **512** | **28× reduction** |

**Quantitative impact:**
- Standard MHA: 213.5 GB KV cache for typical long-context workload
- MLA: **7.6 GB** — fits on a single GPU
- Max generation throughput: **5.76× improvement** over DeepSeek 67B

**Query rope integration:** MLA uses a separate path for rotary position embeddings — queries carry RoPE via a decoupled head projection, while the main KV cache remains position-agnostic (enabling re-use across positions):

$$q_t^R = W^{QR} h_t, \quad k_t^R = W^{KR} h_t$$
$$\text{Attention score} = q_t^C k_j^C + q_t^R k_j^R$$

This ensures positional information is preserved without inflating the cache.

### DeepSeek-MoE Architecture

DeepSeek-V2 uses a **fine-grained MoE** design:

- **256 total experts** per FFN layer (vs 8 in Mixtral)
- **Top-K routing:** Select top-6 routed experts + always activate 2 "shared experts"
- **Shared experts:** Certain experts always active for all tokens — capture global patterns
- **Fine-grained routing:** More experts, smaller each → finer specialization

**Expert routing formula:**
$$g_i(s) = \begin{cases} s_i & s_i \in \text{Top-}K(\{s_j\}) \\ 0 & \text{otherwise} \end{cases}$$

$$\text{FFN}(u) = \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(u) + \sum_{i=1}^{N_r} g_i(u) \cdot \text{FFN}_i^{(r)}(u)$$

Where $N_s$ = shared experts, $N_r$ = routed experts.

**DeepSeek-V2 specs:**

| Attribute | Value |
|-----------|-------|
| Total parameters | 236B |
| Active per token | 21B |
| Context window | 128K |
| Training tokens | 8.1T |
| KV cache reduction | 93.3% |
| Training cost savings vs V1 | 42.5% |

---

## DeepSeek-V3 (December 2024)

**Paper:** "DeepSeek-V3 Technical Report" (arXiv:2412.19437)

### Scale and Efficiency

| Attribute | V2 | V3 | Change |
|-----------|-----|-----|--------|
| Total parameters | 236B | **671B** | +2.84× |
| Active per token | 21B | **37B** | +1.76× |
| Routed experts | 160 | **256** | +60% |
| Training tokens | 8.1T | **14.8T** | +83% |
| Pre-training GPU hours | ~800K | **2.664M** | — |
| Training stability | Some spikes | **Zero rollbacks** | — |

**Training cost:** 2.788M H800 GPU hours total (pre-train + post-train). At ~$2/H800 hour, roughly **$5.6M** — approximately 10× cheaper than training Llama 3 405B despite having 65% more parameters.

### Auxiliary-Loss-Free Load Balancing

**The problem with standard MoE:** To prevent routing collapse (all tokens going to a few experts), standard MoE training adds an auxiliary load-balancing loss. But larger auxiliary loss coefficients degrade model performance.

**DeepSeek's solution:** Dynamic bias adjustment without any auxiliary loss:

**During training:**
$$\hat{s}_{i,t} = s_{i,t} + b_i$$

Where $b_i$ is an expert-specific bias term, updated continuously:

$$b_i \leftarrow \begin{cases} b_i - \gamma & \text{if expert } i \text{ is overloaded in batch} \\ b_i + \gamma & \text{if expert } i \text{ is underloaded in batch} \end{cases}$$

**Key insight:** The bias only affects routing selection, **not the gating weights** used in the weighted sum. So expert utilization is balanced without distorting which expert contributes how much.

**Result:** Balanced expert utilization with zero performance degradation from auxiliary loss regularization.

### Multi-Token Prediction (MTP)

Beyond standard next-token prediction, V3 trains an auxiliary head that predicts $D=1$ additional future token:

$$\mathcal{L}_{\text{MTP}} = -\frac{1}{D} \sum_{k=1}^{D} \sum_{t=1}^{T-k} \log p(x_{t+k} | x_1, \ldots, x_t)$$

**Benefits:**
- Forces the model to plan ahead
- Improves coherence over longer spans
- Better performance on structured tasks (code, math)

The MTP module is used only during training — discarded at inference for efficiency.

### DeepSeek-V3 Benchmarks

| Benchmark | DeepSeek-V3 | LLaMA 3.1 405B | Claude 3.5 Sonnet | GPT-4o |
|-----------|------------|----------------|-------------------|--------|
| MMLU | 88.5% | 88.6% | 90.4% | 88.7% |
| MATH-500 | **90.2%** | 73.8% | 71.1% | 74.6% |
| HumanEval | 82.6% | 89.0% | 92.0% | 90.2% |
| AIME 2024 | **39.2%** | 23.7% | 16.0% | 9.3% |
| Codeforces | 51.6% | — | — | 23.6% |

**Notable:** DeepSeek-V3 **outperforms GPT-4o on MATH and AIME** at ~5% of the training cost.

---

## DeepSeek-R1 (January 2025)

**Paper:** "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (arXiv:2501.12948)

### The Core Claim

**Reasoning capabilities can emerge purely from reinforcement learning — no supervised fine-tuning on human-labeled reasoning demonstrations is needed.**

This challenges the conventional wisdom established by OpenAI's o1 (which relies on human-annotated CoT) and shows that RL alone can elicit sophisticated self-reflective reasoning.

### GRPO: Group Relative Policy Optimization

DeepSeek-R1 uses **GRPO** instead of standard PPO:

**Standard PPO (OpenAI o1 style):**
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}\left[\min\left(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

Requires a large **critic network** to estimate advantages $A_t$.

**GRPO (DeepSeek-R1):**
1. For each question $q$, sample a **group** of $G$ outputs: $\{o_1, o_2, \ldots, o_G\}$
2. Score each output with a reward: $\{r_1, r_2, \ldots, r_G\}$
3. Compute **group-normalized advantage** (no critic needed):

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

4. Policy gradient with KL regularization:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}\left[\sum_{i=1}^G \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} \hat{A}_i - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]\right]$$

**Why GRPO is better for reasoning:**
- No critic network overhead (eliminates ~50% of compute in standard PPO)
- Group-relative comparison is more stable than absolute value estimates
- Naturally encourages diversity within the group

### Reward Design

Rewards are **rule-based** (verifiable), not learned:

```python
def compute_reward(response, ground_truth):
    rewards = []

    # 1. Accuracy reward (main signal)
    if is_correct(extract_answer(response), ground_truth):
        rewards.append(1.0)
    else:
        rewards.append(0.0)

    # 2. Format reward (enforce <think>...</think> structure)
    if has_valid_format(response):
        rewards.append(0.1)
    else:
        rewards.append(-0.1)

    # 3. Length penalty for excessive verbosity
    if too_verbose(response):
        rewards.append(-0.05)

    return sum(rewards)
```

**No human judges** — all rewards are computed programmatically against ground truth answers.

### DeepSeek-R1-Zero: Pure RL Without SFT

**R1-Zero** applies GRPO directly to the DeepSeek-V3 base model with zero supervised fine-tuning:

**Emergent behaviors (not explicitly trained):**

1. **Self-verification:** Model learns to check its own work
2. **Reflection:** Backtracks when reasoning hits a dead end
3. **Strategy switching:** Changes approach when stuck
4. **"Aha moments":** Spontaneous course corrections mid-reasoning

```
[R1-Zero thinking trace — emergent without supervision]
Let me try approach 1... [works through calculation]
Hmm, this gives a negative result which can't be right.
Wait, I made an error — I used the wrong formula for the area.
Let me restart with the correct formula: A = πr² ...
[recalculates]
OK that gives 28.27 cm² which is positive and reasonable. ✓
```

**Challenge:** R1-Zero sometimes mixes languages and has verbose, hard-to-read reasoning.

**DeepSeek-R1:** Adds a cold-start SFT phase on ~1000 curated reasoning examples to improve readability, then applies GRPO. This gets the readability of supervised models with the reasoning depth of pure RL.

### R1 Benchmark Performance

| Benchmark | DeepSeek-R1 | OpenAI o1 | Claude 3.7 (ext. thinking) |
|-----------|------------|-----------|---------------------------|
| AIME 2024 | **79.8%** | 79.2% | 80.0% |
| MATH-500 | **97.3%** | 96.4% | — |
| GPQA Diamond | 71.5% | 75.7% | 84.8% |
| HumanEval | 92.6% | 92.4% | — |
| SWE-bench | 49.2% | 48.9% | 70.3% |
| Codeforces (rating) | **2029** | 1891 | — |

**Key result:** R1 matches o1 on AIME and HumanEval while being **fully open-source and far cheaper to run**.

### Distilled Reasoning Models

DeepSeek distills R1's reasoning into smaller dense models:

| Distilled Model | Base | AIME 2024 | MATH-500 |
|----------------|------|-----------|----------|
| R1-Distill-Qwen-1.5B | Qwen2.5-1.5B | 28.9% | 83.9% |
| R1-Distill-Qwen-7B | Qwen2.5-7B | 55.5% | 92.8% |
| R1-Distill-Llama-8B | LLaMA-3.1-8B | 50.4% | 89.1% |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 69.7% | 93.9% |
| R1-Distill-Qwen-32B | Qwen2.5-32B | **72.6%** | 94.3% |

**R1-Distill-Qwen-32B outperforms o1-mini on most benchmarks** — a 32B dense model matching a frontier reasoning model via distillation.

---

## Running DeepSeek Models

```python
# Via HuggingFace Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto"
)

# Format with thinking tags
prompt = """<|im_start|>user
Solve: if 3x + 7 = 22, what is x?
<|im_end|>
<|im_start|>assistant
<think>"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.6)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))

# Via vLLM for high-throughput inference
# vllm serve deepseek-ai/DeepSeek-V3 --tensor-parallel-size 8
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "Explain MLA attention"}]
)
```

---

## Key Takeaways

1. **MLA (V2):** 28× KV cache compression via low-rank latent projection — enables long context at fraction of memory cost
2. **Auxiliary-loss-free balancing (V3):** Dynamic bias adjustment achieves balanced routing without performance-degrading auxiliary losses
3. **V3 training cost:** 671B model for ~$5.6M — 10× cheaper than comparable open models, proving aggressive efficiency
4. **R1 pure RL:** Reasoning capabilities emerge from GRPO with rule-based rewards only — no human-labeled reasoning steps needed
5. **GRPO vs PPO:** No critic network in GRPO saves ~50% compute; group-relative advantage is more stable
6. **Distillation:** R1-Distill-32B matches o1-mini — reasoning can be compressed into dense models from MoE teachers

## References

- DeepSeek AI (2024) — DeepSeek-V2 (arXiv:2405.04434)
- DeepSeek AI (2024) — DeepSeek-V3 Technical Report (arXiv:2412.19437)
- DeepSeek AI (2025) — DeepSeek-R1: Incentivizing Reasoning via RL (arXiv:2501.12948)
- Shao et al. (2024) — DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO origin)
- HuggingFace Model Hub — deepseek-ai/DeepSeek-V3, deepseek-ai/DeepSeek-R1
