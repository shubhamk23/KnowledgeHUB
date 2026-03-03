---
title: "GPT Series: GPT-4, GPT-4o & o1/o3 Reasoning Models"
slug: gpt-series
summary: "OpenAI's GPT evolution — GPT-4's scaling breakthroughs, GPT-4o's unified multimodal architecture, and o1/o3's test-time compute scaling via reinforcement learning."
tags: ["GPT-4", "GPT-4o", "o1", "o3", "OpenAI", "RLHF", "reasoning", "multimodal", "chain-of-thought"]
visibility: public
---

# GPT Series: GPT-4, GPT-4o & o1/o3 Reasoning Models

## Overview

OpenAI's GPT (Generative Pre-trained Transformer) series represents the dominant commercially deployed LLM lineage. From GPT-3's 175B-parameter in-context learning breakthrough to GPT-4o's unified multimodal architecture to o1/o3's novel test-time compute scaling, each generation introduced paradigm-shifting innovations.

---

## GPT-4 (March 2023)

### Architecture

OpenAI did not disclose exact architecture details for competitive reasons. Key confirmed and widely-estimated facts:

- **Architecture type:** Transformer decoder (autoregressive)
- **Estimated parameters:** ~1.8 trillion in a Mixture of Experts configuration (unofficial; OpenAI unconfirmed)
- **Training data:** Publicly available internet data + licensed third-party sources, cutoff ~September 2021
- **Post-training:** RLHF with PPO + additional safety reward signal

**Predictable scaling:** GPT-4's most notable engineering achievement was infrastructure that predicts final model capabilities using runs at 1/1000th the compute — enabling confident investment before the full training run.

### Benchmark Performance

| Benchmark | GPT-4 | GPT-3.5 | Human |
|-----------|-------|---------|-------|
| MMLU (5-shot) | 86.4% | 70.0% | ~89% |
| HumanEval (code) | 67.0% | 48.1% | — |
| Bar Exam | ~90th %ile | ~40th %ile | ~60th %ile |
| SAT Math | 700/800 | 590/800 | — |
| GRE Quantitative | 163/170 | 157/170 | — |
| AP Biology | 5/5 | 4/5 | — |

GPT-4 was the first model to pass professional exams (bar, USMLE) at competitive human percentiles.

### RLHF Training Pipeline

**Step 1 — Supervised Fine-Tuning (SFT):**
Fine-tune the base model on high-quality human demonstrations:
```
(prompt, ideal_response) pairs → minimize cross-entropy loss
```

**Step 2 — Reward Model Training:**
Collect human preferences on pairs of model outputs:
```
r_{w} > r_{l}  (preferred vs. rejected response)
Loss: -log σ(r_θ(x, y_w) - r_θ(x, y_l))
```

**Step 3 — PPO Fine-Tuning:**
Maximize reward while preventing distribution collapse with KL penalty:
$$\max_{\pi_\theta} \mathbb{E}[r_\phi(x, y)] - \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{SFT}}]$$

**Safety reward:** An additional safety-specific reward was incorporated during RLHF to reduce harmful outputs — run in parallel with the helpfulness reward.

### GPT-4V (Vision) — September 2023

GPT-4 extended with a vision encoder:
- Image + text input → text output
- Analyzes charts, diagrams, documents, photos
- Known limitation: merges nearby text components in images; not for medical diagnosis

---

## GPT-4o: Unified Omni Architecture (May 2024)

### The Architectural Shift

Previous vision-language models (including GPT-4V) attached vision adapters to text LLMs — a post-hoc combination. GPT-4o ("o" = omni) is **trained end-to-end across all modalities simultaneously:**

```
Before (GPT-4V style):
Image → [Separate ViT Encoder] → image tokens → [Text LLM] → text output

GPT-4o (unified):
Text  ─┐
Image ─┼─→ [Single Unified Transformer] → Text / Audio / Image
Audio ─┘
```

**Key consequence:** The model learns deep cross-modal relationships — not just "describe this image" but genuinely reasoning about what it hears, sees, and reads in unified context.

### Technical Specifications

- **Input:** Text, image, audio, video (any combination)
- **Output:** Text, audio, image (any combination)
- **Audio latency:** 232ms minimum, 320ms average — human conversational speed
- **Context:** Up to 128K tokens
- **Inference:** State-of-the-art efficiency via unified tokenization

**Unified tokenization:** All modalities are tokenized into the same token space. Images are divided into patches and tokenized at same cost as text tokens. This enables common processing across all inputs.

### Realtime API

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# WebSocket realtime connection
async def realtime_conversation():
    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview"
    ) as connection:
        await connection.session.update(session={
            "modalities": ["text", "audio"],
            "voice": "alloy",
        })

        # Send audio input
        await connection.conversation.item.create(item={
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "audio": audio_bytes_base64}]
        })
        await connection.response.create()

        # Receive streaming audio output
        async for event in connection:
            if event.type == "response.audio.delta":
                play_audio(event.delta)
```

### GPT-4o Benchmark Performance

| Benchmark | GPT-4o | GPT-4 Turbo | Claude 3.5 Sonnet |
|-----------|--------|-------------|-------------------|
| MMLU | 88.7% | 86.4% | 90.4% |
| HumanEval | 90.2% | 87.1% | 92.0% |
| MATH | 76.6% | 72.6% | 71.1% |
| MMMU (multimodal) | 69.1% | 56.8% | 70.4% |

---

## o1 / o3: Test-Time Compute Scaling (2024-2025)

### The New Paradigm

Standard LLMs scale with **training compute** — more parameters + more data = better performance. The o-series introduces a second scaling axis: **test-time compute** — more thinking during inference = better performance.

$$\text{Performance} = f(\text{train compute}) \times g(\text{test-time compute})$$

Both $f$ and $g$ are monotonically increasing. The o-series dramatically expands $g$.

### Training: Chain-of-Thought via Reinforcement Learning

Unlike standard RLHF (which optimizes final answers), o-series models are trained to produce and refine **intermediate reasoning steps** via RL:

**Training process:**
1. Generate solution attempts with extended chain-of-thought
2. Verify correctness of final answer against ground truth
3. Reward signal: correctness only (no human-labeled reasoning steps)
4. RL (PPO variant) trains the model to improve its reasoning process

**What the model learns:**
- Writing intermediate verification steps
- Self-correcting when it detects an error
- Switching strategies when stuck
- Breaking complex steps into simpler sub-steps
- Running "mental code execution" to verify

```
Problem: Prove that the square root of 2 is irrational.

[Thinking]
Let me assume √2 = p/q where p, q are coprime integers.
Then 2 = p²/q², so p² = 2q².
This means p² is even, so p must be even: p = 2k.
Then (2k)² = 2q², so 4k² = 2q², so q² = 2k².
This means q² is even, so q is even.
But if both p and q are even, they share factor 2 — contradicts coprimeness!
Therefore √2 cannot be rational. ∎

[Answer]
√2 is irrational. Proof by contradiction: assuming √2 = p/q (coprime)
leads to both p and q being even — a contradiction. ∎
```

### Two Dimensions of Test-Time Compute

**Sequential (depth):** Generate a longer chain-of-thought for a single problem:
$$\text{depth scaling}: \text{budget} = 1000, 5000, 20000 \text{ thinking tokens}$$

**Parallel (breadth):** Sample multiple solution paths, select best via majority vote or verifier:
$$\text{breadth scaling}: \text{sample } N \text{ solutions, aggregate}$$

Both show monotonic improvement with more compute — unlike training, there's no equivalent of overfitting.

### o1 Performance (September 2024)

| Benchmark | o1 | GPT-4o | Human Expert |
|-----------|----|---------|----|
| AIME 2024 | 83.3% | 13.4% | ~50% (AIME qualifier) |
| GPQA Diamond (PhD science) | 78.0% | 53.6% | 69.7% |
| Codeforces | 89th %ile | ~50th %ile | — |
| MATH | 94.8% | 76.6% | — |
| SWE-bench | 48.9% | 22.8% | — |

**AIME context:** The American Invitational Mathematics Examination is a competition for top US high school math students. 83.3% puts o1 in the top ~500 US students.

### o3 (April 2025) — Generation 2

Key improvements over o1:

| Aspect | o1 | o3 |
|--------|----|----|
| Reasoning accuracy | Baseline | +22.8% on hard tasks |
| Multimodal in CoT | ❌ | ✅ Images in thinking |
| Code verification | Limited | Writes + runs code in thinking |
| Strategy switching | Basic | Sophisticated |
| SWE-bench | 48.9% | 71.7% |
| ARC-AGI (novel reasoning) | 32% | 87.5% |

**o3 code verification:**
```
[Thinking]
Let me write a brute-force solution to verify my optimized approach:

# Brute force: O(n²)
def brute_force(arr):
    return max(arr[i]*arr[j] for i in range(len(arr))
                              for j in range(i+1, len(arr)))

# My optimized solution: O(n log n)
def optimized(arr):
    arr.sort()
    return max(arr[-1]*arr[-2], arr[0]*arr[1])

# Test on examples
assert brute_force([1,2,3]) == optimized([1,2,3]) == 6
assert brute_force([-3,-2,1]) == optimized([-3,-2,1]) == 6  # ✓
```

### o3-mini and o4-mini

**o3-mini (January 2025):**
- 39% error reduction on difficult real-world questions vs o1-mini
- Users prefer o3-mini 56% vs o1-mini
- Medium effort o3-mini ≈ o1 performance in math, coding, science

**o4-mini (April 2025):**
- Best performance on AIME 2024 and 2025
- Faster inference than o1 at o1+ quality
- Optimized for high-volume reasoning tasks

### API Usage for o-series

```python
from openai import OpenAI

client = OpenAI()

# o1 and o3 use 'reasoning_effort' instead of temperature
response = client.chat.completions.create(
    model="o3",
    messages=[
        {"role": "user", "content": "Find the optimal solution to the traveling salesman problem for these 5 cities..."}
    ],
    reasoning_effort="high"   # "low" | "medium" | "high"
)

# Access reasoning summary (not full trace)
print(response.choices[0].message.content)

# o4-mini for faster reasoning
response = client.chat.completions.create(
    model="o4-mini",
    messages=[{"role": "user", "content": "Solve this differential equation..."}],
    reasoning_effort="medium"
)
```

---

## GPT Model Family Comparison

| Model | Params | Context | Modalities | Best For |
|-------|--------|---------|------------|---------|
| GPT-4 | ~1.8T (est.) | 128K | Text + image | General, vision |
| GPT-4o | Unknown | 128K | Text+image+audio | Realtime, multimodal |
| GPT-4o-mini | Unknown | 128K | Text+image | Fast, cheap tasks |
| o1 | Unknown | 200K | Text | Hard reasoning |
| o3 | Unknown | 200K | Text+image | Frontier reasoning |
| o4-mini | Unknown | 200K | Text+image | Fast reasoning |

---

## Function Calling / Tool Use

GPT-4 and all o-series models support structured function calling:

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"}
                },
                "required": ["ticker"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the price of AAPL?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a function
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    import json
    args = json.loads(tool_call.function.arguments)
    print(f"Calling {tool_call.function.name}({args})")
```

---

## Key Takeaways

1. **GPT-4** — scaling breakthrough + predictable performance from small runs; passes professional exams
2. **GPT-4o** — first truly unified multimodal model (end-to-end, not adapters); enables realtime audio at 320ms latency
3. **o1/o3** — fundamentally new axis of scaling: test-time compute via RL-trained chain-of-thought
4. **o3 on ARC-AGI** — 87.5% on novel reasoning benchmark (designed to defeat pattern memorization) signals qualitatively new reasoning capability
5. **Function calling** makes GPT models the backbone of agentic systems and copilots
6. **Reasoning effort** parameter replaces temperature for o-series — control compute budget, not randomness

## References

- OpenAI (2023) — GPT-4 Technical Report (arXiv:2303.08774)
- OpenAI (2023) — GPT-4V(ision) System Card
- OpenAI (2024) — Hello GPT-4o
- OpenAI (2024) — Learning to Reason with LLMs (o1 release blog)
- OpenAI (2025) — o3 and o4-mini System Card
- Snell et al. (2024) — Scaling LLM Test-Time Compute Optimally (arXiv:2408.03314)
