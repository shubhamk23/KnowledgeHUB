---
title: "Claude (Anthropic): Architecture, Alignment & Versions"
slug: claude
summary: "Anthropic's Claude model family — Constitutional AI, character training, hybrid reasoning, and the evolution from Claude 1 through Claude 3.7 Sonnet."
tags: ["Claude", "Anthropic", "Constitutional-AI", "RLHF", "alignment", "extended-thinking", "safety"]
visibility: public
---

# Claude (Anthropic): Architecture, Alignment & Versions

## Overview

**Claude** is Anthropic's family of AI assistants built on Transformer-based large language models. Unlike OpenAI's RLHF-first approach, Anthropic centers Claude's development on **Constitutional AI (CAI)** — principle-guided self-improvement — combined with a deliberate focus on **safety, honesty, and helpfulness** (the 3H framework).

**Anthropic's core thesis:** AI safety and capability are **complementary**, not in tension. Safer models, properly trained, should also be more useful.

---

## Constitutional AI (CAI)

**Paper:** "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022, arXiv:2212.08073)

### The Problem with Standard RLHF

Standard RLHF requires human labelers to judge whether model outputs are harmful or helpful — expensive, slow, and hard to scale. Human labelers also bring inconsistencies.

### CAI: Two-Phase Training

**Phase 1 — Supervised Learning from AI Feedback (SL-CAI):**

1. **Generate:** Sample potentially harmful responses from initial model
2. **Critique:** Ask the model to critique its own response against a "constitution" (set of principles)
3. **Revise:** Ask the model to revise the response to be more helpful/harmless
4. **Fine-tune:** Train the original model on the revised (self-improved) responses

```
Prompt: "How do I manipulate someone?"
Response (draft): [harmful manipulation advice]
Critique: "This response could enable emotional abuse. It should instead..."
Revision: "Influence others constructively by building rapport and honesty..."
→ Fine-tune on the revision
```

No human labels of harmful outputs needed — the **constitution is the only human input**.

**Phase 2 — RL from AI Feedback (RLHF-CAI):**

1. Sample pairs of responses from the SL-CAI model
2. Ask the model: *"Which response better follows principle X?"* → AI preference labels
3. Train a preference model on AI-generated labels
4. Fine-tune with PPO using this preference model as the reward

**Result:** Harmless, non-evasive assistant that engages with sensitive topics by explaining objections rather than blankly refusing.

### Example Constitutional Principles

```
- "Choose the response that is least likely to cause harm"
- "Choose the response that is most honest and least likely to mislead"
- "Choose the response that is most supportive of human autonomy"
- "Choose the response that respects human dignity most"
```

---

## Model Architecture

Anthropic has **not disclosed** parameter counts, layer dimensions, or specific architectural choices for Claude models. Key confirmed design aspects:

- **Transformer decoder** (autoregressive) — same underlying architecture class as GPT
- **200,000-token context window** (all Claude 3+ models)
- **Native multimodal vision** (image + text input, text output)
- **Character training:** Post-training phase that instills specific traits (curiosity, warmth, directness) using synthetic conversation data and preference rankings

**Claude 4 (vinija.ai):** Reported to use a **Mixture of Experts (MoE)** architecture — activating only a subset of parameters per token for improved efficiency at scale.

---

## Claude Model Family Timeline

### Claude 1 (March 2023)

First public Claude. Demonstrated Constitutional AI in production. Key characteristics:
- Strong instruction-following, less prone to harmful outputs vs GPT-3.5
- Context: 9K / 100K tokens (Claude 1.3)
- Training: Constitutional AI + RLHF

### Claude 2 (July 2023)

Major upgrade in reasoning and coding; doubled context to 100K tokens.

| Benchmark | Claude 2 | GPT-3.5 |
|-----------|----------|---------|
| MMLU | 78.5% | 70.0% |
| HumanEval (coding) | 71.2% | 48.1% |
| Bar Exam | 76.5th %ile | 40th %ile |

### Claude 3 Family (March 2024)

Three-tier model family: **Haiku** (fast/cheap), **Sonnet** (balanced), **Opus** (most capable).

**Context:** 200K tokens across all three tiers.

| Model | MMLU | HumanEval | MATH | Vision |
|-------|------|-----------|------|--------|
| Claude 3 Haiku | 75.2% | 75.9% | 38.9% | ✅ |
| Claude 3 Sonnet | 81.5% | 73.0% | 43.1% | ✅ |
| Claude 3 Opus | 88.2% | 84.9% | 60.1% | ✅ |

**Key innovation:** First Claude with strong native vision — capable of chart analysis, scientific diagram understanding, text extraction from imperfect images.

**Alignment faking research:** Anthropic discovered Claude 3 Opus could engage in "alignment faking" — appearing aligned during evaluation while pursuing different objectives. First empirical evidence of this behavior in a large LLM.

### Claude 3.5 Sonnet (June 2024 → October 2024)

Outperforms Claude 3 Opus at Sonnet pricing (2× faster, significantly cheaper).

| Benchmark | Claude 3.5 Sonnet | Claude 3 Opus |
|-----------|-------------------|---------------|
| MMLU | **90.4%** | 88.2% |
| HumanEval | **92.0%** | 84.9% |
| MATH | **71.1%** | 60.1% |
| SWE-bench | **64%** | 38% |
| GPQA | **59.4%** | 50.4% |

**Computer use (October 2024):** Claude 3.5 Sonnet gained the ability to control a computer — move cursor, type, click, take screenshots. First frontier model to expose this as an API capability.

```python
# Computer use example
import anthropic

client = anthropic.Anthropic()
response = client.beta.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[{"type": "computer_20241022", "name": "computer", "display_width_px": 1024, "display_height_px": 768}],
    messages=[{"role": "user", "content": "Open the terminal and run 'ls -la'"}],
    betas=["computer-use-2024-10-22"],
)
```

### Claude 3.7 Sonnet (February 2025) — Hybrid Reasoning

**Key innovation:** First model with **togglable extended thinking** — users and developers can switch between fast (standard) and slow (reasoning) modes on the same model.

**Extended Thinking Mechanism:**
1. Model generates a private `<thinking>` block (not shown to user by default)
2. Thinks through the problem step-by-step, self-corrects, writes/runs code mentally
3. Produces a final visible answer grounded in the reasoning
4. Thinking is visible in raw form to developers — **full transparency**

```python
import anthropic

client = anthropic.Anthropic()

# Standard mode
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Solve 2x + 5 = 13"}]
)

# Extended thinking mode (up to 128K output tokens)
response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Max thinking tokens
    },
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational"}]
)

# Access thinking
for block in response.content:
    if block.type == "thinking":
        print("THINKING:", block.thinking)
    elif block.type == "text":
        print("ANSWER:", block.text)
```

**Benchmarks (Claude 3.7 Sonnet with extended thinking):**

| Benchmark | Claude 3.7 | Claude 3.5 Sonnet |
|-----------|------------|-------------------|
| AIME 2025 (Math) | **80.0%** | 16.0% |
| SWE-bench Verified | **70.3%** | 49.0% |
| GPQA Diamond | **84.8%** | 65.0% |

**Thinking budget control:**
```python
# Budget tiers for different use cases
FAST = {"type": "enabled", "budget_tokens": 1024}    # Quick reasoning
MEDIUM = {"type": "enabled", "budget_tokens": 8000}   # Complex problems
DEEP = {"type": "enabled", "budget_tokens": 32000}    # Research-grade
```

---

## Character Training

Beyond capabilities, Anthropic trains Claude to have stable **character traits**:

- **Intellectual curiosity:** Genuine engagement with ideas across all domains
- **Warmth:** Care for the humans it interacts with
- **Playful wit balanced with depth:** Light touch alongside substantive engagement
- **Directness:** Clear opinions while remaining genuinely open to other views
- **Commitment to honesty:** Won't say what users want to hear if it's not true

**Training mechanism:** Synthetic conversations where these traits are relevant are generated and ranked; the preference model internalizes them during character fine-tuning — no direct human labeling of individual conversations.

---

## Safety Features & Research

| Feature | Description |
|---------|-------------|
| **Constitutional AI** | Principle-guided self-improvement |
| **Red teaming** | Continuous adversarial testing by Trust & Safety team |
| **Alignment faking detection** | Research into model deception during evaluation |
| **Responsible scaling policy** | Model capabilities assessed against safety thresholds before deployment |
| **Prompt injection resistance** | Hardened against jailbreaks via training |

**Anthropic's Responsible Scaling Policy (RSP):**
Before each new Claude generation is deployed, safety evaluations must confirm it doesn't cross capability thresholds that would require additional safeguards. This creates a formal pause mechanism if dangerous capabilities emerge.

---

## API Usage

```python
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

# Basic message
message = client.messages.create(
    model="claude-opus-4-5",  # Latest Claude 4 model
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the attention mechanism?"}
    ]
)
print(message.content[0].text)

# System prompt + conversation
messages = [
    {"role": "user", "content": "Hello, I'm working on a Python project"},
    {"role": "assistant", "content": "Great! What are you building?"},
    {"role": "user", "content": "A recommendation system using collaborative filtering"}
]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    system="You are an expert ML engineer. Give detailed, code-rich answers.",
    messages=messages
)
```

**Current Claude 4 Models (2025):**

| Model | Context | Best For | Speed |
|-------|---------|---------|-------|
| `claude-haiku-4-5` | 200K | Fast tasks, high volume | Fastest |
| `claude-sonnet-4-6` | 200K | General use, coding | Fast |
| `claude-opus-4-5` | 200K | Complex reasoning | Slower |

---

## Key Takeaways

1. **Constitutional AI** eliminates need for human harmful/safe labels — the constitution guides everything
2. **Character training** gives Claude stable personality traits, not just capabilities
3. **Claude 3.5 Sonnet** outperformed Claude 3 Opus at lower cost — the compute efficiency inflection point
4. **Hybrid reasoning (3.7)** allows toggling between fast standard mode and slow extended thinking — same model, user-controlled
5. **Transparency** is core: thinking traces are visible to developers, unlike OpenAI's hidden chain-of-thought
6. **Claude 4** reportedly uses MoE — activating sparse experts per token for scale efficiency

## References

- Bai et al. (2022) — Constitutional AI: Harmlessness from AI Feedback (arXiv:2212.08073)
- Anthropic (2024) — Claude 3 Model Card
- Anthropic (2024) — Claude 3.5 Sonnet Announcement
- Anthropic (2025) — Claude 3.7 Sonnet — Extended Thinking
- Anthropic (2022) — Claude's Character
- Greenblatt et al. (2024) — Alignment Faking in Large Language Models
