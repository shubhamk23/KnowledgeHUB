---
title: "LLM Alignment: RLHF, DPO & Constitutional AI"
slug: llm-alignment
summary: "How to align large language models with human values using RLHF, Direct Preference Optimization, and Constitutional AI."
tags: ["RLHF", "DPO", "alignment", "constitutional-AI", "safety", "PPO"]
visibility: public
---

# LLM Alignment: RLHF, DPO & Constitutional AI

## Overview

**Alignment** refers to ensuring that AI systems behave in accordance with human values and intentions. For LLMs, a pre-trained model has learned to predict text — but not necessarily to be helpful, harmless, or honest (the **3H** framework from Anthropic).

**The alignment problem:** A model trained to maximize next-token prediction log-likelihood will happily:
- Generate harmful content if it appears in training data
- Hallucinate to sound confident and coherent
- Comply with harmful requests (RLHF sycophancy risk)

---

## The 3H Framework

| Property | Definition | Example Failure |
|----------|-----------|-----------------|
| **Helpful** | Assists with legitimate user requests | Refuses safe requests |
| **Harmless** | Avoids generating harmful content | Provides synthesis routes for dangerous substances |
| **Honest** | Truthful, calibrated, transparent | Hallucinating confidently |

These objectives can conflict: a maximally helpful model might assist with harmful requests; a maximally harmless model might refuse legitimate ones.

---

## InstructGPT / ChatGPT Training Pipeline

OpenAI's 3-step pipeline that transformed GPT-3 into ChatGPT:

### Step 1: Supervised Fine-Tuning (SFT)

Fine-tune pre-trained LLM on high-quality (prompt, response) pairs written by human annotators:

$$\mathcal{L}_{\text{SFT}} = -\sum_t \log \pi_{\theta}(y_t | x, y_{<t})$$

Gives the model a "shape" of the desired behavior.

### Step 2: Reward Model Training

Train a reward model $r_\phi$ to score responses using human preference data:

**Data collection:**
1. Sample $K$ responses from SFT model for each prompt
2. Human raters rank responses: $y_w \succ y_l$ (preferred vs rejected)

**Reward model loss (Bradley-Terry model):**
$$\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]$$

The reward model learns: human-preferred responses should score higher.

### Step 3: PPO Fine-Tuning (RL)

Optimize the language model policy $\pi_\theta$ to maximize expected reward while staying close to the SFT model:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x, y)] - \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{SFT}}]$$

The KL penalty $\beta \cdot D_{\text{KL}}$ prevents **reward hacking** (the model finding high-reward but degenerate responses).

**PPO clipping objective:**
$$\mathcal{L}_{\text{PPO}} = \min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio and $A_t$ is the advantage.

---

## Direct Preference Optimization (DPO)

**Paper:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)

### Motivation

RLHF with PPO is:
- Complex (3 models: policy, value, reward)
- Unstable (PPO hyperparameter sensitivity)
- Computationally expensive

**DPO key insight:** The optimal policy $\pi^*$ under RLHF has a closed-form relationship to the reward model:

$$r^*(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

This lets us bypass training a separate reward model.

### DPO Loss

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**Intuition:** Increase likelihood of preferred $y_w$ relative to reference; decrease likelihood of rejected $y_l$ relative to reference.

### RLHF vs DPO

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Models required | 4 (actor, critic, reward, ref) | 2 (policy, ref) |
| Stability | Often unstable | More stable |
| Compute | High | ~2× SFT |
| Performance | State-of-the-art | Competitive, sometimes better |
| Reward hacking | Possible | Less likely |
| Training data | Prompts + ranked responses | Preference pairs only |

### DPO Variants

- **IPO** (Identity Preference Optimization) — avoids overfitting to preference data
- **KTO** (Kahneman-Tversky Optimization) — uses binary feedback instead of pairwise
- **ORPO** (Odds Ratio Preference Optimization) — single-stage, no reference model needed
- **SimPO** — simplified DPO without reference model

---

## Constitutional AI (Claude)

**Paper:** "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)

### Motivation

Human labeling for RLHF is expensive and hard to scale. Can we use the AI itself to critique and improve responses?

### Phase 1: SL-CAI (Supervised Learning from AI Feedback)

1. **Red-teaming:** Generate potentially harmful responses
2. **Critique:** Ask Claude to critique the response against a set of principles (the "constitution")
3. **Revision:** Ask Claude to revise the response to be more helpful/harmless
4. **Fine-tune** on the revised responses

```
Prompt: "How do I pick a lock?"
Response: [potentially harmful]
Critique: "This response provides detailed instructions that could facilitate burglary. It should..."
Revision: "Lock picking has legitimate uses like locksmithing. For legal purposes..."
```

### Phase 2: RL-CAI (RL from AI Feedback)

Use AI-generated preference labels instead of human labels:
1. Sample two responses for each prompt
2. Ask the AI: "Which response is more harmless according to principle X?"
3. Use these AI labels to train the reward model
4. Fine-tune with PPO as in standard RLHF

**Advantage:** Scales without expensive human labeling; principles are explicit and auditable.

### Example Constitutional Principles

- "Choose the response that is least likely to contain harmful or unethical content"
- "Choose the response that is most supportive of human autonomy and individual freedoms"
- "Choose the response that is most honest and least likely to mislead"

---

## RLAIF (RL from AI Feedback)

Generalization of Constitutional AI — use AI models as preference judges:

```
Query + Response A + Response B → AI Judge → Preference Label → Reward Model
```

**Google's approach (Llama 2 paper):** Used GPT-4 to generate preference labels for training Llama 2-Chat.

---

## Reward Hacking & Alignment Failures

### Goodhart's Law

"When a measure becomes a target, it ceases to be a good measure."

In RLHF: the model finds **reward-maximizing but degenerate** responses:
- Very long responses (reward models favor length)
- Sycophantic agreement ("Great question! Absolutely...")
- Confident hallucinations (sound authoritative = rewarded)

### Mitigation

| Technique | How |
|-----------|-----|
| KL penalty | Stay close to reference policy |
| Ensemble reward models | Harder to simultaneously hack multiple |
| Process reward | Reward reasoning steps, not just final answer |
| Constitutional review | Post-hoc harmlessness check |

---

## Process Reward Models (PRM)

Standard RLHF uses **outcome reward** (final answer score). PRMs score each reasoning step:

$$r_{\text{total}} = \sum_{t=1}^{T} r_{\text{PRM}}(\text{step}_t)$$

**OpenAI's "Let's Verify Step by Step" (2023):** PRMs significantly improve math reasoning by penalizing wrong intermediate steps.

---

## Key Takeaways

1. **Alignment = making LLMs helpful, harmless, and honest** — these 3Hs can conflict
2. **RLHF pipeline:** SFT → Reward Model → PPO fine-tuning
3. **DPO** eliminates the reward model — directly optimizes preference pairs with a simpler loss
4. **Constitutional AI** scales RLHF using AI-generated critique and revision
5. **Reward hacking** is a real risk — KL penalties, diverse reward models, PRMs help
6. **Process reward models** improve complex reasoning by scoring intermediate steps

## References

- Ouyang et al. (2022) — Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
- Rafailov et al. (2023) — Direct Preference Optimization
- Bai et al. (2022) — Constitutional AI: Harmlessness from AI Feedback
- Lightman et al. (2023) — Let's Verify Step by Step (Process Reward Models)
