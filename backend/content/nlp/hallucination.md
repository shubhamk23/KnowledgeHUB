---
title: "LLM Hallucination: Causes, Detection & Mitigation"
slug: hallucination
summary: "Why LLMs generate false information, how to detect it, and proven strategies to reduce hallucinations in production systems."
tags: ["hallucination", "LLM", "factuality", "reliability", "RAG", "RLHF"]
visibility: public
---

# LLM Hallucination: Causes, Detection & Mitigation

## Overview

**Hallucination** in LLMs refers to the generation of content that is fluent and confident-sounding but factually incorrect, unsupported, or fabricated. Unlike typos or grammatical errors, hallucinations are particularly dangerous because they appear authoritative.

**Scale of the problem:** Studies show GPT-4 hallucinates in ~3-20% of responses depending on domain; domain-specific models can hallucinate at rates up to 80% in low-resource areas.

---

## Taxonomy of Hallucinations

### By Faithfulness

| Type | Definition | Example |
|------|-----------|---------|
| **Intrinsic Hallucination** | Contradicts the provided source/context | Summary contradicts the article being summarized |
| **Extrinsic Hallucination** | Adds unverifiable information not in source | Invents statistics not in the source document |

### By Cause

| Type | Description |
|------|-------------|
| **Factual hallucination** | States false facts ("Einstein won the Nobel Prize in 1905") |
| **Entity hallucination** | Invents people, places, papers, citations |
| **Temporal hallucination** | Wrong dates, order of events |
| **Numerical hallucination** | Wrong numbers, statistics |
| **Logical hallucination** | Valid-sounding but logically inconsistent reasoning |
| **Instruction hallucination** | Claims to have done something it didn't |

### Severity Levels

```
Low:    Minor inaccuracies (wrong date by 1 year)
Medium: Wrong facts that could mislead (wrong dosage)
High:   Dangerous fabrications (false medical/legal advice)
```

---

## Root Causes

### 1. Training Data Issues

- **Misinformation in corpus:** LLMs trained on internet data absorb false claims
- **Knowledge cutoff:** No information after training date
- **Frequency bias:** Rare facts underrepresented → model fills gaps with common patterns
- **Memorization vs generalization:** Model may not have memorized specific facts

### 2. Decoding Process

The autoregressive generation process maximizes:
$$p(y_t | y_{<t}, x) \propto \exp(\text{logit}(y_t) / T)$$

**Temperature $T$:** Higher T → more random → more likely hallucination
**Beam search:** Can propagate early errors through the beam

The model always produces *something* — it has no "I don't know" token by default.

### 3. Overconfidence in Training

- **RLHF optimization:** Reward models may reward confident, fluent responses over accurate ones
- **Instruction following pressure:** Models trained to answer everything may fabricate rather than abstain
- **Sycophancy:** Models trained with human feedback learn to agree rather than correct

### 4. Context Window Limitations

- **Lost in the middle:** Models attend poorly to content in the middle of long contexts
- **Conflicting context:** When documents disagree, models may synthesize inaccurate middle ground

---

## Detection Methods

### 1. Consistency-Based Detection

Ask the same question multiple ways, check if answers agree:

```python
def check_consistency(question, model, n_samples=5):
    answers = [model.generate(question, temperature=0.8)
               for _ in range(n_samples)]
    # High variance in answers → likely hallucination
    return compute_agreement_score(answers)
```

**SelfCheckGPT:** Samples multiple stochastic generations; if facts are inconsistent across samples, likely hallucinated.

$$\text{hallucination\_score}(s) = 1 - \frac{1}{N} \sum_{i=1}^{N} \text{NLI}(s, g_i)$$

Where $g_i$ are sampled generations and NLI is Natural Language Inference.

### 2. Retrieval-Based Verification

1. Extract factual claims from the response
2. Retrieve evidence for each claim
3. Check claim entailment with retrieved evidence

```
Response → Claim Extraction → Retrieval → NLI Check → Factuality Score
```

Tools: **FActScore**, **FEVER**, **FactChecking pipelines**

### 3. Uncertainty Estimation

**Token-level uncertainty:**
$$\text{uncertainty}(y_t) = -\log p(y_t | y_{<t}, x) = H(y_t)$$

High entropy at token level → model is uncertain → potential hallucination.

**Semantic entropy:**
Cluster semantically equivalent answers; if no dominant cluster → high hallucination risk.

### 4. NLI-Based Detection

Use a Natural Language Inference model to check if response is entailed by source:
```
Premise: [source document]
Hypothesis: [model response claim]
Label: Entailed / Neutral / Contradicted
```

**Contradicted** → definite hallucination
**Neutral** → extrinsic/unverifiable claim

---

## Mitigation Strategies

### 1. Retrieval-Augmented Generation (RAG)

Ground responses in retrieved documents:
```
User: "What is the capital of Australia?"
Retrieved: "Australia's capital is Canberra, established in 1913..."
Prompt: "Answer ONLY based on context. [context] Question: ..."
```

**Impact:** Reduces factual hallucinations by 30-60% depending on domain.

**Limitations:** Only works for knowledge available in the retrieval corpus.

### 2. Chain-of-Thought (CoT) Prompting

Encourage explicit reasoning before answering:
```
"Let me think step by step:
1. Australia is a country in Oceania.
2. Many people confuse Sydney (largest city) with the capital.
3. The actual capital is Canberra.
Answer: Canberra"
```

CoT reduces hallucinations on complex reasoning tasks by forcing intermediate verifiable steps.

### 3. RLHF with Truthfulness Rewards

Train a reward model that scores **accuracy** not just **helpfulness**:
```
Reward = α·Helpfulness + β·Truthfulness + γ·Harmlessness
```

**Constitutional AI (Anthropic):**
1. Generate response
2. Critique against truthfulness principles
3. Revise response
4. Train on critiqued outputs

### 4. Direct Preference Optimization (DPO)

Train directly on preference pairs where truthful answers are preferred:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} \right) \right]$$

Where $y_w$ = preferred (truthful) output, $y_l$ = rejected (hallucinated) output.

### 5. Calibration and Abstention

Train models to express uncertainty:
- **Verbalized uncertainty:** "I'm not sure, but..." or "I don't have information about..."
- **Calibration training:** Align confidence scores with actual accuracy

```python
# Detect low-confidence responses
if response_confidence < threshold:
    return "I'm not confident enough to answer this accurately."
```

### 6. Self-Consistency Decoding

Sample multiple outputs and return the most consistent answer:

```python
answers = [model.generate(prompt, temperature=0.7) for _ in range(10)]
final_answer = majority_vote(answers)  # or clustering-based consensus
```

Particularly effective for mathematical and factual questions.

### 7. Fact-Checked Fine-Tuning

Fine-tune on datasets where facts have been verified:
- **TruthfulQA** — adversarial questions that expose common misconceptions
- **FEVER** — fact-checking dataset
- **HaluEval** — hallucination evaluation benchmark

### 8. Structured Output with Citations

Force the model to cite sources for every factual claim:
```
Format: [Claim] [Source: doc_id, page_x]

"The Eiffel Tower is 330 meters tall [Source: britannica.com/eiffel-tower]"
```

---

## Evaluation Benchmarks

| Benchmark | Task | Focus |
|-----------|------|-------|
| **TruthfulQA** | QA on misconception topics | Truthfulness vs parroting |
| **HaluEval** | Hallucination detection | Classification accuracy |
| **FActScore** | Long-form generation | Claim-level factuality |
| **FEVER** | Fact verification | Source grounding |
| **SummEval** | Summarization | Faithfulness |

---

## Production Checklist

```
✅ Use RAG for knowledge-intensive tasks
✅ Set temperature ≤ 0.3 for factual tasks
✅ Add "If unsure, say 'I don't know'" to system prompt
✅ Implement retrieval verification post-generation
✅ Log and monitor cases where model abstains
✅ Use smaller, specialized models for narrow domains
✅ Human-in-the-loop for high-stakes outputs
✅ Evaluate on TruthfulQA and domain-specific benchmarks
```

---

## Key Takeaways

1. **Hallucination is fundamental**, not a simple bug — it emerges from language modeling objectives
2. **Two types:** intrinsic (contradicts source) and extrinsic (unverifiable addition)
3. **Root causes:** training data noise, decoding dynamics, RLHF sycophancy, context limits
4. **RAG** is the most practical mitigation for factual hallucination
5. **CoT + self-consistency** reduces reasoning hallucinations significantly
6. **Calibration** — train models to say "I don't know" — is underutilized but effective
7. **Evaluate explicitly:** TruthfulQA, FActScore, SelfCheckGPT for systematic measurement

## References

- Maynez et al. (2020) — On Faithfulness and Factuality in Abstractive Summarization
- Manakul et al. (2023) — SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection
- Min et al. (2023) — FActScore: Fine-grained Atomic Evaluation of Factual Precision
- OpenAI (2023) — GPT-4 Technical Report (hallucination analysis)
- Anthropic (2022) — Constitutional AI: Harmlessness from AI Feedback
