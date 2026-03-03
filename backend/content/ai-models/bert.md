---
title: "BERT: Bidirectional Encoder Representations from Transformers"
slug: bert
summary: "How BERT's masked language modeling and bidirectional attention revolutionized NLP pre-training and transfer learning."
tags: ["BERT", "bidirectional", "masked-LM", "NLP", "pre-training", "fine-tuning", "encoder"]
visibility: public
---

# BERT: Bidirectional Encoder Representations from Transformers

## Overview

**BERT** (Devlin et al., Google, 2018) revolutionized NLP by demonstrating that deeply bidirectional Transformer encoders pre-trained on large text corpora dramatically outperform previous approaches on almost all NLP benchmarks.

**Key innovation:** Unlike GPT (left-to-right) or ELMo (shallow bidirectionality), BERT processes the entire sequence simultaneously — every token attends to all other tokens in both directions.

---

## Architecture

BERT uses the **Transformer encoder** stack (not decoder):

| Model | Layers (L) | Hidden (H) | Attention Heads (A) | Parameters |
|-------|-----------|------------|---------------------|-----------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

**Input representation:**

$$\text{Input} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}$$

```
[CLS] The cat sat on [MASK] [SEP] It was warm . [SEP]
  ↑                              ↑
CLS token                    Sentence separator
(classification head)
```

- **[CLS]:** Special classification token — its output is used for sequence-level tasks
- **[SEP]:** Separator between sentences in pair tasks
- **Segment embeddings:** $E_A$ or $E_B$ indicating which sentence each token belongs to

---

## Pre-training Objectives

### 1. Masked Language Modeling (MLM)

Randomly mask 15% of input tokens, predict the original token from context:

```
Input:  The cat [MASK] on the mat
Target: The cat sat  on the mat
```

Of the 15% selected tokens:
- 80%: Replace with `[MASK]` token
- 10%: Replace with a random token
- 10%: Keep original (but still predict)

This prevents the model from simply always predicting `[MASK]`.

**Loss:**
$$\mathcal{L}_{\text{MLM}} = -\sum_{m \in \text{masked}} \log p(x_m | \mathbf{x}_{\setminus m})$$

**Why bidirectionality helps:** To predict `sat`, BERT can use both "The cat" (left) AND "on the mat" (right) — much more context than GPT's left-only view.

### 2. Next Sentence Prediction (NSP)

Given two sentences A and B, predict if B follows A in the original text:

```
A: "The man went to the store."
B: "He bought some milk."  → Label: IsNext

A: "The man went to the store."
B: "Penguins are flightless birds." → Label: NotNext
```

**Purpose:** Learn sentence-level relationships for tasks like question answering and NLI.

**Note:** Later research showed NSP provides minimal benefit — RoBERTa dropped it entirely.

---

## Fine-tuning BERT

BERT is pre-trained once, then fine-tuned on downstream tasks by adding a task-specific head:

### Sequence Classification

Add a linear layer on top of `[CLS]` token:
$$\hat{y} = \text{softmax}(W_C \mathbf{h}_{\text{[CLS]}})$$

### Token Classification (NER, POS)

Add a linear layer on top of each token representation:
$$\hat{y}_i = \text{softmax}(W_T \mathbf{h}_i)$$

### Question Answering (SQuAD)

Predict start and end positions of the answer span:
$$P_{\text{start}}(i) = \text{softmax}(\mathbf{s}^T \mathbf{h}_i), \quad P_{\text{end}}(i) = \text{softmax}(\mathbf{e}^T \mathbf{h}_i)$$

```python
from transformers import BertForQuestionAnswering, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

inputs = tokenizer(question, context, return_tensors='pt')
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
```

---

## BERT Variants & Evolution

### RoBERTa (Liu et al., 2019 — Facebook)

"A Robustly Optimized BERT Pretraining Approach"

Key changes from BERT:
- **No NSP:** Removed — hurt downstream performance
- **Dynamic masking:** Different masks per epoch instead of static
- **Larger batches:** 8192 vs 256
- **More data:** 160GB vs 16GB
- **Longer training:** 10× more updates

**Result:** Significantly outperforms BERT on all benchmarks.

### ALBERT (Lan et al., 2019 — Google)

"A Lite BERT for Self-supervised Learning of Language Representations"

Key changes:
- **Factorized embedding:** Separate embedding size from hidden size
- **Cross-layer parameter sharing:** Share weights across all transformer layers
- **SOP (Sentence Order Prediction):** Replaces NSP — predict if two sentences are swapped

**Result:** 18× fewer parameters than BERT-Large, competitive performance.

### DistilBERT (Sanh et al., 2019 — HuggingFace)

Knowledge distillation of BERT-Base:
- 40% fewer parameters (66M vs 110M)
- 60% faster inference
- 97% of BERT performance on GLUE

### DeBERTa (He et al., 2020 — Microsoft)

"Decoding-enhanced BERT with Disentangled Attention"
- Separate content and position embeddings
- Enhanced mask decoder
- State-of-the-art on many NLU benchmarks

---

## BERT for Embeddings

BERT produces contextualized embeddings — the same word gets different vectors depending on context:

```python
# "bank" in two contexts
sentence1 = "I went to the bank to deposit money."
sentence2 = "We sat on the river bank."

# Different embeddings for "bank" in each sentence
emb1 = bert_encode(sentence1)["bank"]  # financial institution
emb2 = bert_encode(sentence2)["bank"]  # river bank
```

**Sentence Embeddings:**
Mean pooling of all token embeddings works but isn't optimal.
**Sentence-BERT (SBERT):** Fine-tuned with siamese networks for sentence similarity:

$$\text{sentence\_emb} = \text{mean}(\mathbf{h}_1, \mathbf{h}_2, \ldots, \mathbf{h}_T)$$

---

## BERT vs GPT

| Aspect | BERT | GPT |
|--------|------|-----|
| Architecture | Encoder only | Decoder only |
| Attention | Bidirectional (full) | Causal (left-only) |
| Pre-training | MLM + NSP | Autoregressive LM |
| Primary use | Understanding, embeddings | Generation |
| Fine-tuning | Task-specific heads | Instruction tuning / few-shot |
| Context at token | Full sequence | Past tokens only |
| Best for | Classification, NER, QA | Generation, chatbots |

---

## Practical Usage

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

# CLS token for classification
cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [batch, 768]

# Mean pooling for sentence embedding
token_embeddings = outputs.last_hidden_state  # shape: [batch, seq_len, 768]
attention_mask = inputs['attention_mask']
sentence_embedding = (token_embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
```

---

## Key Takeaways

1. **BERT = Transformer encoder with bidirectional attention** — sees full context for each token
2. **MLM pre-training** creates rich contextual representations by predicting masked tokens
3. **Fine-tuning** only requires adding a small task-specific head — minimal changes to pre-trained weights
4. **RoBERTa** is the practical default over BERT (no NSP, more data, longer training)
5. **DistilBERT** for production when speed matters; DeBERTa for maximum accuracy
6. **BERT family is best for understanding tasks** — encoding, classification, NER, QA

## References

- Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers
- Liu et al. (2019) — RoBERTa: A Robustly Optimized BERT Pretraining Approach
- Lan et al. (2019) — ALBERT
- Sanh et al. (2019) — DistilBERT
- Reimers & Gurevych (2019) — Sentence-BERT
