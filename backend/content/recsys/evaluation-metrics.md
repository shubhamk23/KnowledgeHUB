---
title: "Recommender System Evaluation Metrics"
slug: evaluation-metrics
summary: "NDCG, MAP, Precision@k, Recall@k, AUC, and MRR — how to measure and compare recommender system quality."
tags: ["NDCG", "MAP", "AUC", "precision", "recall", "MRR", "evaluation", "recommendation"]
visibility: public
---

# Recommender System Evaluation Metrics

## Overview

Evaluating recommender systems requires metrics that capture:
1. **Accuracy** — are recommendations relevant?
2. **Ranking quality** — are the best items at the top?
3. **Coverage** — diversity of recommendations
4. **Novelty/Serendipity** — are recommendations interesting?

Most production systems use **offline** metrics on historical data + **online** A/B tests for final decisions.

---

## Binary Relevance Metrics

### Precision@k

Of the top-$k$ items recommended, what fraction are relevant?

$$\text{Precision@k} = \frac{|\text{Recommended}_k \cap \text{Relevant}|}{k}$$

**Example:** Recommend 10 items, 4 are relevant → P@10 = 0.4

**Limitation:** Doesn't account for the order within the top-k list.

### Recall@k

Of all relevant items, what fraction appear in the top-$k$?

$$\text{Recall@k} = \frac{|\text{Recommended}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$

**Example:** 12 relevant items, 4 appear in top-10 → R@10 = 0.33

**Use when:** Evaluating retrieval stages (high recall is the goal).

### F1@k

Harmonic mean of Precision@k and Recall@k:

$$F1@k = \frac{2 \cdot \text{P@k} \cdot \text{R@k}}{\text{P@k} + \text{R@k}}$$

### Hit Rate@k (HR@k)

Binary: did any relevant item appear in the top-$k$?

$$\text{HR@k} = \frac{1}{|U|} \sum_{u \in U} \mathbf{1}[\text{relevant item in top-}k \text{ for } u]$$

Useful for single-item recommendation (next item prediction).

---

## Ranking Quality Metrics

### Mean Reciprocal Rank (MRR)

Average reciprocal rank of the **first** relevant item:

$$\text{MRR} = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{\text{rank}_u}$$

Where $\text{rank}_u$ is the position of the first relevant item for user $u$.

**Example:**
- User 1: first relevant item at rank 1 → 1/1 = 1.0
- User 2: first relevant item at rank 3 → 1/3 ≈ 0.33
- User 3: no relevant item in top-k → 0
- MRR = (1.0 + 0.33 + 0) / 3 = 0.44

**Limitation:** Only cares about the first relevant item.

### Average Precision (AP)

Computes precision at every position where a relevant item appears:

$$\text{AP} = \frac{1}{|\text{Relevant}|} \sum_{k=1}^{K} \text{P@k} \cdot \text{rel}(k)$$

Where $\text{rel}(k) = 1$ if item at position $k$ is relevant, else 0.

**Example (4 relevant items, top-10 list):**
Relevant items at positions 1, 3, 7, 10:

$$\text{AP} = \frac{1}{4}\left(\frac{1}{1} + \frac{2}{3} + \frac{3}{7} + \frac{4}{10}\right) = \frac{1}{4}(1.0 + 0.667 + 0.429 + 0.4) = 0.624$$

### Mean Average Precision (MAP)

Average of AP over all users:

$$\text{MAP@k} = \frac{1}{|U|} \sum_{u \in U} \text{AP}_u@k$$

MAP@k is sensitive to both precision and recall up to rank $k$.

---

## Graded Relevance Metrics

### Discounted Cumulative Gain (DCG@k)

Items at lower positions contribute less (logarithmic discount):

$$\text{DCG@k} = \sum_{i=1}^k \frac{\text{rel}_i}{\log_2(i+1)}$$

Where $\text{rel}_i \in \{0, 1, 2, 3\}$ is the graded relevance of item at position $i$.

**Example (relevances = [3, 2, 0, 1, 2]):**
$$\text{DCG@5} = \frac{3}{\log_2 2} + \frac{2}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5} + \frac{2}{\log_2 6} = 3 + 1.26 + 0 + 0.43 + 0.77 = 5.46$$

### Ideal DCG (IDCG@k)

The best possible DCG — sort items by relevance (descending):

$$\text{IDCG@k} = \text{DCG@k of the ideal ranking}$$

For our example: $[3, 2, 2, 1, 0]$:
$$\text{IDCG@5} = \frac{3}{1} + \frac{2}{1.585} + \frac{2}{2} + \frac{1}{2.322} + \frac{0}{2.585} = 3 + 1.26 + 1 + 0.43 + 0 = 5.69$$

### Normalized DCG (NDCG@k)

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}} \in [0, 1]$$

For our example: NDCG@5 = 5.46 / 5.69 = 0.96

**Why NDCG?**
- Handles graded relevance (not just binary)
- Position-aware (higher positions matter more)
- Normalized so comparable across queries
- Industry standard for search and recommendation

---

## Classification Metrics

### AUC-ROC

Area Under the Receiver Operating Characteristic curve:

$$\text{AUC} = P(\hat{r}_{u, i^+} > \hat{r}_{u, i^-})$$

Probability that a randomly chosen positive item is ranked above a randomly chosen negative.

AUC = 0.5 → random; AUC = 1.0 → perfect.

**Limitation:** Doesn't account for position; treats all rank inversions equally.

### AUC-PR

Area Under the Precision-Recall curve — preferred when positive class is rare.

---

## Beyond Accuracy

### Coverage

What fraction of the item catalog is ever recommended?

$$\text{Coverage} = \frac{|\text{Recommended items}|}{|\mathcal{I}|}$$

Low coverage = popularity bias.

### Intra-List Diversity (ILD)

Average pairwise dissimilarity within a recommendation list $L$:

$$\text{ILD}(L) = \frac{1}{|L|(|L|-1)} \sum_{i \in L} \sum_{j \in L, j \neq i} d(i, j)$$

### Novelty

Average unexpectedness of recommended items:

$$\text{Novelty} = \frac{1}{K} \sum_{i=1}^K -\log_2 p(i)$$

Where $p(i)$ is the global popularity of item $i$.

### Serendipity

Relevant AND unexpected recommendations — measures pleasant surprise.

---

## Online Metrics

| Metric | Measures |
|--------|---------|
| **CTR** (Click-Through Rate) | $\frac{\text{clicks}}{\text{impressions}}$ |
| **Watch Time** | Total video consumption |
| **Session Length** | Time spent per session |
| **Conversion Rate** | Purchases / show-to-purchase |
| **Long-term Retention** | Users returning over weeks/months |

**Critical distinction:** Offline NDCG ↑ does not always mean online CTR ↑. A/B testing is necessary.

---

## Evaluation Protocol

### Time-Split Evaluation (Recommended)

```
Train: [Jan - Oct]  →  Validation: [Nov]  →  Test: [Dec]
```

Prevents data leakage; simulates real deployment.

### Leave-One-Out

For each user, hold out their last interaction for testing. Efficient for session-based models.

### Leave-p-Out

Hold out $p$ random interactions per user — tests recall in a less constrained setting.

---

## Metric Selection Guide

| Task | Recommended Metric |
|------|-------------------|
| Retrieval (stage 1) | Recall@k, Hit Rate@k |
| Ranking quality | NDCG@10, MAP@10 |
| Binary relevance | Precision@k, MRR |
| Click prediction | AUC-ROC, Log Loss |
| Graded relevance | NDCG |
| Diversity | ILD, Coverage |
| Production evaluation | CTR + Retention (A/B test) |

---

## Key Takeaways

1. **NDCG** is the gold standard for ranking — accounts for position and graded relevance
2. **MAP** is excellent for binary relevance across many relevant items
3. **MRR** is ideal when only the top-1 result matters
4. **Recall@k** is the primary metric for candidate generation (not ranking)
5. **AUC** measures discrimination but ignores ranking position
6. **Offline metrics ≠ online metrics** — always validate with A/B tests
7. **Beyond accuracy:** Diversity, novelty, and coverage matter for long-term user satisfaction

## References

- Järvelin & Kekäläinen (2002) — Cumulated Gain-Based Evaluation of IR Techniques (NDCG)
- Cremonesi et al. (2010) — Performance of Recommender Algorithms on Top-N Recommendation Tasks
- Herlocker et al. (2004) — Evaluating Collaborative Filtering Recommender Systems
