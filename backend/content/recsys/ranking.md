---
title: "Ranking in Recommender Systems"
slug: ranking
summary: "Pointwise, pairwise, and listwise learning-to-rank methods, plus deep learning ranking models used in production recommendation systems."
tags: ["ranking", "pointwise", "pairwise", "listwise", "LTR", "GBDT", "deep-learning"]
visibility: public
---

# Ranking in Recommender Systems

## Overview

The **ranking stage** scores the candidates produced by retrieval (typically 200-1000 items) using a richer, more expensive model to produce a final ordered list. Unlike retrieval, ranking can afford to:
- Use cross-features between user and item
- Apply hundreds of real-time features
- Run computationally expensive neural networks

**Goal:** Assign a relevance score to each (user, item) pair and sort by score.

---

## Learning to Rank (LTR) Paradigms

### 1. Pointwise

Treats ranking as regression or classification — predict a relevance score for each item independently:

$$\hat{r}(u, i) = f(x_{u,i})$$

**Regression:** MSE loss against numerical relevance labels.
**Classification:** Cross-entropy against binary labels (clicked / not clicked).

**Pros:** Simple, lots of well-understood ML applies.
**Cons:** Doesn't directly optimize for ranking quality; items scored independently.

### 2. Pairwise

Optimize the relative order of pairs $(i, j)$ where $i$ is more relevant than $j$:

$$\mathcal{L} = \sum_{(i,j): r_i > r_j} \phi(f(x_i) - f(x_j))$$

**RankNet (Burges et al., 2005):**
$$P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1 + e^{-(s_i - s_j)}}$$

$$\mathcal{L}_{\text{RankNet}} = -\bar{P}_{ij} \log P_{ij} - (1 - \bar{P}_{ij}) \log(1 - P_{ij})$$

**LambdaRank:** Weighted pairwise where weights are $|\Delta \text{NDCG}|$ — directly approximates NDCG.

### 3. Listwise

Optimize the entire list as a whole:

**ListNet:** Defines probability distributions over permutations:

$$P_s(\pi) = \prod_{j=1}^n \frac{e^{s_{\pi(j)}}}{\sum_{k=j}^n e^{s_{\pi(k)}}}$$

$$\mathcal{L}_{\text{ListNet}} = -\sum_\pi P_y(\pi) \log P_s(\pi)$$

**ListMLE:** Maximum likelihood estimation of the ranking permutation.

### Paradigm Comparison

| Paradigm | Pros | Cons | Examples |
|----------|------|------|---------|
| Pointwise | Simple, scalable | Doesn't optimize ranking metric | LR, GBDT regression |
| Pairwise | Considers relative order | Expensive pairs computation | RankNet, LambdaRank |
| Listwise | Directly optimizes list quality | Complex, computationally heavy | ListNet, SoftRank |

---

## Feature Engineering for Ranking

### Feature Categories

**User Features:**
- User ID embedding
- Long-term interests (genres, categories)
- Recent interaction sequence
- Demographics

**Item Features:**
- Item ID embedding
- Content features (text, image embeddings)
- Popularity statistics (global CTR, average rating)
- Recency (publish time)

**Cross Features (User × Item):**
- Has user seen this creator before?
- User's historical CTR on this category
- Time since user last engaged with similar items

**Context Features:**
- Query text (for search)
- Device type, time of day
- Position in feed

**Interaction Statistics:**
- Item's CTR in past 24h
- User's CTR history
- Position bias correction

---

## Deep Learning Ranking Models

### Wide & Deep (Google Play Store, 2016)

Combines memorization (wide) and generalization (deep):

```
Input Features
      ├──────────────── Wide (Linear) ────────────────┐
      │                                               │
      └──────── Deep (Fully Connected) ───────────────┤
                                                      ↓
                                              Output (logit)
```

**Wide component:** Cross-product feature transformations (memorization):
$$y_{\text{wide}} = \mathbf{w}^T [\mathbf{x}, \phi(\mathbf{x})]$$

**Deep component:** Embeddings → dense layers (generalization):
$$y_{\text{deep}} = f(W_L \cdots f(W_1 [\mathbf{e}_{u}, \mathbf{e}_{i}]))$$

### DeepFM (Feature Interactions)

Factorization Machine + Deep network for automatic high-order feature interactions:

$$\hat{y} = \sigma(y_{\text{FM}} + y_{\text{DNN}})$$

FM component handles pairwise interactions:
$$y_{\text{FM}} = w_0 + \sum_i w_i x_i + \sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j$$

Where inner products $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ replace per-pair weights ($O(n^2) \to O(kn)$).

### DCN v2 (Deep & Cross Network)

Efficient high-degree feature crossing:
$$\mathbf{x}_{l+1} = \mathbf{x}_0 (\mathbf{W}_l \mathbf{x}_l + \mathbf{b}_l) + \mathbf{x}_l$$

Each cross layer mixes the original features $\mathbf{x}_0$ with the current representation.

### DLRM (Meta, 2019)

Production-scale RecSys model:
1. Embed sparse features (user ID, item ID, categories)
2. Process dense features through MLP
3. Second-order interactions between all embedding pairs
4. Final MLP → prediction

### Multi-Task Learning

Production rankers optimize multiple objectives simultaneously:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{CTR}} + \lambda_2 \mathcal{L}_{\text{watch\_time}} + \lambda_3 \mathcal{L}_{\text{like}}$$

**MMoE (Multi-gate Mixture-of-Experts):**
- Shared expert networks + task-specific gating
- Different tasks can use different expert combinations

---

## Gradient Boosted Decision Trees (GBDT)

Still widely used in ranking (LightGBM, XGBoost):

**LambdaMART:** GBDT with LambdaRank gradients — optimizes NDCG directly.

```python
import lightgbm as lgb

model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[5, 10],
    n_estimators=500,
    learning_rate=0.05
)
model.fit(X_train, y_train, group=query_groups)
```

**Why GBDT still wins:** Interpretable, fast inference, handles missing values, doesn't need feature normalization.

**Industry practice:** GBDT as baseline, deep learning for embedding features → combine with GBDT features.

---

## Position Bias

Items shown in higher positions get more clicks regardless of actual quality. This creates **selection bias** in training data.

**Inverse Propensity Scoring (IPS):**
$$\mathcal{L}_{\text{unbiased}} = \sum_{(u,i)} \frac{r_{u,i}}{p(i | u)} \cdot \mathcal{L}(f(x_{u,i}), r_{u,i})$$

Where $p(i | u)$ is the propensity (probability of showing item $i$ to user $u$).

**Randomization experiments:** Randomly shuffle some results to collect unbiased click data.

---

## Key Takeaways

1. **Pointwise** is simplest but doesn't directly optimize ranking metrics
2. **Pairwise** (LambdaRank) is most common in production — good tradeoff
3. **Feature interactions** (DeepFM, DCN) are crucial — user×item cross features drive performance
4. **Multi-task learning** is standard in production (CTR + watch time + likes)
5. **GBDT** remains competitive, especially for tabular features; often combined with neural approaches
6. **Position bias** is a major issue — correct with IPS or randomization experiments

## References

- Cheng et al. (2016) — Wide & Deep Learning for Recommender Systems
- Guo et al. (2017) — DeepFM
- Burges et al. (2005) — Learning to Rank using Gradient Descent (RankNet)
- Wang et al. (2021) — DCN v2
- Naumov et al. (2019) — DLRM
