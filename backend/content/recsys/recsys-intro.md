---
title: "Introduction to Recommender Systems"
slug: recsys-intro
summary: "Overview of recommender system types, challenges, and the modern industrial pipeline from candidate generation to serving."
tags: ["recommendation", "collaborative-filtering", "content-based", "matrix-factorization", "two-tower"]
visibility: public
---

# Introduction to Recommender Systems

## Overview

A **recommender system** predicts a user's preference for items and surfaces the most relevant ones. They are central to platforms like Netflix, YouTube, Spotify, Amazon, and TikTok — often responsible for 60-80% of content consumed.

**Core problem:** Given a user $u$ and a catalog of items $\mathcal{I}$, predict a relevance score $\hat{r}_{u,i}$ for each item $i$ and return the top-$K$ items.

$$\text{TopK}(u) = \arg\text{TopK}_{i \in \mathcal{I}} \hat{r}(u, i)$$

---

## Types of Recommender Systems

### 1. Collaborative Filtering (CF)

**Core idea:** Users who agreed in the past will agree in the future.

- **User-based CF:** Find similar users → recommend what they liked
- **Item-based CF:** Find similar items to what the user liked

**User-item matrix:**

|  | Item A | Item B | Item C | Item D |
|--|--------|--------|--------|--------|
| User 1 | 5 | 3 | ? | 1 |
| User 2 | 4 | ? | 4 | 2 |
| User 3 | ? | 4 | 5 | ? |

Fill in the `?` entries by learning latent factors.

**Matrix Factorization:**
$$\hat{r}_{u,i} = \mathbf{p}_u^T \mathbf{q}_i + b_u + b_i + \mu$$

Where $\mathbf{p}_u \in \mathbb{R}^k$ is user latent factor, $\mathbf{q}_i \in \mathbb{R}^k$ is item latent factor.

**Training:**
$$\min_{\mathbf{P},\mathbf{Q}} \sum_{(u,i) \in \mathcal{O}} (r_{u,i} - \mathbf{p}_u^T \mathbf{q}_i)^2 + \lambda(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)$$

**Strengths:** Only needs interaction data (no item features needed)
**Weaknesses:** Cold start problem; doesn't use side features

### 2. Content-Based Filtering

Recommend items similar to what the user has liked before, based on item features:

$$\hat{r}(u, i) = \cos(\text{profile}(u), \text{features}(i))$$

User profile = weighted average of features of liked items.

**Example:** If a user liked action movies with Tom Hanks → recommend more action movies with Tom Hanks.

**Strengths:** No cold start for items; interpretable
**Weaknesses:** Filter bubble; requires rich item features; no cross-user signals

### 3. Hybrid Systems

Combine CF + content-based:
- **Weighted:** $\hat{r} = \alpha \cdot \hat{r}_{\text{CF}} + (1-\alpha) \cdot \hat{r}_{\text{CB}}$
- **Switching:** Use CF when enough data, CB for cold-start items
- **Feature augmentation:** Add CF embeddings as features to a content model

### 4. Session-Based Recommendations

Recommend based on the **current session** without user history:
- Sequential models (RNNs, SASRec, BERT4Rec)
- Graph neural networks on session graphs

---

## The Modern Industrial RecSys Pipeline

Real systems can't score billions of items for every request. They use a **funnel architecture**:

```
Billions of Items
        ↓ [Candidate Generation] ~1,000 items
        ↓ [Pre-Ranking / Scoring] ~200 items
        ↓ [Ranking] ~50 items
        ↓ [Re-ranking / Post-processing] ~10-20 items
        ↓ [UI / Business Logic] Final list served
```

### Stage 1: Candidate Generation (Retrieval)

Fast recall — retrieve a small set from the full catalog:
- ANN (Approximate Nearest Neighbor) over user/item embeddings
- Two-tower neural model
- Classical: item-based CF, BM25 for text items

### Stage 2: Ranking

Score the candidates with a more expensive model:
- Deep learning model with hundreds of features
- Typically optimized for CTR, watch time, purchase probability

### Stage 3: Re-ranking

Apply business constraints:
- Remove already-seen items
- Enforce diversity (don't show 10 items from same creator)
- Boost promoted/monetized items (ads)
- Apply safety filters

---

## Key Challenges

| Challenge | Description | Solution |
|-----------|-------------|---------|
| **Scalability** | Score millions of users × billions of items | Two-stage funnel |
| **Cold start** | New users/items with no history | Content-based, popularity, onboarding |
| **Sparsity** | Most (user, item) pairs unobserved | Matrix factorization, embedding learning |
| **Filter bubble** | Over-personalization limits discovery | Exploration/diversity |
| **Position bias** | Items shown higher get more clicks | Inverse propensity scoring |
| **Temporal dynamics** | Interests change over time | Time-aware features, session models |
| **Feedback loops** | Model causes its own training data | Exploration, debiasing |

---

## Implicit vs Explicit Feedback

| Type | Examples | Challenge |
|------|----------|-----------|
| **Explicit** | Star ratings, thumbs up/down | Sparse, not representative |
| **Implicit** | Clicks, watch time, purchases | Noisy (click ≠ like), abundant |

**WARP loss** for implicit feedback — maximizes AUC over positive-negative pairs.
**BPR (Bayesian Personalized Ranking):** Assumes user prefers interacted item over uninteracted:

$$\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j) \in D_S} \ln \sigma(\hat{r}_{u,i} - \hat{r}_{u,j})$$

---

## Key Takeaways

1. **Three paradigms:** Collaborative filtering, content-based, hybrid
2. **Matrix factorization** decomposes the user-item matrix into latent factors
3. **Industrial systems use a funnel:** retrieve → rank → re-rank
4. **Implicit feedback** is dominant in practice — noisy but abundant
5. **Cold start** is the hardest practical challenge; solved by content-based fallback
6. **Diversity vs relevance** is a fundamental tradeoff in production systems

## References

- Koren et al. (2009) — Matrix Factorization Techniques for Recommender Systems
- Covington et al. (2016) — Deep Neural Networks for YouTube Recommendations
- Yi et al. (2019) — Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
