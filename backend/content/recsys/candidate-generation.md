---
title: "Candidate Generation in Recommender Systems"
slug: candidate-generation
summary: "Two-tower models, approximate nearest neighbor search, and embedding-based retrieval for scaling recommendations to billions of items."
tags: ["two-tower", "ANN", "embedding", "retrieval", "recommendation", "FAISS"]
visibility: public
---

# Candidate Generation in Recommender Systems

## Overview

**Candidate generation** (also called retrieval) is the first stage of the recommendation funnel. It efficiently retrieves a small set of potentially relevant items (hundreds to thousands) from a catalog that may contain billions of items.

**Goal:** High recall at low latency. We'd rather retrieve some irrelevant items than miss relevant ones — the ranking stage will filter them.

**Constraints:**
- Must score all $N$ items in milliseconds
- Can't use expensive features (no cross-attention between query and items)
- Approximate results are acceptable

---

## Two-Tower Model (Dual Encoder)

The dominant architecture for neural candidate generation:

```
User Tower              Item Tower
   ↑                       ↑
[user features]         [item features]
   ↓                       ↓
Dense layers            Dense layers
   ↓                       ↓
user embedding (k-dim) ←→ item embedding (k-dim)
                   ↓
          dot product / cosine sim
```

**Scoring:**
$$s(u, i) = \mathbf{e}_u \cdot \mathbf{e}_i = \sum_{d=1}^k e_u^{(d)} \cdot e_i^{(d)}$$

**Key property:** The towers are **decoupled** — item embeddings can be pre-computed and indexed offline. Only the user tower runs at query time.

### Training

**Positive pairs:** $(u, i)$ where user interacted with item
**Negative pairs:** Items the user didn't interact with

**In-batch negatives (YouTube DNN paper):**
Use other items in the batch as negatives. With batch size $B$:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \log \frac{e^{s(u_i, i_i)/\tau}}{\sum_{j=1}^B e^{s(u_i, i_j)/\tau}}$$

This is equivalent to NT-Xent/InfoNCE loss.

**Problem:** Popular items appear as negatives more often → model learns to push down popular items too aggressively.

**Solution: Sampling-bias correction (Google, 2019):**
Divide scores by item frequency to correct for popularity bias:
$$s_{\text{corrected}}(u, i) = s(u, i) - \log p_i$$

### User Features

Typical user tower inputs:
- User ID embedding
- Watched video IDs (embedded, pooled)
- Search history embeddings
- Geographic/demographic features
- Time of day, device type
- Recent interaction sequence

### Item Features

Typical item tower inputs:
- Item ID embedding
- Title/description text embedding (from pre-trained model)
- Category, creator ID
- Item metadata (duration, language, age)
- Engagement statistics (global CTR, watch rate)

---

## Approximate Nearest Neighbor (ANN) Search

After training, we pre-compute all item embeddings and build an ANN index. At query time, we compute the user embedding and find the nearest items in the embedding space.

**Challenge:** Exact nearest neighbor over billions of vectors is too slow ($O(N \cdot d)$ per query).

### FAISS (Facebook AI Similarity Search)

Most widely used ANN library.

**Flat Index (exact, slow):**
```python
import faiss
index = faiss.IndexFlatL2(dim)  # exact L2 search
index.add(item_embeddings)      # add all items
D, I = index.search(query_embedding, k=100)  # top-100
```

**IVF (Inverted File Index):**
Partition space into $C$ cells (Voronoi regions via k-means), search only nearest cells:

```python
index = faiss.IndexIVFFlat(quantizer, dim, C)
index.train(item_embeddings)  # learn partitions
index.add(item_embeddings)
index.nprobe = 10  # search top-10 cells (recall vs speed tradeoff)
D, I = index.search(query_embedding, k=100)
```

**HNSW (Hierarchical Navigable Small World):**
Graph-based ANN — navigates a hierarchical graph of items. Very fast queries, high recall.

```python
index = faiss.IndexHNSWFlat(dim, 32)  # M=32 edges per node
index.add(item_embeddings)
D, I = index.search(query_embedding, k=100)
```

### ANN Tradeoffs

| Method | Build Time | Query Time | Memory | Recall |
|--------|-----------|------------|--------|--------|
| Flat (exact) | Fast | O(N·d) | Low | 100% |
| IVF | Moderate | O(N/C·d) | Low | 90-99% |
| HNSW | Slow | O(log N) | High | 95-99% |
| PQ (Product Quantization) | Moderate | Fast | Very Low | 85-95% |

**ScaNN (Google):** Combines AH quantization + reordering, often best quality-speed tradeoff.

---

## Retrieval Strategies

### 1. Embedding-Based (Neural)

Two-tower as described above. Strengths: captures complex patterns, handles heterogeneous features.

### 2. Item-to-Item (I2I)

For each item the user recently interacted with, retrieve similar items:
$$\text{Candidates} = \bigcup_{i \in \text{user\_history}} \text{ANN}(e_i, k=100)$$

Strengths: interpretable, works well for session-based recommendations.

### 3. User-to-Item (U2I)

Direct user embedding → ANN search. The two-tower model does this.

### 4. Keyword / BM25 (Sparse Retrieval)

For text-rich catalogs (articles, jobs, products), BM25 on title/description is a strong baseline.

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{\text{TF}(t, d) \cdot (k_1 + 1)}{\text{TF}(t, d) + k_1 \cdot (1 - b + b \cdot |d| / \text{avgdl})}$$

### 5. Hybrid Retrieval

Combine neural + sparse:
$$\text{score}(q, d) = \alpha \cdot \text{neural}(q, d) + (1-\alpha) \cdot \text{BM25}(q, d)$$

RRF (Reciprocal Rank Fusion) is a popular rank-fusion method requiring no calibration:
$$\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}, \quad k=60$$

---

## Serving Architecture

```
User request
     ↓
[User Tower Inference] → user embedding (real-time)
     ↓
[ANN Index Lookup] → top-1000 item IDs (precomputed embeddings)
     ↓
[Item Feature Fetching] → feature store / cache
     ↓
[Send to Ranking Stage] → 1000 candidates with features
```

**Latency budget:** Candidate generation typically has 10-50ms budget.

---

## Key Takeaways

1. **Two-tower** model is the industry standard — decoupled user/item towers enable pre-computation
2. **In-batch negatives** make training efficient but can bias toward popular items
3. **ANN search** (FAISS/ScaNN) enables sub-millisecond retrieval over billions of items
4. **Multiple sources** (neural + I2I + BM25) provide better recall than any single source
5. **Sampling-bias correction** is critical for unbiased retrieval in skewed catalogs
6. **Recall@k** is the key metric — higher recall at stage 1 = better potential at stage 2

## References

- Covington et al. (2016) — Deep Neural Networks for YouTube Recommendations
- Yi et al. (2019) — Sampling-Bias-Corrected Neural Modeling (Google)
- Johnson et al. (2021) — Billion-Scale Similarity Search with GPUs (FAISS)
- Guo et al. (2020) — Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)
