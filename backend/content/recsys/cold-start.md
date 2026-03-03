---
title: "The Cold Start Problem in Recommender Systems"
slug: cold-start
summary: "Strategies for handling new users and new items with no interaction history in recommendation systems."
tags: ["cold-start", "new-user", "new-item", "recommendation", "exploration", "content-based"]
visibility: public
---

# The Cold Start Problem

## Overview

**Cold start** occurs when a recommender system encounters entities with little or no historical interaction data:

- **New user cold start:** A user just signed up — we know nothing about their preferences
- **New item cold start:** A newly published article, product, or video — no one has interacted with it
- **New system cold start:** A new platform with no users or items

This is one of the most practically important challenges in production recommender systems.

---

## New User Cold Start

### Problem

Collaborative filtering requires interaction history. Without it, there's no user vector to query.

**Severity scale:**
```
0 interactions   → random/popular recommendations only
1-5 interactions → weak signal, high uncertainty
5-20 interactions → reasonable personalization begins
20+ interactions → cold start effectively resolved
```

### Strategies

#### 1. Onboarding & Preference Elicitation

Ask users directly about their interests during signup:

```
Welcome! What topics interest you?
☐ Technology  ☐ Sports  ☐ Music  ☐ Cooking  ☐ Finance

Who do you follow? (Suggest popular creators in chosen topics)
```

**Tradeoff:** More questions → better personalization, but lower signup completion.

#### 2. Content-Based Bootstrap

Recommend based on available user attributes without interactions:
- Age, location, language, device type
- Referred-from context (came from a tech article → tech interest)
- Time of day / day of week patterns

$$\hat{r}(u, i) = \mathbf{w}^T \phi(u_{\text{attributes}}, i_{\text{features}})$$

#### 3. Popularity-Based Fallback

Show globally or regionally popular items:

$$\text{score}(i) = \frac{\text{interactions}(i, t_{\text{recent}})}{\text{impressions}(i, t_{\text{recent}})}$$

**Segmented popularity:** Trending among users with similar demographics.

#### 4. Social Graph Bootstrap

If the user signs up via social (Google, Facebook):
- Connect their social network
- Recommend items liked by their friends
- Use friend embeddings as proxy user embedding

#### 5. Exploration Strategies

**ε-greedy:** With probability ε, recommend randomly (explore); otherwise recommend best known:

$$a = \begin{cases} \text{random item} & \text{with prob. } \epsilon \\ \arg\max_{i} \hat{r}(u, i) & \text{with prob. } 1-\epsilon \end{cases}$$

**UCB (Upper Confidence Bound):**
$$\text{score}(i) = \hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}}$$

Balance exploitation ($\hat{\mu}_i$ = estimated reward) with exploration ($\sqrt{2 \ln t / n_i}$ = uncertainty term).

#### 6. Meta-Learning (Model-Agnostic Meta-Learning)

Train a model that can quickly adapt to new users with few examples:

- MAML-based approaches for RecSys
- Learn a good initialization that fine-tunes rapidly on 1-5 user interactions

---

## New Item Cold Start

### Problem

New items have no interaction data → collaborative filtering can't embed them → they won't be retrieved → never get impressions → never collect interactions → vicious cycle.

**Matthew effect:** Popular items get more exposure → more interactions → recommended more → richer representations.

### Strategies

#### 1. Content-Based Embedding

Generate item embeddings from content features without interactions:

```python
# Text-based: title, description, tags
text_embedding = text_encoder(item.title + item.description)

# Image-based
image_embedding = vision_encoder(item.thumbnail)

# Metadata-based: category, duration, language
meta_embedding = meta_encoder(item.category, item.duration)

# Combined
item_embedding = concat([text_embedding, image_embedding, meta_embedding])
```

Use this as the item tower output until interaction data is available.

#### 2. Warm-Up Period

Expose new items to a random subset of users and collect initial signals:

- **Scheduled exploration:** Every N requests, inject one new item
- **Explore-exploit bucketing:** 5% of users in "exploration" bucket see new items
- **New item promotion:** Explicitly boost recall of new items in retrieval

#### 3. Attribute-Based Retrieval

Instead of item ID embedding, use attribute-based retrieval:

$$\text{score}(u, i_{\text{new}}) = \sum_{a \in \text{attributes}(i_{\text{new}})} \text{score}(u, a)$$

If user liked "Python tutorial" items, score new Python tutorials highly.

#### 4. Popularity-by-Category

Bootstrap new items with category-level popularity:

$$\hat{r}(u, i_{\text{new}}) = r_{\text{global}}(i_{\text{new}}) \cdot \text{affinity}(u, \text{category}(i_{\text{new}}))$$

#### 5. Transfer from Similar Items

Find the most similar existing item (by content embedding), use its interaction patterns:

$$\mathbf{q}_{i_{\text{new}}} = \frac{1}{K} \sum_{k=1}^K \mathbf{q}_{i_k^{\text{similar}}}$$

#### 6. ID-Free (Content-Only) Models

Train the item tower to use ONLY content features (no item ID embedding):

**Pros:** Zero cold start for new items
**Cons:** Lower performance for warm items (ID embeddings are very powerful)

**Hybrid approach:** Train with both ID + content; at inference, fall back to content when ID embedding not available.

---

## New System Cold Start

When the platform itself is new (no users, no items, no interactions):

1. **Manual curation:** Editorial team selects initial content
2. **Import from similar domains:** Use embeddings from pre-trained models
3. **Synthetic data:** Generate interactions from behavioral models
4. **Crawl public signals:** Web scraping, social media engagement
5. **Partner data:** License interaction data from related services

---

## Multi-Armed Bandit for Cold Start

Model cold start as an exploration problem:

### Thompson Sampling

Maintain a Beta distribution over CTR for each item:

$$\text{Beta}(\alpha_i, \beta_i) \quad \alpha_i = 1 + \text{clicks}_i, \quad \beta_i = 1 + \text{no-clicks}_i$$

At each step, sample CTR estimates and show items with highest sampled CTR:

```python
for item in candidate_items:
    sampled_ctr = np.random.beta(item.alpha, item.beta)
    item.score = sampled_ctr

top_items = sorted(candidate_items, key=lambda x: -x.score)[:K]
```

**Advantages:** Automatically explores uncertain items more, converges to optimal as data accumulates.

---

## Practical Cold Start Decision Tree

```
New user arrives:
    → Has social graph? → Use social bootstrap
    → Did onboarding? → Use preference embeddings
    → Has demographics? → Use demographic model
    → Otherwise → Trending / Popular fallback

New item published:
    → Has rich content? → Content-based embedding
    → Similar to existing items? → Transfer embedding
    → Otherwise → Scheduled warm-up exposure
```

---

## Key Takeaways

1. **Cold start is inevitable** in any recommender system — plan for it from day 1
2. **Content features** are the primary solution for new item cold start
3. **Onboarding flows** significantly reduce new user cold start
4. **Exploration strategies** (UCB, Thompson sampling) provide a principled tradeoff
5. **Hybrid models** (ID + content) gracefully degrade to content-only for cold items
6. **Vicious cycles** (new items never exposed → never get data) must be broken with deliberate exploration
7. **Cold start usually resolves** within 5-20 interactions for users, 100+ impressions for items

## References

- Schein et al. (2002) — Methods and Metrics for Cold-Start Recommendations
- Wei et al. (2021) — Contrastive Learning for Cold-Start Recommendation
- Vartak et al. (2017) — A Meta-Learning Perspective on Cold-Start Recommendations
- Auer et al. (2002) — Finite-time Analysis of the Multiarmed Bandit Problem (UCB)
