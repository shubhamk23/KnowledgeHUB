---
title: "Loss Functions in Deep Learning"
slug: loss-functions
summary: "Comprehensive guide to classification, regression, ranking, and contrastive loss functions with formulas and when to use each."
tags: ["loss", "cross-entropy", "KL-divergence", "focal-loss", "triplet-loss", "contrastive"]
visibility: public
---

# Loss Functions in Deep Learning

## Overview

A **loss function** (also called cost function or objective function) measures how far the model's predictions are from the true labels. Training minimizes:

$$\theta^* = \arg\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}}[\mathcal{L}(f_\theta(x), y)]$$

Choosing the right loss function is crucial — it directly shapes what the model learns.

---

## Classification Losses

### Binary Cross-Entropy (BCE)

For binary classification ($y \in \{0, 1\}$, $\hat{p} = \sigma(\text{logit})$):

$$\mathcal{L}_{\text{BCE}} = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

**Use when:** Binary classification, multi-label classification (each label independently).

**Property:** Penalizes confident wrong predictions very heavily (log goes to $-\infty$).

### Categorical Cross-Entropy

For multi-class classification ($y$ is one-hot, $\hat{p}$ is softmax output):

$$\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{p}_c = -\log \hat{p}_{y_{\text{true}}}$$

**Relation to NLL:** Cross-entropy = Negative Log-Likelihood when target is a distribution.

### KL Divergence

Measures how distribution $Q$ (model) diverges from distribution $P$ (true):

$$D_{\text{KL}}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

Note: **Not symmetric** — $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$.

**Cross-entropy = KL divergence + entropy:**
$$H(P, Q) = H(P) + D_{\text{KL}}(P \| Q)$$

Since $H(P)$ is constant during training, minimizing CE ≡ minimizing KL divergence.

**Use when:** Knowledge distillation (match teacher soft labels), variational autoencoders.

### Focal Loss

Addresses **class imbalance** by down-weighting easy examples:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where:
- $p_t = p$ if $y=1$, else $1-p$ — probability of the correct class
- $\gamma \geq 0$ — focusing parameter (typically $\gamma = 2$)
- $\alpha_t$ — class weight balancing factor

**Intuition:** $(1-p_t)^\gamma$ acts as a modulating factor:
- Easy examples ($p_t \to 1$): factor $\to 0$, loss suppressed
- Hard examples ($p_t \to 0$): factor $\to 1$, normal loss

**Use when:** Object detection (RetinaNet), extreme class imbalance.

### Hinge Loss (SVM Loss)

$$\mathcal{L}_{\text{hinge}} = \max(0, 1 - y \cdot \hat{y})$$

For multi-class (Weston-Watkins):
$$\mathcal{L} = \sum_{j \neq y_i} \max(0, \hat{y}_j - \hat{y}_{y_i} + \Delta)$$

**Use when:** SVMs, margin-based classifiers.

### Dice Loss

For segmentation tasks, based on Dice coefficient:

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2|A \cap B|}{|A| + |B|} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}$$

**Use when:** Medical image segmentation, unbalanced pixel classification.

---

## Regression Losses

### Mean Absolute Error (MAE / L1 Loss)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

**Properties:**
- Robust to outliers (linear penalty)
- Non-differentiable at 0
- Produces **sparse** solutions

### Mean Squared Error (MSE / L2 Loss)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**Properties:**
- Sensitive to outliers (quadratic penalty amplifies errors)
- Differentiable everywhere
- Produces **smooth** solutions

### Root Mean Squared Error (RMSE)

$$\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

Same units as target variable — more interpretable than MSE.

### Huber Loss (Smooth L1)

Combines L1 and L2 — robust to outliers while remaining differentiable:

$$\mathcal{L}_{\delta}(a) = \begin{cases} \frac{1}{2} a^2 & \text{if } |a| \leq \delta \\ \delta |a| - \frac{1}{2}\delta^2 & \text{if } |a| > \delta \end{cases}$$

**Intuition:**
- Small errors ($|a| \leq \delta$): L2 behavior (smooth gradient)
- Large errors ($|a| > \delta$): L1 behavior (bounded gradient)

**Use when:** Regression with outliers, Q-learning in RL, object detection box regression.

### Log-Cosh Loss

$$\mathcal{L} = \sum_i \log(\cosh(\hat{y}_i - y_i))$$

Approximately L2 for small errors, L1 for large errors. Fully differentiable.

---

## Ranking & Contrastive Losses

### Triplet Loss

Learns embeddings where anchor $a$ is closer to positive $p$ than negative $n$:

$$\mathcal{L}_{\text{triplet}} = \max(0, d(a, p) - d(a, n) + \text{margin})$$

Where $d(\cdot, \cdot)$ is Euclidean or cosine distance, margin $> 0$ (typically 0.3–1.0).

**Mining strategies:**
- **Random:** Slow convergence
- **Hard negative:** $n = \arg\max d(a, n)$ — risks collapsed training
- **Semi-hard:** Negatives farther than positive but within margin

**Use when:** Face recognition, metric learning, embedding similarity.

### Contrastive Loss (Siamese Networks)

For pairs $(x_1, x_2)$ with label $y=1$ if similar, $y=0$ if dissimilar:

$$\mathcal{L} = y \cdot d^2 + (1-y) \cdot \max(0, \text{margin} - d)^2$$

### NT-Xent / InfoNCE (SimCLR, CLIP)

Noise-Contrastive Estimation — contrastive loss over a batch of $N$ samples:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

Where:
- $(z_i, z_j)$ — positive pair (augmented views of same image)
- $\tau$ — temperature (controls sharpness; typically 0.07–0.5)
- All other $2N-2$ samples in batch are negatives

**Use when:** Self-supervised learning (SimCLR, MoCo), CLIP, sentence embeddings.

### Multiple Negatives Ranking Loss (MNR)

Used in bi-encoder training for retrieval (Sentence Transformers):

$$\mathcal{L}_{\text{MNR}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\cos(a_i, p_i)/\tau)}{\sum_{j=1}^N \exp(\cos(a_i, p_j)/\tau)}$$

In a batch of $N$ pairs, uses all $N-1$ other positives as negatives.

---

## Reinforcement Learning Losses

### Q-Value Loss (DQN)

Minimize temporal difference error:

$$\mathcal{L} = \mathbb{E}\left[(r + \gamma \max_{a'} Q(s', a') - Q(s, a))^2\right]$$

Often using Huber loss for stability.

### Policy Gradient Loss (REINFORCE)

$$\mathcal{L}_{\text{PG}} = -\mathbb{E}_\tau \left[ \sum_t \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$

Where $G_t$ is the discounted return. Used in RLHF for LLM alignment.

---

## Loss Selection Guide

| Task | Recommended Loss |
|------|-----------------|
| Binary classification | BCE |
| Multi-class classification | Categorical CE |
| Class-imbalanced classification | Focal Loss |
| Regression (clean data) | MSE |
| Regression (outliers) | Huber / MAE |
| Segmentation | Dice + BCE combo |
| Metric learning | Triplet / NT-Xent |
| Similarity learning | Contrastive / MNR |
| Knowledge distillation | KL Divergence |
| RLHF fine-tuning | Policy gradient + Huber |

---

## Key Takeaways

1. **CE loss** is standard for classification — equivalent to minimizing KL divergence
2. **Focal loss** rescues training on imbalanced datasets by down-weighting easy negatives
3. **Huber loss** is the practitioner's default for regression — robust without sacrificing differentiability
4. **Triplet/InfoNCE** are the workhorses of metric learning and contrastive self-supervised learning
5. **Temperature $\tau$** in contrastive losses is critical — too high = uniform, too low = collapse
6. **Combining losses** (e.g., Dice + BCE for segmentation) often outperforms single losses

## References

- Lin et al. (2017) — Focal Loss for Dense Object Detection (RetinaNet)
- Schroff et al. (2015) — FaceNet: Triplet Loss
- Chen et al. (2020) — SimCLR: NT-Xent Loss
- Radford et al. (2021) — CLIP: InfoNCE Loss
