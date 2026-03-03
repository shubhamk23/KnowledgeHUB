---
title: "Gradient Descent & Optimization"
slug: gradient-descent
summary: "From vanilla gradient descent to Adam — understanding how neural networks learn."
tags: ["optimization", "gradient-descent", "adam", "sgd"]
visibility: public
---

# Gradient Descent & Optimization

Gradient descent is the backbone of neural network training. Given a loss function $L(\theta)$, we iteratively update parameters in the direction that reduces the loss.

## Vanilla Gradient Descent

$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta)$$

where $\eta$ is the learning rate. The entire dataset is used to compute the gradient — **expensive** for large datasets.

## Stochastic Gradient Descent (SGD)

Use a single sample (or mini-batch) per update:

$$\theta \leftarrow \theta - \eta \nabla_\theta L(\theta; x^{(i)}, y^{(i)})$$

**Pros:** Fast updates, can escape local minima due to noise.
**Cons:** High variance in gradient estimates.

## Momentum

Accumulate a velocity vector to dampen oscillations:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta L(\theta)$$
$$\theta \leftarrow \theta - v_t$$

Typical $\gamma = 0.9$. Momentum "builds up speed" in consistent directions.

## Adam Optimizer

Adam (Adaptive Moment Estimation) combines momentum and RMSProp:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(2nd moment)}$$

Bias-corrected estimates:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Update rule:
$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

## Learning Rate Schedules

```python
# Cosine annealing
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

# Warmup + decay (common in Transformers)
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return (1 - (step - warmup_steps) / (total_steps - warmup_steps)) ** 0.9
```

## Comparison Table

| Optimizer | Adaptive LR | Momentum | When to Use |
|-----------|-------------|----------|-------------|
| SGD | No | Optional | When you need maximum control |
| SGD + Momentum | No | Yes | Computer vision, large-scale training |
| RMSProp | Yes | No | RNNs, non-stationary problems |
| Adam | Yes | Yes | Default choice, NLP, smaller datasets |
| AdamW | Yes | Yes | Transformers (weight decay decoupled) |

## Practical Tips

- **Learning rate** is the most important hyperparameter — use learning rate finders
- **Gradient clipping** (clip by norm to 1.0) prevents exploding gradients in RNNs/Transformers
- **Weight decay** (L2 regularization) is better implemented via AdamW than adding to the loss
- **Warmup** is critical for Transformers — start with a small LR and ramp up
