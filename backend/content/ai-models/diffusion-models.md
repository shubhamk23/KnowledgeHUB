---
title: "Diffusion Models"
slug: diffusion-models
summary: "How denoising diffusion probabilistic models generate images by learning to reverse a noise corruption process."
tags: ["diffusion", "generative", "DDPM", "stable-diffusion", "image-generation", "UNet"]
visibility: public
---

# Diffusion Models

## Overview

**Diffusion models** are a class of generative models that learn to generate data by learning to reverse a gradual noising process. They have become the dominant approach for high-quality image generation, powering Stable Diffusion, DALL-E 2, Midjourney, and Imagen.

**Core papers:**
- DDPM (Ho et al., 2020) — "Denoising Diffusion Probabilistic Models"
- Score matching (Song & Ermon, 2019)
- Stable Diffusion (Rombach et al., 2022) — "High-Resolution Image Synthesis with Latent Diffusion Models"

---

## The Core Idea

**Two processes:**
1. **Forward process (q):** Gradually add noise to destroy data
2. **Reverse process (p):** Learn to remove noise to recover data

```
Data: x₀ → x₁ → x₂ → ... → x_T ≈ N(0,I)    [forward: add noise]
Gen:  N(0,I) → ... → x₂ → x₁ → x₀           [reverse: denoise]
```

The model learns the reverse process by training on the forward process.

---

## Forward Process

The forward process gradually adds Gaussian noise over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})$$

Where $\beta_t \in (0, 1)$ is the noise schedule (typically linear or cosine).

**Closed-form marginal** — sample at any timestep $t$ directly from $x_0$:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I})$$

Where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

**Reparameterization trick:**
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

This is crucial for training — we can directly sample $x_t$ without iterating through all steps.

---

## Reverse Process

The reverse process is parameterized as:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

A neural network $\epsilon_\theta(x_t, t)$ predicts the noise added at step $t$.

**DDPM training objective:**

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**Algorithm:**
1. Sample $x_0 \sim q(x_0)$ (real data)
2. Sample $t \sim \text{Uniform}(1, T)$
3. Sample $\epsilon \sim \mathcal{N}(0, I)$
4. Compute $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
5. Predict $\hat{\epsilon} = \epsilon_\theta(x_t, t)$
6. Loss = $\|\epsilon - \hat{\epsilon}\|^2$

---

## Sampling (Inference)

Start from pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoise:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

Where $z \sim \mathcal{N}(0, I)$ is added noise (except at final step).

**Problem:** DDPM requires $T = 1000$ steps → slow sampling.

### DDIM (Denoising Diffusion Implicit Models)

Non-Markovian reverse process — allows **fewer steps** (50-200) with deterministic sampling:

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \epsilon_\theta(x_t, t) + \sigma_t z$$

With $\sigma_t = 0$ (deterministic): same samples from same noise with any schedule.

**Speedup:** 50 DDIM steps ≈ 1000 DDPM steps in quality.

---

## The UNet Architecture

The noise prediction network $\epsilon_\theta$ is typically a **UNet** with:
- Encoder: Downsampling blocks with residual connections
- Bottleneck: Attention layers
- Decoder: Upsampling with skip connections from encoder

```
Input (noisy image, timestep t)
    ↓
[Conv Block] [Down 64ch]
    ↓ skip_1
[Conv Block] [Down 128ch]
    ↓ skip_2
[Conv Block] [Down 256ch]
    ↓
[Attention + Conv Bottleneck 512ch]
    ↑
[Up 256ch] ← skip_2
    ↑
[Up 128ch] ← skip_1
    ↑
[Up 64ch]
    ↓
Output (predicted noise ε)
```

**Timestep conditioning:** $t$ is sinusoidally embedded and added to residual blocks.

---

## Latent Diffusion Models (LDM / Stable Diffusion)

**Problem:** Running diffusion in pixel space for high-res images is extremely expensive.

**Solution (Rombach et al., 2022):** Train diffusion in the **latent space** of a VAE:

```
Image x (512×512×3)
    ↓ VAE Encoder
Latent z (64×64×4)  ← Diffusion runs here!
    ↓ VAE Decoder
Image x' (512×512×3)
```

**80× smaller latent** → much faster training and sampling.

### Conditional Generation

For text-to-image, condition the UNet on text embeddings via **cross-attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Where:
- $Q = W_Q \cdot \phi(z_t)$ — queries from image latent
- $K = W_K \cdot \tau_\theta(y)$ — keys from text embedding
- $V = W_V \cdot \tau_\theta(y)$ — values from text embedding
- $\tau_\theta$ — text encoder (CLIP or T5)

### Classifier-Free Guidance (CFG)

Amplify the conditional signal:

$$\hat{\epsilon}_\theta(x_t, y) = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \emptyset))$$

- $\epsilon_\theta(x_t, y)$ — conditional prediction (with text)
- $\epsilon_\theta(x_t, \emptyset)$ — unconditional prediction (without text)
- $w$ — guidance scale (typically 7-12; higher = more text-aligned, less diverse)

**Training:** Randomly drop conditioning with 10-20% probability → model learns both conditional and unconditional denoising.

---

## Model Comparison

| Model | Key Feature | Organization |
|-------|------------|--------------|
| DDPM | Original diffusion | Berkeley |
| DDIM | Fast sampling | Stanford |
| Stable Diffusion 1.x | LDM, CLIP conditioning | Stability AI |
| Stable Diffusion XL | Larger UNet, 2-stage | Stability AI |
| DALL-E 2 | CLIP prior + LDM | OpenAI |
| Imagen | T5 conditioning, cascaded | Google |
| DiT | Transformer UNet | Meta |
| Flux.1 | Rectified flow + DiT | Black Forest Labs |

### DiT (Diffusion Transformer)

Replace UNet with a **Transformer** operating on image patches:
- Patches are tokens, timestep $t$ modulates LayerNorm
- Scales better than UNet with more compute
- Used in Stable Diffusion 3, Flux, Sora

---

## Applications Beyond Images

| Application | Examples |
|-------------|---------|
| Video generation | Sora (OpenAI), Gen-2 |
| Audio/music | Stable Audio, MusicGen |
| 3D shape generation | DreamFusion, Magic3D |
| Protein structure | RFDiffusion |
| Drug discovery | DiffDock (molecular docking) |

---

## Key Takeaways

1. **Diffusion = learn to reverse noise** — train on forward corruption, sample by denoising
2. **DDPM training** is simply predicting the noise $\epsilon$ added at timestep $t$
3. **DDIM** enables 10-50× faster sampling with same model (deterministic, non-Markovian)
4. **Latent Diffusion (Stable Diffusion)** operates in VAE latent space — 80× more efficient
5. **Cross-attention** injects text conditioning into UNet; CFG amplifies the effect
6. **DiT architecture** (Transformers replacing UNet) is the current frontier

## References

- Ho et al. (2020) — Denoising Diffusion Probabilistic Models (DDPM)
- Song et al. (2020) — Denoising Diffusion Implicit Models (DDIM)
- Rombach et al. (2022) — High-Resolution Image Synthesis with Latent Diffusion Models
- Peebles & Xie (2023) — Scalable Diffusion Models with Transformers (DiT)
- Ho & Salimans (2022) — Classifier-Free Diffusion Guidance
