---
title: "CNN Architectures: From LeNet to EfficientNet"
slug: cnn-architectures-evolution
summary: "A detailed walkthrough of landmark CNN architectures — LeNet, AlexNet, VGG, GoogLeNet, ResNet, and EfficientNet — and the innovations each introduced."
tags: ["CNN", "ResNet", "VGG", "AlexNet", "EfficientNet", "GoogLeNet", "deep-learning", "computer-vision"]
visibility: public
level: intermediate
---

# CNN Architectures: From LeNet to EfficientNet

## The Evolution of Depth and Design

Each generation of CNN architectures introduced a key innovation that pushed the boundaries of what was possible. This article traces that evolution from the first practical CNN to modern compound-scaled networks.

---

## LeNet-5 (1998)

**Paper:** LeCun et al., "Gradient-Based Learning Applied to Document Recognition"

The first successful CNN, designed for handwritten digit recognition (MNIST).

**Architecture:**
```
Input (32×32×1) → Conv(5×5, 6) → Pool → Conv(5×5, 16) → Pool → FC(120) → FC(84) → Output(10)
```

| Property | Value |
|----------|-------|
| Parameters | ~60K |
| Depth | 5 layers (2 conv + 3 FC) |
| Activation | Sigmoid / Tanh |
| Pooling | Average pooling |

**Key contribution:** Demonstrated that learned convolutional features outperform hand-crafted features for pattern recognition.

**Limitation:** Too shallow for complex tasks. Sigmoid activations cause vanishing gradients in deeper networks.

---

## AlexNet (2012)

**Paper:** Krizhevsky, Sutskever, & Hinton, "ImageNet Classification with Deep Convolutional Neural Networks"

The model that launched the deep learning revolution by winning ILSVRC 2012 with a massive margin.

**Architecture:**
```
Input (227×227×3) → Conv(11×11, 96) → Pool → Conv(5×5, 256) → Pool
→ Conv(3×3, 384) → Conv(3×3, 384) → Conv(3×3, 256) → Pool
→ FC(4096) → FC(4096) → FC(1000)
```

| Property | Value |
|----------|-------|
| Parameters | ~60M |
| Depth | 8 layers (5 conv + 3 FC) |
| Top-5 error | 16.4% (vs. 25.8% previous best) |
| Training | 2 GPUs, 6 days |

**Key innovations:**
1. **ReLU activation:** $\text{ReLU}(x) = \max(0, x)$ — 6× faster training than tanh
2. **GPU training:** Split model across 2 GPUs (GTX 580, 3GB each)
3. **Dropout:** 50% dropout in FC layers for regularization
4. **Data augmentation:** Random crops, horizontal flips, PCA color augmentation
5. **Local Response Normalization (LRN):** Cross-channel normalization (later replaced by BatchNorm)

---

## VGGNet (2014)

**Paper:** Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition"

Showed that network depth is critical — and that small 3×3 filters are sufficient.

**Key insight:** Two stacked 3×3 convolutions have the same effective receptive field as one 5×5 convolution, but with fewer parameters and more non-linearity:

$$\text{Two 3×3 layers: } 2 \times (3^2 C^2) = 18C^2 \text{ params}$$
$$\text{One 5×5 layer: } 5^2 C^2 = 25C^2 \text{ params}$$

**VGG-16 Architecture:**
```
[Conv3-64] × 2 → Pool
[Conv3-128] × 2 → Pool
[Conv3-256] × 3 → Pool
[Conv3-512] × 3 → Pool
[Conv3-512] × 3 → Pool
→ FC(4096) → FC(4096) → FC(1000)
```

| Model | Layers | Parameters | Top-1 Accuracy |
|-------|--------|-----------|---------------|
| VGG-11 | 11 | 133M | 70.4% |
| VGG-16 | 16 | 138M | 74.4% |
| VGG-19 | 19 | 144M | 74.5% |

**Strengths:** Simple, uniform architecture. Excellent feature extractor for transfer learning.

**Weakness:** Extremely parameter-heavy (138M parameters, mostly in FC layers). Slow inference.

---

## GoogLeNet / Inception (2014)

**Paper:** Szegedy et al., "Going Deeper with Convolutions"

Introduced the **Inception module** — process input at multiple scales simultaneously.

### The Inception Module

Instead of choosing one filter size, use all of them in parallel:

```
          Input
       /   |   |   \
   1×1   3×3  5×5  MaxPool
   Conv  Conv  Conv  3×3
       \   |   |   /
      Concatenate
```

**1×1 convolutions** serve as dimensionality reduction bottlenecks before expensive 3×3 and 5×5 operations:

$$\text{Without bottleneck: } 5 \times 5 \times 256 \times 256 = 1.6M \text{ ops}$$
$$\text{With 1×1 bottleneck (64): } 1 \times 1 \times 256 \times 64 + 5 \times 5 \times 64 \times 256 = 0.43M \text{ ops}$$

| Property | Value |
|----------|-------|
| Parameters | ~7M (22× fewer than VGG-16) |
| Depth | 22 layers |
| Top-5 error | 6.7% |
| Inception modules | 9 stacked modules |

**Key innovation:** Multi-scale feature extraction with computational efficiency through bottleneck layers.

**Auxiliary classifiers:** Added intermediate classification heads at layers 4 and 7 to combat vanishing gradients during training (removed at inference).

---

## ResNet (2015)

**Paper:** He, Zhang, Ren, & Sun, "Deep Residual Learning for Image Recognition"

The most influential CNN architecture — solved the degradation problem in very deep networks.

### The Degradation Problem

Before ResNet, making networks deeper beyond ~20 layers actually **increased** training error (not just test error). This isn't overfitting — it's an optimization difficulty.

### Residual Connections

Instead of learning a mapping $H(x)$, learn the residual $F(x) = H(x) - x$:

$$\mathbf{y} = F(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

This is implemented as a **skip connection** (identity shortcut):

```
Input (x) ─────────────────────┐
    ↓                          │
  Conv → BN → ReLU             │
    ↓                          │
  Conv → BN                    │
    ↓                          │
  + ←──────────────────────────┘
    ↓
  ReLU
    ↓
Output: F(x) + x
```

**Why it works:**
- If the optimal mapping is close to identity, learning $F(x) \approx 0$ is easier than learning $H(x) \approx x$
- Gradients flow directly through skip connections — no vanishing gradient
- Enables training of networks with 100+ layers

### Bottleneck Block (ResNet-50+)

For deeper networks, use 1×1 → 3×3 → 1×1 bottleneck to reduce computation:

```
Input (256-d)
  ↓ 1×1 Conv → 64-d    (reduce)
  ↓ 3×3 Conv → 64-d    (process)
  ↓ 1×1 Conv → 256-d   (expand)
  + Input
```

### ResNet Family

| Model | Layers | Parameters | Top-1 (ImageNet) | GFLOPs |
|-------|--------|-----------|-------------------|--------|
| ResNet-18 | 18 | 11.7M | 69.8% | 1.8 |
| ResNet-34 | 34 | 21.8M | 73.3% | 3.7 |
| ResNet-50 | 50 | 25.6M | 76.0% | 4.1 |
| ResNet-101 | 101 | 44.5M | 77.4% | 7.9 |
| ResNet-152 | 152 | 60.2M | 78.3% | 11.6 |

**Impact:** ResNet skip connections became a fundamental building block in virtually all subsequent architectures (Transformers included).

---

## Squeeze-and-Excitation Networks (SENet, 2017)

**Paper:** Hu, Shen, & Sun, "Squeeze-and-Excitation Networks"

Introduced **channel attention** — let the network learn which feature channels are most important.

### SE Block

```
Input (H×W×C)
    ↓
Global Average Pool → (1×1×C)     ← Squeeze
    ↓
FC(C/r) → ReLU → FC(C) → Sigmoid  ← Excitation
    ↓
Scale each channel of Input        ← Reweight
```

$$\mathbf{s} = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{z}))$$

Where $\mathbf{z}$ is the squeezed descriptor and $r$ is the reduction ratio (typically 16).

**Result:** +0.5–1% accuracy improvement when added to any existing architecture, with minimal parameter overhead (~2.5M extra parameters for SE-ResNet-50).

---

## EfficientNet (2019)

**Paper:** Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"

Discovered that width, depth, and resolution should be scaled together.

### Compound Scaling

Previous approaches scaled only one dimension:
- **Wider:** More filters per layer (WideResNet)
- **Deeper:** More layers (ResNet-152)
- **Higher resolution:** Larger input images

EfficientNet scales all three uniformly:

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

Subject to constraint: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (to roughly double FLOPs per step).

### EfficientNet Family

| Model | Resolution | Parameters | Top-1 Accuracy | GFLOPs |
|-------|-----------|-----------|---------------|--------|
| B0 | 224 | 5.3M | 77.3% | 0.39 |
| B1 | 240 | 7.8M | 79.2% | 0.70 |
| B3 | 300 | 12M | 81.7% | 1.8 |
| B5 | 456 | 30M | 83.7% | 9.9 |
| B7 | 600 | 66M | 84.4% | 37 |

**Base architecture (B0)** uses **MBConv** (Mobile Inverted Bottleneck) blocks with SE attention — found via neural architecture search (NAS).

**Impact:** Achieved state-of-the-art accuracy with 8.4× fewer parameters than the best existing models.

---

## ConvNeXt (2022)

**Paper:** Liu et al., "A ConvNet for the 2020s"

Modernized ResNet with design choices borrowed from Vision Transformers, proving CNNs can still compete.

**Key modifications to ResNet:**
1. Training recipe updates (AdamW, augmentation, longer training)
2. Macro design: Swin Transformer stage ratios (3:3:9:3 → 3:3:27:3)
3. Patchify stem: 4×4 stride-4 convolution (like ViT's patch embedding)
4. Inverted bottleneck: Expand channels in depthwise conv
5. Larger kernel: 7×7 depthwise convolution
6. LayerNorm instead of BatchNorm
7. GELU instead of ReLU
8. Fewer activation functions (one per block, like Transformer)

| Model | Parameters | Top-1 (ImageNet) |
|-------|-----------|-------------------|
| ConvNeXt-T | 29M | 82.1% |
| ConvNeXt-S | 50M | 83.1% |
| ConvNeXt-B | 89M | 83.8% |
| ConvNeXt-L | 198M | 84.3% |

**Takeaway:** Pure CNNs can match Vision Transformers when given equivalent training recipes and design modernizations.

---

## Architecture Comparison Summary

| Architecture | Year | Key Innovation | Params | ImageNet Top-1 |
|-------------|------|---------------|--------|----------------|
| LeNet-5 | 1998 | First practical CNN | 60K | — |
| AlexNet | 2012 | ReLU, GPU training, dropout | 60M | 63.3% |
| VGG-16 | 2014 | Uniform 3×3 filters, depth | 138M | 74.4% |
| GoogLeNet | 2014 | Inception multi-scale modules | 7M | 74.8% |
| ResNet-50 | 2015 | Skip connections | 25.6M | 76.0% |
| SENet | 2017 | Channel attention | 28M | 82.7% |
| EfficientNet-B7 | 2019 | Compound scaling | 66M | 84.4% |
| ConvNeXt-B | 2022 | Modernized CNN | 89M | 83.8% |

---

## Key Takeaways

1. **Depth matters:** Going from 8 layers (AlexNet) to 152 layers (ResNet) dramatically improved accuracy
2. **Small filters win:** VGG showed that stacked 3×3 convolutions beat larger filters
3. **Multi-scale processing:** Inception modules capture features at multiple receptive field sizes
4. **Skip connections are fundamental:** ResNet's residual connections enabled very deep training and appear in virtually all modern architectures
5. **Attention improves CNNs:** SE blocks add channel-wise reweighting with minimal overhead
6. **Scale all dimensions together:** EfficientNet's compound scaling achieves better accuracy-efficiency tradeoffs
7. **CNNs are not dead:** ConvNeXt proves that modernized CNNs match Vision Transformers

## References

- LeCun, Y., et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proc. IEEE*, 86(11). [Paper](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS*. [Paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR 2015*. [arXiv:1409.1556](https://arxiv.org/abs/1409.1556)
- Szegedy, C., et al. (2015). Going Deeper with Convolutions. *CVPR 2015*. [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. *CVPR 2018*. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for CNNs. *ICML 2019*. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR 2022*. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
