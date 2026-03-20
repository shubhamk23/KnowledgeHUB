---
title: "Image Classification Fundamentals"
slug: image-classification-fundamentals
summary: "A beginner-friendly introduction to image classification — from pixels to predictions using CNNs and softmax classifiers."
tags: ["image-classification", "CNN", "softmax", "ImageNet", "computer-vision", "deep-learning"]
visibility: public
level: beginner
---

# Image Classification Fundamentals

## What is Image Classification?

**Image classification** is the task of assigning a label (or class) to an entire image. Given an input image, the model outputs a probability distribution over predefined categories.

**Example:** Given a photo of a dog, the model outputs:
- Dog: 0.92
- Cat: 0.05
- Horse: 0.03

This is the foundational task in computer vision — most other tasks (detection, segmentation) build on classification backbones.

---

## From Pixels to Features

### How Computers See Images

A digital image is a 3D array of numbers:

$$\text{Image} \in \mathbb{R}^{H \times W \times C}$$

Where:
- $H$ = height (pixels)
- $W$ = width (pixels)
- $C$ = channels (3 for RGB, 1 for grayscale)

Each pixel value ranges from 0 to 255 (8-bit) or 0.0 to 1.0 (normalized). A 224×224 RGB image has **150,528 input values** — far too many for simple classifiers to handle directly.

### The Feature Extraction Problem

Raw pixels are poor features for classification:
- A 1-pixel shift changes every input value
- Lighting changes affect all pixel intensities
- Objects can appear at different scales and orientations

**Solution:** Learn hierarchical features that are invariant to these transformations.

| Feature Level | What It Captures | Example |
|---------------|------------------|---------|
| Low-level | Edges, corners, textures | Horizontal lines, color gradients |
| Mid-level | Parts, patterns | Eyes, wheels, windows |
| High-level | Objects, scenes | "Cat face", "car", "beach" |

---

## The Convolution Operation

### Why Convolutions?

Convolutions solve three key problems:

1. **Parameter efficiency:** A 3×3 filter has only 9 weights, regardless of image size
2. **Translation equivariance:** The same filter detects features anywhere in the image
3. **Local connectivity:** Each output depends on a small neighborhood of the input

### How Convolution Works

A filter (kernel) $K$ of size $k \times k$ slides across the input $X$ to produce output $Y$:

$$Y[i,j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} X[i+m, j+n] \cdot K[m,n] + b$$

**Key hyperparameters:**

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| Kernel size | Receptive field | 3×3, 5×5, 7×7 |
| Stride | Step size | 1 (preserve size), 2 (downsample) |
| Padding | Border handling | "same" (preserve size), "valid" (no padding) |
| Filters | Number of output channels | 32, 64, 128, 256, 512 |

### Output Size Formula

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

Where $W$ = input size, $K$ = kernel size, $P$ = padding, $S$ = stride.

---

## Building a CNN Classifier

### Standard Architecture Pattern

A typical CNN classifier follows this structure:

```
Input Image (224×224×3)
    ↓
[Conv → BatchNorm → ReLU → Pool] × N    ← Feature extraction
    ↓
Global Average Pooling                    ← Spatial compression
    ↓
Fully Connected Layer                     ← Classification
    ↓
Softmax                                   ← Probability distribution
```

### Activation Functions

**ReLU (Rectified Linear Unit)** is the standard activation:

$$\text{ReLU}(x) = \max(0, x)$$

Why ReLU works:
- Computationally efficient (just a threshold)
- Avoids vanishing gradient for positive values
- Introduces non-linearity (essential for learning complex patterns)

### Pooling Layers

Pooling reduces spatial dimensions while retaining important features:

- **Max Pooling:** Takes the maximum value in each window — preserves dominant features
- **Average Pooling:** Takes the mean — smoother downsampling
- **Global Average Pooling (GAP):** Reduces each channel to a single value — replaces fully connected layers

### Softmax Classifier

The final layer converts raw scores (logits) into probabilities:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

Properties:
- All outputs sum to 1.0
- Largest logit gets the highest probability
- Used with **cross-entropy loss** during training

### Cross-Entropy Loss

$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

Where $y_c$ is the true label (one-hot) and $\hat{y}_c$ is the predicted probability. For single-label classification, this simplifies to:

$$\mathcal{L} = -\log(\hat{y}_{\text{true class}})$$

---

## ImageNet and Benchmarking

### The ImageNet Dataset

**ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** drove the deep learning revolution in computer vision:

| Property | Value |
|----------|-------|
| Training images | 1.28 million |
| Validation images | 50,000 |
| Classes | 1,000 |
| Image size | Variable (resized to 224×224 or 256×256) |
| Years active | 2010–2017 |

### Historical Accuracy Progress

| Year | Model | Top-1 Accuracy | Top-5 Error |
|------|-------|---------------|-------------|
| 2012 | AlexNet | 63.3% | 16.4% |
| 2014 | VGG-16 | 74.4% | 7.3% |
| 2014 | GoogLeNet | 74.8% | 6.7% |
| 2015 | ResNet-152 | 77.8% | 3.6% |
| 2019 | EfficientNet-B7 | 84.4% | — |
| 2021 | ViT-H/14 | 88.6% | — |

**Top-1 accuracy:** Model's top prediction is correct.
**Top-5 accuracy:** Correct class is in the model's top 5 predictions.

---

## Data Augmentation

Training data is limited, but augmentation creates diverse views of existing images:

| Augmentation | Effect | When to Use |
|-------------|--------|-------------|
| Random crop | Scale/position invariance | Almost always |
| Horizontal flip | Left-right invariance | Most tasks (not text/handwriting) |
| Color jitter | Lighting invariance | Outdoor/variable lighting |
| Random rotation | Orientation invariance | Satellite, medical imaging |
| Cutout / Random erasing | Occlusion robustness | General robustness |
| Mixup | Regularization | When overfitting |
| RandAugment | Automated augmentation | State-of-the-art training |

**Key principle:** Augmentations should reflect real-world variations the model will encounter.

---

## Practical Training Pipeline

A typical image classification training pipeline:

1. **Data preparation:** Resize, normalize (ImageNet mean/std), split train/val/test
2. **Model selection:** Start with a pretrained backbone (ResNet-50, EfficientNet)
3. **Training:** SGD or AdamW optimizer, cosine learning rate schedule
4. **Augmentation:** RandAugment or AutoAugment
5. **Evaluation:** Top-1/Top-5 accuracy, confusion matrix, per-class metrics
6. **Deployment:** Export to ONNX or TorchScript for inference

### Common Training Hyperparameters

| Parameter | Typical Value |
|-----------|--------------|
| Batch size | 32–256 |
| Learning rate | 0.001–0.1 (with warmup) |
| Optimizer | SGD + momentum (0.9) or AdamW |
| Weight decay | 1e-4 to 5e-4 |
| Epochs | 90–300 |
| LR schedule | Cosine annealing or step decay |

---

## Key Takeaways

1. **Image classification** maps an entire image to a single class label — it's the foundation of computer vision
2. **Convolutions** provide parameter-efficient, translation-equivariant feature extraction
3. **Hierarchical features** emerge naturally: edges → parts → objects across CNN layers
4. **Softmax + cross-entropy** is the standard classification head and loss function
5. **ImageNet** was the benchmark that drove the deep learning revolution in vision
6. **Data augmentation** is essential — it provides regularization and improves generalization
7. **Pretrained models** (transfer learning) should be the starting point for most practical tasks

## References

- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS 2012*. [Paper](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- Russakovsky, O., et al. (2015). ImageNet Large Scale Visual Recognition Challenge. *IJCV*, 115(3), 211–252. [arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Cubuk, E. D., Zoph, B., Shlens, J., & Le, Q. V. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *NeurIPS 2020*. [arXiv:1909.13719](https://arxiv.org/abs/1909.13719)
