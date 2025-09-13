---
title: "Image classification - Model layer"
css: styles.css
author: "Maria A"
description: "Model architectures and methods for image classification."
tags: ["deep learning", "computer vision", "image classification", "research"]
---
# Image classification - Model layer

## Hugging Face Model Zoo Options

### **CNN Families (classic baselines)**

* **ResNet-50 / ResNet-101**
  *Checkpoint:* `microsoft/resnet-50`
  *Why it matters:* Deep residual connections solved vanishing gradients, still strong baselines.

* **EfficientNet (B0–B7)**
  *Checkpoint:* `google/efficientnet-b0`
  *Why it matters:* Compound scaling (depth, width, resolution) → strong performance/efficiency trade-off.

---

### **Vision Transformers (modern default)**

* **ViT-Base / ViT-Large (patch16/224)**
  *Checkpoint:* `google/vit-base-patch16-224`
  *Why it matters:* First pure-transformer image classifier; competitive with CNNs when pretrained on large corpora.

* **DeiT (Data-efficient Image Transformer)**
  *Checkpoint:* `facebook/deit-base-distilled-patch16-224`
  *Why it matters:* Distillation tricks make transformers viable with less data; faster training.

* **Swin Transformer**
  *Checkpoint:* `microsoft/swin-base-patch4-window7-224`
  *Why it matters:* Hierarchical windows + shifting → better locality modeling than vanilla ViT.

---

### **Hybrid / Advanced Architectures**

* **ConvNeXt**
  *Checkpoint:* `facebook/convnext-base-224`
  *Why it matters:* CNN redesigned with transformer-era tricks (layer norm, GELU, large kernels).

* **BEiT (BERT for Images)**
  *Checkpoint:* `microsoft/beit-base-patch16-224`
  *Why it matters:* Masked image modeling (like MLM in NLP) → powerful self-supervised pretraining.

* **CLIP (multimodal, classification via zero-shot)**
  *Checkpoint:* `openai/clip-vit-base-patch32`
  *Why it matters:* Joint image–text embeddings enable zero-shot classification and flexible labeling.

---

## Architectural Innovations

* **CNNs (ResNet, EfficientNet):** Inductive biases (convolutions, pooling) → data-efficient, fast convergence.
* **ViTs (ViT, DeiT, Swin):** Attention-only, no convolutions; scalable with pretraining, interpretability via attention maps.
* **Hybrid (ConvNeXt):** CNNs reimagined with transformer-era training strategies.
* **Self-supervised Transformers (BEiT, MAE):** Learn representations without labels (masked image modeling).
* **Multimodal (CLIP):** Align vision and text → flexible zero-shot classification.

---

## Short Hugging Face Code Snippets

### 1) Standard supervised classification (ViT)

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset

# Load dataset
dataset = load_dataset("cifar10")

# Load model + processor
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Preprocess one example
inputs = processor(images=dataset["train"][0]["img"], return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(-1)
```

---

### 2) Data-efficient Distilled Transformer (DeiT)

```python
model = AutoModelForImageClassification.from_pretrained(
    "facebook/deit-base-distilled-patch16-224"
)
```

---

### 3) Zero-shot classification with CLIP

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("dog.jpg")
labels = ["a photo of a dog", "a photo of a cat"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
```

