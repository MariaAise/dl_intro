---
title: "Image classification - Evaluation layer"
css: styles.css
author: "Maria A"
description: "Model architectures and methods for image classification."
tags: ["deep learning", "computer vision", "image classification", "research"]
---
# Image classification - Evaluation layer

## Core metrics (what + when)

* **Accuracy (Top-1 / Top-k)**
  *Use when:* Classes are balanced or roughly so; you want a simple overall hit-rate.
  *Insight:* Fraction of samples where the correct label is ranked top-1 (or within top-k).

* **Precision / Recall / F1 (macro / weighted / per-class)**
  *Use when:* Class imbalance matters or minority classes are critical.
  *Insight:* Trade-off between false positives and false negatives; macro treats classes equally, weighted respects support.

* **AUROC (macro, one-vs-rest) & AUPRC**
  *Use when:* Severe imbalance, or you care about ranking quality (threshold-free).
  *Insight:* How well the model separates classes across thresholds; PR is especially informative under imbalance.

* **Log Loss (Cross-Entropy)**
  *Use when:* You care about *probability* quality (not just correctness).
  *Insight:* Penalizes overconfident wrong predictions; good for calibration checks and early stopping.

* **Matthews Correlation Coefficient (MCC)**
  *Use when:* Robust single-number summary under imbalance.
  *Insight:* Correlation between predictions and labels; balanced and informative even if classes are skewed.

* **Calibration Error (ECE / MCE)**
  *Use when:* Downstream decisions rely on calibrated probabilities.
  *Insight:* How close predicted confidences are to empirical accuracies.

> For most workshops: report **Top-1**, **Top-5**, **macro-F1**, and **log loss**. If imbalance is pronounced, add **macro-AUROC** and a **reliability diagram (ECE)**.

---

## Visualization methods (why + when)

* **Grad-CAM / Grad-CAM++ (CNNs, ConvNeXt)**
  *Why:* Localize the evidence for a prediction; sanity-check spurious correlations.
  *When:* Explaining a single prediction; model is convolutional or has conv-like final stages.

* **Attention Rollout / Attention Maps (ViTs, DeiT, Swin)**
  *Why:* Trace how information flows across transformer layers/heads.
  *When:* Transformer backbones; global context explanations.

* **Embedding Projections (t-SNE / UMAP) of penultimate features**
  *Why:* See class clusters, overlap, and outliers.
  *When:* Dataset diagnostics; curriculum design; failure analysis.

* **Confusion Matrix**
  *Why:* Identify which classes get mixed up.
  *When:* Always—fast, high signal.

*(Bounding-box plotting is a detection-specific tool; for classification, focus on CAMs, attention, embeddings, and confusion matrices.)*

---

## Python: metrics + evaluation loop (Hugging Face + PyTorch)

```python
import torch, numpy as np
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import v2 as T
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score, f1_score,
    log_loss, confusion_matrix, roc_auc_score
)

# 1) Data
ds = load_dataset("cifar10")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

train_t = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                     T.Normalize(mean=processor.image_mean, std=processor.image_std)])
eval_t  = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize(mean=processor.image_mean, std=processor.image_std)])

def with_transform(split_t):
    def _f(batch):
        batch["pixel_values"] = [split_t(img.convert("RGB")) for img in batch["img"]]
        return batch
    return _f

ds = ds.with_transform(with_transform(eval_t))

# 2) Model
id2label = {i: c for i, c in enumerate(ds["train"].features["label"].names)}
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(id2label),
    id2label=id2label, label2id=label2id
).eval()

# 3) Batched evaluation
def batched(iterable, n=32):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for batch_idx in batched(list(range(len(ds["test"]))), n=64):
        ims = torch.stack([ds["test"][i]["pixel_values"] for i in batch_idx])
        labs = torch.tensor([ds["test"][i]["label"] for i in batch_idx])
        out = model(pixel_values=ims)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        y_true.extend(labs.tolist())
        y_pred.extend(probs.argmax(-1).tolist())
        y_prob.append(probs.cpu().numpy())

y_prob = np.concatenate(y_prob, axis=0)

# 4) Metrics
top1 = accuracy_score(y_true, y_pred)
top5 = top_k_accuracy_score(y_true, y_prob, k=5, labels=list(range(y_prob.shape[1])))
macro_f1 = f1_score(y_true, y_pred, average="macro")
ce = log_loss(y_true, y_prob, labels=list(range(y_prob.shape[1])))
cm = confusion_matrix(y_true, y_pred)

# One-vs-rest macro AUROC (works if probs available)
try:
    auroc_macro = roc_auc_score(
        np.eye(y_prob.shape[1])[y_true], y_prob, average="macro", multi_class="ovr"
    )
except Exception:
    auroc_macro = float("nan")

print(f"Top-1: {top1:.4f} | Top-5: {top5:.4f} | Macro-F1: {macro_f1:.4f} | "
      f"LogLoss: {ce:.4f} | AUROC(macro, ovr): {auroc_macro:.4f}")
print("Confusion matrix shape:", cm.shape)
```

---

## Reliability diagram (ECE) — quick implementation

```python
def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece, bin_confs, bin_accs, bin_sizes = 0.0, [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            bin_conf = conf[mask].mean()
            bin_acc  = correct[mask].mean()
            w = mask.mean()
            ece += w * abs(bin_acc - bin_conf)
            bin_confs.append(bin_conf); bin_accs.append(bin_acc); bin_sizes.append(mask.sum())
    return ece, (bin_confs, bin_accs, bin_sizes)

ece, (bconfs, baccs, _) = expected_calibration_error(y_true, y_prob, n_bins=15)
print(f"ECE: {ece:.4f}")
```

*(Plotting those bars/lines → quick reliability diagram.)*

---

## Confusion matrix plot

```python
import matplotlib.pyplot as plt
import seaborn as sns

labels = ds["test"].features["label"].names
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", cbar=True, ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Normalized Confusion Matrix")
ax.set_xticks(np.arange(len(labels))+0.5); ax.set_yticks(np.arange(len(labels))+0.5)
ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels, rotation=0)
plt.tight_layout(); plt.show()
```

---

## Grad-CAM (CNN/ConvNeXt) — minimal example

> Works well with CNN backbones (e.g., `facebook/convnext-base-224`).
> Uses the `pytorch-grad-cam` package.

```python
# pip install grad-cam
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForImageClassification
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model_name = "facebook/convnext-base-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name).eval()

img = Image.open("example.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
pred = logits.argmax(-1).item()

# Pick a target conv block
target_layers = [model.convnext.features[-1]]  # last stage

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
rgb = np.array(img).astype(np.float32)/255.0
grayscale_cam = cam(input_tensor=inputs["pixel_values"], targets=[ClassifierOutputTarget(pred)])[0]
vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

plt.imshow(vis); plt.axis("off"); plt.title(f"Grad-CAM (pred={model.config.id2label[pred]})")
plt.show()
```

---

## Attention rollout (ViT/DeiT) — quick visualization

```python
import torch
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel

name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(name)
vit = AutoModel.from_pretrained(name, output_attentions=True).eval()

inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
    out = vit(**inputs)
# attentions: tuple of (num_layers) x (B, num_heads, tokens, tokens)
atts = torch.stack(out.attentions).squeeze(1)  # (L, H, T, T)

# Average heads, add identity (residual), normalize, then rollout
att = atts.mean(dim=1)  # (L, T, T)
eye = torch.eye(att.shape[-1])
att = [ (a + eye) / (a + eye).sum(dim=-1, keepdim=True) for a in att ]
rollout = att[0]
for a in att[1:]:
    rollout = rollout @ a  # (T, T)

# Class token attention to patches (exclude CLS=0)
cls_attn = rollout[0, 1:].reshape(int((rollout.shape[-1]-1)**0.5), -1).cpu().numpy()

plt.imshow(cls_attn, cmap="viridis"); plt.colorbar(); plt.title("ViT Attention Rollout (CLS → patches)")
plt.axis("off"); plt.show()
```

---

## Embedding projection (penultimate layer)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def extract_features(model, pixel_values):
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_hidden_states=True)
        # ViT: penultimate hidden state CLS token
        feats = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    return feats

# Collect a small eval batch
idx = list(range(512))
X = torch.stack([ds["test"][i]["pixel_values"] for i in idx])
y = np.array([ds["test"][i]["label"] for i in idx])

feats = extract_features(model, X)
Z = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(feats)

plt.figure(figsize=(6,5))
for c in np.unique(y):
    m = y==c
    plt.scatter(Z[m,0], Z[m,1], s=12, label=id2label[c], alpha=0.7)
plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("t-SNE of Penultimate Features"); plt.tight_layout(); plt.show()
```

---

### Quick recommendations

* **Balanced data:** Top-1/Top-5 + confusion matrix.
* **Imbalanced data:** Macro-F1, AUROC (ovr), AUPRC + per-class metrics.
* **Decision-making with probabilities:** Add **log loss** and **ECE** + reliability diagram.
* **Explainability for stakeholders:** Grad-CAM (CNNs) or attention rollout (ViTs) + a few exemplar plots.
