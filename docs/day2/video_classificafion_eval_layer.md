---
title: "Video Classification - Evaluation layer"
css: styles.css
author: "Maria A"
description: "Model architectures and methods for video classification."
tags: ["deep learning", "computer vision", "video classification", "research"]
---
# Video Classification - Evaluation layer

### Core metrics (what to use and why)

* **Top-1 / Top-k Accuracy (single-label)**
  *Use when each clip has exactly one class.*
  **Insight:** How often the correct class is ranked #1 (or within the top-k, e.g., k=5). Top-k is useful when classes are visually similar.

* **F1 (macro / micro) & Precision/Recall (single- or multi-label)**
  *Use when class imbalance exists, or errors have asymmetric cost.*
  **Insight:**

  * **Macro-F1:** treats all classes equally → good for skewed datasets.
  * **Micro-F1:** aggregates over all instances → good for overall performance with imbalance.

* **mAP / Average Precision (multi-label)**
  *Use when clips can have multiple labels (e.g., Charades, EPIC actions).*
  **Insight:** Area under Precision–Recall curve per class, then averaged → robust to threshold choice and strong under imbalance.

* **Balanced Accuracy**
  *Use when severe class imbalance but single-label classification.*
  **Insight:** Mean of per-class recalls → less biased toward dominant classes.

* **AUROC / AUPRC (diagnostics)**
  *Use for threshold-free comparison of probabilistic outputs; especially with heavy imbalance.*
  **Insight:** Ranking quality across thresholds; **AUPRC** more informative than AUROC under rare positives.

* **Calibration (ECE / Reliability)**
  *Use in production settings where scores drive decisions.*
  **Insight:** Are predicted probabilities aligned with empirical accuracy?

* **Clip → Video Aggregation**
  *Use when you evaluate on untrimmed videos (sample multiple clips).*
  **Insight:** Report both **clip-level** and **video-level** (e.g., average logits or majority vote) to reflect deployment.

---

### Visualization methods (to understand “why”)

* **Grad-CAM / Grad-CAM++ (per-frame heatmaps)**
  Works with 3D CNNs or by applying to spatial blocks in video transformers. Visualize *where* the model looks in frames.

* **Attention rollout / attention maps (Transformers: TimeSformer/VideoMAE)**
  Aggregate attention across layers/heads to get spatiotemporal importance. Great for ViT-style models.

* **Saliency → Bounding boxes (diagnostic)**
  Threshold heatmaps and draw **proxy boxes** around the most salient regions. (Not the same as detection, but helps sanity-check focus.)

* **Per-class Confusion Matrix (single-label)**
  See which classes the model confuses; pair with **per-class PR curves** for multi-label.

---

## Python snippets — metrics & visualization

> Snippets use `torch`, `torchvision`, `scikit-learn`, `torchmetrics`, `transformers`, and optional `pytorch-grad-cam`. Treat them as templates you can drop into your trainer/eval loop.

### 1) Top-k Accuracy, Macro-F1 (single-label)

```python
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

num_classes, topk = 101, 5
top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1).to("cuda")
topk_acc = MulticlassAccuracy(num_classes=num_classes, top_k=topk).to("cuda")
f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro").to("cuda")

# logits: (B, C), labels: (B,)
def update_metrics(logits, labels):
    preds = torch.softmax(logits, dim=-1)
    top1.update(preds, labels)
    topk_acc.update(preds, labels)
    f1_macro.update(preds.argmax(dim=-1), labels)

# end of epoch
print({"top1": top1.compute().item(), "top5": topk_acc.compute().item(), "f1_macro": f1_macro.compute().item()})
top1.reset(); topk_acc.reset(); f1_macro.reset()
```

### 2) mAP for multi-label (Charades-style)

```python
import torch
from torchmetrics.classification import MultilabelAveragePrecision

num_classes = 157
map_metric = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to("cuda")

# logits: (B, C) ; targets: (B, C) with {0,1} multi-hot
def update_map(logits, targets):
    probs = torch.sigmoid(logits)
    map_metric.update(probs, targets.int())

print({"mAP_macro": map_metric.compute().item()})
map_metric.reset()
```

### 3) Calibration: Expected Calibration Error (ECE)

```python
import torch
import numpy as np

def expected_calibration_error(probs, labels, n_bins=15):
    # probs: (N, C) after softmax; labels: (N,)
    confidences, preds = probs.max(dim=1)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for b in range(n_bins):
        in_bin = (confidences > bins[b]) & (confidences <= bins[b+1])
        if in_bin.any():
            acc_bin = (preds[in_bin] == labels[in_bin]).float().mean()
            conf_bin = confidences[in_bin].mean()
            ece += (in_bin.float().mean()) * (conf_bin - acc_bin).abs()
    return ece.item()
```

### 4) Confusion matrix (single-label) & PR curves (multi-label)

```python
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

# Single-label
cm = confusion_matrix(y_true=labels.cpu().numpy(), y_pred=logits.argmax(-1).cpu().numpy(), labels=list(range(num_classes)))
# Plot cm with matplotlib imshow; normalize by row for readability.

# Multi-label (per-class AP)
probs = torch.sigmoid(logits).cpu().numpy()
targets = targets.cpu().numpy()
ap_per_class = [average_precision_score(targets[:, c], probs[:, c]) for c in range(num_classes)]
```

### 5) Grad-CAM on video (per-frame heatmaps with 3D CNN or 2D spatial blocks)

```python
# pip install pytorch-grad-cam
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

model.eval()
# target_layer: pick a late spatial layer (e.g., last conv block of a 3D CNN),
# or for a TimeSformer/VideoMAE port, the patch-embedding or a spatial attention block.
target_layers = [model.backbone.layer4[-1]]  # example for a ResNet3D-like backbone

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# frames_np: (T, H, W, 3) in [0,1]; input_tensor: (1, T, C, H, W) for your model
targets = [ClassifierOutputTarget(int(predicted_class_id))]
grayscale_cams = cam(input_tensor, targets=targets)  # shape: (1, T, H, W) normalized

vis_frames = []
for t in range(grayscale_cams.shape[1]):
    rgb = frames_np[t]
    heat = grayscale_cams[0, t]
    vis = show_cam_on_image(rgb, heat, use_rgb=True)  # returns uint8
    vis_frames.append(vis)
# Save as a video/gif with imageio
```

### 6) Attention rollout (TimeSformer/ViT-style)

```python
# Extract attention from each block, average heads, multiply (rollout) across layers.
# Pseudocode (depends on model exposing attn weights):
def attention_rollout(attn_list, discard_ratio=0.0):
    # attn_list: list of tensors [(B, heads, tokens, tokens), ...]
    rollout = None
    for attn in attn_list:
        a = attn.mean(dim=1)  # average heads
        if discard_ratio > 0:
            flat = a.view(a.size(0), -1)
            num = int(flat.size(1) * discard_ratio)
            vals, idx = torch.topk(flat, num, dim=1, largest=False)
            a = a.clone()
            a.view(a.size(0), -1).scatter_(1, idx, 0)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else rollout @ a
    return rollout  # (B, tokens, tokens)

# Map token attention back to frame+patch grid to visualize spatiotemporal saliency.
```

### 7) Saliency → proxy bounding boxes (diagnostic)

```python
import cv2
import numpy as np

# heat: (H, W) in [0,1]; create binary mask by threshold, then draw box
def heatmap_to_bbox(heat, thr=0.6):
    mask = (heat >= thr).astype(np.uint8)*255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (x, y, w, h)

frame = (frame_rgb*255).astype(np.uint8)
bbox = heatmap_to_bbox(heat)
if bbox:
    x,y,w,h = bbox
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)  # diagnostic box
```

### 8) Clip→Video aggregation (untrimmed evaluation)

```python
import torch
import numpy as np

# Suppose we sample N clips per long video; collect logits per clip
# logits_list: list of (C,) tensors for each clip
video_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # average pooling
# Alternative: majority vote on argmax, or logit max-pooling per class
video_pred = video_logits.softmax(-1).argmax().item()
```

---

## How to choose metrics (quick guide)

* **Balanced single-label:** report **Top-1, Top-k**, and **macro-F1**.
* **Imbalanced single-label:** add **Balanced Accuracy** and **per-class F1**.
* **Multi-label:** use **mAP (macro)** + **per-class AP**, and **micro-F1**.
* **Production:** track **ECE** (calibration), **AUPRC** (rare positives), and **confusion matrix** to prioritize fixes.
* **Long videos:** report **clip-level** and **video-level** metrics (with your chosen aggregation).
* **Explainability:** provide **Grad-CAM**/**attention rollout** videos for a few TP/FP/FN examples to guide error analysis.
