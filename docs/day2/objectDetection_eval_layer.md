---
title: "Object Detection - Evaluation layer"
css: styles.css
author: "Maria A"
description: "Evaluation metrics and methods for object detection."
tags: ["deep learning", "computer vision", "object detection", "research"]
---
# Object Detection - Evaluation layer

### Core metrics for Object Detection (images)

**COCO-style Average Precision (AP / mAP)**

* **AP@\[.5:.95] (primary):** mean AP over IoU thresholds 0.50:0.95 (step 0.05).
  *Use when you want a single, strict quality number balancing classification + localization.*
* **AP50 / AP75:** AP at IoU=0.50 (lenient) and IoU=0.75 (strict).
  *AP50 = sensitivity to detection presence; AP75 ≈ localization precision.*
* **APS / APM / APL:** AP on Small/Medium/Large objects.
  *Reveals scale-specific strengths/weaknesses (e.g., tiny object recall).*

**Average Recall (AR)**

* **AR\@1/10/100:** max recall with at most N detections per image; also **ARS/ARM/ARL**.
  *Use to diagnose missed detections independent of precision.*

**Precision–Recall (PR) curves (per-class)**

* *Use when you need threshold selection and class-wise behavior.*

**Latency/Throughput**

* **ms/image, FPS, memory**
  *Operational KPI; evaluate at the same input size used in production.*

---

### When to use which

* **Model selection/reporting:** AP@\[.5:.95] + AP50/AP75 + APS/APM/APL.
* **Debugging misses:** AR\@100 + per-class PR curves (find low-recall classes).
* **Deployment tuning:** Choose confidence/NMS thresholds from PR; report latency/FPS.
* **Open-vocabulary/grounded:** Standard AP on held-out “novel” classes; phrase grounding uses IoU ≥ 0.5 between predicted box for a phrase and the GT region.
* **Video detection / tracking:** Evaluate per-frame with AP, and if tracking IDs are produced, add **MOTA/MOTP/IDF1** (not shown here; image OD focus).

---

## Visualization Methods

**Bounding boxes & diagnostics**

* Draw predicted vs GT boxes, color by TP/FP/FN to spot failure modes (occlusion, small objects, duplicates).

**Grad-CAM / CAM (for CNN backbones)**

* Heatmap indicating spatial importance for a prediction (helps explain false positives/negatives).

**Attention maps (Transformers / DETR)**

* Visualize **cross-attention** of decoder queries over image features to understand object localization behavior.

---

## Python Snippets

### 1) COCO evaluation with `pycocotools` (mAP, AR)

```python
# pip install pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 1) Load annotations and predictions (COCO json format)
coco_gt = COCO("instances_val2017.json")
# predictions: list of dicts with fields: image_id, category_id, bbox=[x,y,w,h], score
coco_dt = coco_gt.loadRes("preds_val2017.json")

# 2) Standard COCO eval
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

# Summarize prints:
# AP @[.5:.95], AP50, AP75, APS/APM/APL, AR@1/10/100, ARS/ARM/ARL, etc.
```

**Tip:** For your own datasets, export GT to COCO JSON and convert model outputs into the prediction JSON to reuse this tooling.

---

### 2) Per-class Precision–Recall curves (IoU=0.5) with simple matching

```python
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, average_precision_score

# Suppose you’ve matched predictions to GTs at IoU>=0.5 and built class-wise lists:
# per_class_scores[c] = [scores...]
# per_class_labels[c] = [1 for TP, 0 for FP] (same length as scores)

def pr_for_class(scores, labels):
    precision, recall, thresh = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    return precision, recall, ap

per_class_ap = {}
for c, scores in per_class_scores.items():
    p, r, ap = pr_for_class(np.array(scores), np.array(per_class_labels[c]))
    per_class_ap[c] = ap
    # (Optionally plot p–r)
```

---

### 3) Bounding-box plotting (GT vs predictions, TP/FP/FN)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_boxes(img_path, gts, preds, iou_thresh=0.5):
    """
    gts:   list of dicts {'bbox':[x,y,w,h], 'cls':int}
    preds: list of dicts {'bbox':[x,y,w,h], 'cls':int, 'score':float}
    """
    im = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im)

    # naive IoU matcher to color TPs (green) and FPs (red)
    def iou(b1, b2):
        x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
        xa, ya = max(x1,x2), max(y1,y2)
        xb, yb = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
        inter = max(0, xb-xa)*max(0, yb-ya)
        union = w1*h1 + w2*h2 - inter + 1e-6
        return inter/union

    matched_gt = set()
    for pr in preds:
        best_iou, best_j = 0, -1
        for j, gt in enumerate(gts):
            if j in matched_gt or pr['cls']!=gt['cls']: continue
            i = iou(pr['bbox'], gt['bbox'])
            if i > best_iou: best_iou, best_j = i, j
        color = 'lime' if best_iou>=iou_thresh else 'red'
        x,y,w,h = pr['bbox']
        ax.add_patch(patches.Rectangle((x,y), w,h, fill=False, linewidth=2, edgecolor=color))
        ax.text(x, y-2, f"{pr['cls']} {pr['score']:.2f}", color=color, fontsize=10, backgroundcolor='k')
        if best_j!=-1 and best_iou>=iou_thresh:
            matched_gt.add(best_j)

    # Unmatched GTs (FN) in yellow
    for j, gt in enumerate(gts):
        if j in matched_gt: continue
        x,y,w,h = gt['bbox']
        ax.add_patch(patches.Rectangle((x,y), w,h, fill=False, linewidth=2, edgecolor='yellow'))
        ax.text(x, y-2, f"GT {gt['cls']}", color='yellow', fontsize=10, backgroundcolor='k')

    ax.axis('off'); plt.show()
```

---

### 4) Quick evaluation via 🤗 `pipeline` (for smoke tests)

```python
from transformers import pipeline
from PIL import Image

det = pipeline("object-detection", model="facebook/detr-resnet-50", threshold=0.25)
img = Image.open("image.jpg").convert("RGB")
preds = det(img)
# Convert preds to COCO format → evaluate with COCO tools as above
```

---

### 5) Grad-CAM with `torchcam` (CNN backbones)

```python
# pip install torchcam
import torch
from torchcam.methods import SmoothGradCAMpp   # or GradCAM, ScoreCAM, etc.
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
cam_extractor = SmoothGradCAMpp(model, target_layer="backbone.body.layer4.2.relu")  # last block

from PIL import Image
img = Image.open("image.jpg").convert("RGB")
x = F.to_tensor(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)[0]  # boxes, labels, scores

# Take the top detection and build a target for CAM (class index)
idx = out['scores'].argmax().item()
cls = int(out['labels'][idx])

# torchcam expects classification-like targets; for detection, many users
# approximate by using the class logit at the ROI head. Depending on lib version,
# you may need light adaptations or use captum with custom hooks.
cams = cam_extractor(class_idx=cls, scores=None, inputs=x)  # may vary by version

# Overlay CAM
plt.imshow(img); plt.imshow(cams[0][0].detach().cpu().numpy(), alpha=0.5); plt.axis('off'); plt.show()
```

*Note:* CAM tooling for detection heads often requires hooking the ROI head logits; libraries differ—expect small adjustments.

---

### 6) DETR cross-attention visualization (decoder → image features)

```python
from transformers import AutoModelForObjectDetection, AutoImageProcessor
import torch, matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model_id = "facebook/detr-resnet-50"
proc = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForObjectDetection.from_pretrained(model_id, output_attentions=True).eval()

img = Image.open("street.jpg").convert("RGB")
inputs = proc(images=img, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)

# out.decoder_attentions: tuple (layers) of [batch, heads, queries, tokens]
att = out.decoder_attentions[-1].mean(1)[0]   # last layer, mean over heads → [queries, tokens]

# Map token attention back to feature map (DETR uses CNN backbone → 2D tokens)
h, w = out.encoder_last_hidden_state.shape[1], None  # token layout depends on processor/backbone
# For simplicity, many demos reshape using intermediate feature map size known from processor:
H, W = inputs["pixel_values"].shape[-2]//32, inputs["pixel_values"].shape[-1]//32  # stride 32
q = att[0].reshape(H, W).cpu().numpy()  # attention for first query

plt.imshow(img); plt.imshow(np.clip(q, 0, q.max()), alpha=0.5); plt.axis('off'); plt.show()
```

*Note:* Exact reshaping depends on backbone stride and processor internals; use the hidden-state spatial size from the model’s backbone for precise mapping.

---

### 7) Measuring latency & throughput

```python
import torch, time
from transformers import pipeline
det = pipeline("object-detection", model="facebook/detr-resnet-50", device=0)  # GPU if available

import numpy as np
import cv2
im = cv2.imread("image.jpg")[:, :, ::-1]  # RGB

# Warmup
for _ in range(10): det(im)

# Timed
N=50; t0 = time.time()
for _ in range(N): det(im)
t1 = time.time()
print(f"Avg latency: {(t1-t0)/N*1000:.1f} ms, FPS: {N/(t1-t0):.1f}")
```

---

## Practical checklist

* **Report:** AP@\[.5:.95], AP50/AP75, APS/APM/APL, AR\@100, latency\@input-size.
* **Analyze:** per-class AP + PR; TP/FP/FN visual grids for hard images; size buckets.
* **Explain:** Grad-CAM (CNN) or cross-attention maps (DETR) on successes/failures.
* **Decide thresholds:** Use PR curves to pick confidence & NMS for the target precision/recall trade-off.
* **Be consistent:** Fix evaluation transforms, category mapping, and IoU thresholds across runs.
