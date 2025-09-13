---
title: "Visual Question Answering (VQA) - Evaluation layer"
css: styles.css
author: "Maria A"
description: "Evaluation metrics and methods for visual question answering."
tags: ["deep learning", "computer vision", "visual question answering", "research"]
---

# Visual Question Answering (VQA) - Evaluation layer

## Core metrics (what & when)

### 1) VQA “Soft Accuracy” (VQA v2, VizWiz, OK-VQA commonly)

* **What:** Consensus scoring with 10 human answers:

  $$
  \text{acc} = \min\left(\frac{\#\text{humans agreeing with pred}}{3},\ 1\right)
  $$
* **When:** Default for open-ended VQA benchmarks where multiple phrasings are acceptable.
* **Insight:** Rewards agreement with humans; robust to synonyms/typos after normalization.

### 2) Exact Match (EM) / Normalized EM

* **What:** Binary 0/1 after lowercasing, stripping punctuation/articles (“a/an/the”), collapsing whitespace.
* **When:** Tighter evaluation for closed-vocab, short answers (yes/no, numbers, colors) or internal QA checks.
* **Insight:** Precision of literal matching; great for unit tests and regression checks.

### 3) Token-level F1 (precision/recall on tokens)

* **What:** Overlap of predicted vs. gold tokens (SQuAD-style).
* **When:** Free-form answers with multiple tokens; complements EM.
* **Insight:** Partial credit for near-misses; sensitive to verbosity.

### 4) **ANLS** (Average Normalized Levenshtein Similarity) — TextVQA/DocVQA

* **What:** $\text{ANLS} = \text{avg}_i \max(0, 1 - \frac{ED(\hat{y}_i, y_i)}{\max(|\hat{y}_i|, |y_i|)})$
* **When:** OCR-heavy tasks where minor string diffs matter (menus, receipts, signs).
* **Insight:** Graded similarity score tolerant to small edit distances.

### 5) GQA diagnostics (beyond accuracy)

* **What:** **Accuracy**, **Consistency** (same reasoning → same answer), **Plausibility** (answer in vocabulary of image), **Validity** (well-formed), **Grounding**.
* **When:** Compositional reasoning or multi-hop datasets (GQA).
* **Insight:** Separates “got it right” from “understood it” (consistency/grounding).

### 6) Answerability / Unanswerable rate (VizWiz)

* **What:** Accuracy on “unanswerable” detection + standard accuracy on answerable subset.
* **When:** Real-world/noisy images; end-users need “I don’t know.”
* **Insight:** Avoids hallucinations; calibrates abstention.

### 7) Calibration metrics (ECE, Brier score)

* **What:** **ECE** (Expected Calibration Error) bins predicted confidence vs. empirical accuracy; **Brier** = mean squared error on probabilities.
* **When:** Human-facing systems; abstention thresholds; risk-sensitive apps.
* **Insight:** Are confidences trustworthy?

### 8) Efficiency (latency, throughput, memory)

* **What:** Tokens/s, images/s, peak VRAM.
* **When:** Productization & batch-serving.
* **Insight:** Sizing, cost, SLO compliance.

---

## Visualization & inspection (how to “see” what the model used)

* **Attention rollout (ViT/CLIP/ViLT):** Aggregate self-attentions across layers → heatmap over image patches.
  *Use when:* Patch-based encoders; fast, model-native.
* **Cross-attention maps (BLIP-2/InstructBLIP):** Visualize Q-Former or decoder cross-attn weights onto image tokens.
  *Use when:* Explaining *why* an LLM concluded an answer.
* **Grad-CAM / Score-CAM (CNN backbones):** Class-activation style maps for answer logits or pre-answer heads.
  *Use when:* CNN-based encoders or region-feature pipelines.
* **OCR overlays (TextVQA/DocVQA):** Show recognized words + confidences; highlight words weighted by attention.
  *Use when:* Answers depend on text; quickly catches OCR failure.
* **Region/box visualization (legacy region-features / DETR):** Draw detected objects (labels/scores) used as inputs.
  *Use when:* Bottom-up attention or grounding analyses.

---

## Python snippets (HF + friends)

### A) VQA v2 “soft accuracy” + normalization

```python
import re
from collections import Counter

_ARTICLES = {"a", "an", "the"}
def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(tokens)

def vqa_soft_acc(pred: str, gt_answers: list[str]) -> float:
    pred_n = _normalize(pred)
    gt_norm = [_normalize(a) for a in gt_answers]
    counts = Counter(gt_norm)
    agree = counts.get(pred_n, 0)
    return min(1.0, agree / 3.0)  # VQA v2 scoring

# Example:
# pred = "two"; gt_answers = ["Two", "2", "two", "three", ...]  # len==10
# print(vqa_soft_acc(pred, gt_answers))
```

### B) Token-level F1 (SQuAD-style) for free-form VQA

```python
def token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common = Counter(p) & Counter(g)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    prec = num_same / len(p)
    rec  = num_same / len(g)
    return 2 * prec * rec / (prec + rec)
```

### C) **ANLS** (TextVQA/DocVQA)

```python
def edit_distance(a: str, b: str) -> int:
    # Levenshtein (O(nm)) for clarity
    n, m = len(a), len(b)
    dp = list(range(m+1))
    for i in range(1, n+1):
        prev, dp[0] = dp[0], i
        for j in range(1, m+1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]

def anls(pred: str, golds: list[str]) -> float:
    pred_n = _normalize(pred)
    best = 0.0
    for g in golds:
        g_n = _normalize(g)
        M = max(len(pred_n), len(g_n)) or 1
        sim = 1.0 - edit_distance(pred_n, g_n) / M
        best = max(best, max(0.0, sim))  # clamp at 0
    return best
```

### D) Expected Calibration Error (ECE) for VQA (binary or one-vs-rest)

```python
import numpy as np

def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    # probs: (N,) predicted confidence of chosen answer
    # labels: (N,) 1 if correct else 0
    bins = np.linspace(0, 1, n_bins+1)
    ece_val = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i+1])
        if m.any():
            conf = probs[m].mean()
            acc  = labels[m].mean()
            ece_val += (m.mean()) * abs(conf - acc)
    return ece_val
```

### E) Attention rollout heatmap (ViT) — quick visual attribution

```python
import torch, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel

model_id = "google/vit-base-patch16-224"
proc = AutoImageProcessor.from_pretrained(model_id)
vit = AutoModel.from_pretrained(model_id, output_attentions=True).eval()

img = Image.open("demo.jpg").convert("RGB")
enc = proc(images=img, return_tensors="pt")
with torch.no_grad():
    out = vit(**enc)

# out.attentions: list[L](B, heads, tokens, tokens)
attns = [a.squeeze(0).mean(0) for a in out.attentions]  # avg over heads
roll = torch.eye(attns[0].size(-1))
for A in attns:
    A = A + torch.eye(A.size(-1))  # add identity (residual)
    A = A / A.sum(dim=-1, keepdim=True)
    roll = A @ roll
# CLS attention to patches (exclude CLS token at 0)
cls_to_patches = roll[0, 1:]
h = w = int(np.sqrt(cls_to_patches.numel()))
heat = cls_to_patches.reshape(h, w).cpu().numpy()
heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

# upscale & overlay
img_resized = img.resize((224, 224))
plt.imshow(img_resized)
plt.imshow(heat, alpha=0.4, extent=(0,224,224,0))
plt.axis("off"); plt.tight_layout(); plt.show()
```

### F) Cross-attention visualization (BLIP-2 Q-Former) — indicative

```python
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
model_id = "Salesforce/blip2-flan-t5-xl"
proc = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id, output_attentions=True
).eval()

img = Image.open("scene.jpg").convert("RGB")
prompt = "Q: How many umbrellas are visible? A:"
inputs = proc(images=img, text=prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=8, output_attentions=True, return_dict_in_generate=True)

# Depending on version, cross-attn may be exposed via model.qformer.* or decoder.*
# For illustration (pseudo-access):
# cross = model.qformer_outputs.cross_attentions  # list[L](B, heads, Q, image_tokens)
# heat = cross[-1].mean(1).squeeze(0).mean(0)     # avg heads, avg queries
# Map heat from image tokens back to ViT patch grid and overlay as in E)
```

### G) OCR overlay for TextVQA/DocVQA (Donut or OCR pipeline)

```python
from PIL import Image, ImageDraw, ImageFont

# Suppose you have OCR results: [{"text": "soup", "bbox": [x1,y1,x2,y2], "score": 0.93}, ...]
img = Image.open("menu.jpg").convert("RGB")
draw = ImageDraw.Draw(img)
for w in ocr_words:  # produced by your OCR engine or donut intermediate
    x1,y1,x2,y2 = w["bbox"]
    draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
    draw.text((x1, y1-10), f"{w['text']} ({w['score']:.2f})", fill="red")
img.save("ocr_overlay.jpg")
```

---

## Practical guidance (tie metrics to datasets)

* **VQA v2 / OK-VQA / VizWiz:** Report **VQA soft accuracy**; for VizWiz add **answerability**.
* **GQA:** Report **Accuracy** + **Consistency/Plausibility/Validity**; include a **grounding/attention** visualization.
* **TextVQA/DocVQA:** Report **ANLS** (primary) + EM/F1 as secondary; include **OCR overlays**.
* **Productization:** Add **ECE** (calibration) + **latency/throughput**.
