---
title: "Video Classification - Model layer"
css: styles.css
author: "Maria A"
description: "Model architectures and methods for video classification."
tags: ["deep learning", "computer vision", "video classification", "research"]
---
# Video Classification - Model layer

## Model Layer — Video Classification

### 1) Transformer family (spatiotemporal attention)

#### **VideoMAE (Masked Autoencoding for Video)**

* **Why it matters:** Strong pretrained features via masked video reconstruction; efficient “tubelet” tokens; SOTA-ish fine-tuning on Kinetics/SSv2 with modest compute.
* **Checkpoints (HF):** `MCG-NJU/videomae-base-finetuned-kinetics`, `MCG-NJU/videomae-large-finetuned-kinetics`, `MCG-NJU/videomae-base`
* **Key ideas:** Tubelet embedding (3D patches), masked pretraining, temporal positional encodings, lightweight heads for classification.

#### **TimeSformer (Divided Space–Time attention)**

* **Why it matters:** Pioneering pure-ViT for video; factorizes attention into spatial+temporal for scalability.
* **Checkpoints (HF):** `facebook/timesformer-base-finetuned-k400`, `facebook/timesformer-base-finetuned-k600`
* **Key ideas:** Divided attention (space then time), ImageNet-style patch embeddings, standard ViT blocks extended to time.

#### **X-CLIP (Video–Text contrastive)**

* **Why it matters:** Zero-shot and few-shot action recognition by aligning videos with text prompts (“a video of …”); excellent when labels are scarce.
* **Checkpoints (HF):** `microsoft/xclip-base-patch32`, `microsoft/xclip-base-patch16`
* **Key ideas:** CLIP-style contrastive pretraining extended to video (frame sampling + temporal pooling), prompt engineering for classes.

---

### 2) ConvNet & hybrid families (strong baselines, efficient)

#### **I3D / R(2+1)D / C3D (3D CNNs)**

* **Why it matters:** Classic, reliable baselines; great for teaching and controlled ablations.
* **Checkpoints:** Often via PyTorchVideo/torchvision (exportable to HF Datasets pipelines).
* **Key ideas:** 3D convolutions (or (2+1)D factorization) to model time and space jointly.

#### **SlowFast / X3D (meta-efficient 3D CNNs)**

* **Why it matters:** High accuracy–efficiency trade-offs; dual-pathway (Slow for semantics, Fast for motion); X3D scales width/height/frames smartly.
* **Checkpoints:** PyTorchVideo (`slowfast_r50`, `x3d_m`, etc.).
* **Key ideas:** Multi-rate pathways (SlowFast), principled compound scaling (X3D).

#### **Video Swin / UniFormer / MViT (hybrids)**

* **Why it matters:** Windowed or multiscale attention with Conv-like inductive bias; strong accuracy/latency.
* **Checkpoints:** Common in MMAction2/PyTorchVideo ecosystems; some ports exist on HF Hub.
* **Key ideas:** Hierarchical tokens, windowed attention (Swin), multiscale attention (MViT), Conv–Attention fusion (UniFormer).

---

### 3) Multimodal variants (optional but powerful)

#### **Audio–Visual models (AVSlowFast, fused Transformers)**

* **Why it matters:** Actions with salient sounds (e.g., musical instruments, speech) benefit from audio fusion.
* **How:** Concatenate or cross-attend visual tokens with log-mel spectrogram features.

#### **Video–Text (zero-shot)**

* **Why it matters:** Open-vocabulary classification; great when class taxonomy changes often (e.g., social media trends).
* **Examples:** **X-CLIP** (above), CLIP-based video pooling heads.

---

## Architectural innovations (what to teach and why)

* **3D Conv vs (2+1)D factorization:** 3D Conv models motion directly; (2+1)D reduces parameters by separating spatial and temporal convs.
* **Divided space–time attention (TimeSformer):** Improves scalability by factorizing attention; helps on long clips.
* **Tubelet tokens (VideoMAE/ViViT):** 3D patches lower token count vs per-frame patches → faster training/inference.
* **Masked video pretraining (VideoMAE):** Learns robust motion/appearance features without labels → superior fine-tuning.
* **Dual-pathway (SlowFast):** Explicitly models fast-changing motion and slow semantics.
* **Multiscale attention (MViT/Video Swin):** Hierarchical tokens and pooled attention handle long contexts efficiently.
* **Contrastive video–text (X-CLIP):** Open-vocabulary actions via prompts; strong zero-shot transfer.

---

## Short code snippets (Hugging Face `transformers`)

> These examples expect inputs as a **list of PIL frames** or a decoded frame tensor shaped `(T, C, H, W)`. Install `decord` or `pyav` to read videos. Most processors handle normalization for you.

### A) **VideoMAE** — supervised fine-tuning / inference

```python
from transformers import AutoImageProcessor, VideoMAEForVideoClassification
from PIL import Image

ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
processor = AutoImageProcessor.from_pretrained(ckpt)
model = VideoMAEForVideoClassification.from_pretrained(ckpt)

# frames: list[PIL.Image] length T (e.g., 16/32), already resized/cropped if you want
inputs = processor(frames, return_tensors="pt")  # pixel_values shape: (1, T, C, H, W)
with torch.no_grad():
    logits = model(**inputs).logits  # (1, num_labels)
pred = logits.softmax(-1).argmax(-1)
```

### B) **TimeSformer** — transformer with divided attention

```python
from transformers import AutoImageProcessor, TimeSformerForVideoClassification

ckpt = "facebook/timesformer-base-finetuned-k400"
processor = AutoImageProcessor.from_pretrained(ckpt)
model = TimeSformerForVideoClassification.from_pretrained(ckpt)

inputs = processor(frames, return_tensors="pt")  # (1, T, C, H, W)
with torch.no_grad():
    logits = model(**inputs).logits
```

### C) **Zero-shot with X-CLIP** — open-vocabulary video classification

```python
import torch
from transformers import AutoProcessor, XCLIPModel

ckpt = "microsoft/xclip-base-patch32"
processor = AutoProcessor.from_pretrained(ckpt)
model = XCLIPModel.from_pretrained(ckpt)

class_names = ["playing guitar", "cutting vegetables", "swimming"]
text_inputs = processor(text=[f"a video of {c}" for c in class_names], padding=True, return_tensors="pt")

# frames: list[RGB PIL] or tensor (T, C, H, W)
video_inputs = processor(videos=frames, return_tensors="pt")

with torch.no_grad():
    outputs = model(**video_inputs, **text_inputs)
    # similarity (B, num_text)
    sims = outputs.logits_per_video.softmax(-1)
    pred_idx = sims.argmax(-1).item()
    print(class_names[pred_idx])
```

### D) **High-level pipeline** (quickstart)

```python
from transformers import pipeline
clf = pipeline("video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")

# You can pass a path/URL if decord/pyav is installed, or a list of PIL frames
out = clf("path/to/clip.mp4", top_k=5)
print(out)  # [{'label': 'archery', 'score': 0.92}, ...]
```

### E) **Training sketch** (VideoMAE) — minimal Trainer loop

```python
from transformers import TrainingArguments, Trainer
# dataset returns dict with "video": list[PIL], "label": int
def preprocess(ex):
    px = processor(ex["video"], return_tensors="pt")["pixel_values"].squeeze(0)
    return {"pixel_values": px, "labels": ex["label"]}

train_ds = raw_train.map(preprocess)
eval_ds  = raw_val.map(preprocess)

args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=2, per_device_eval_batch_size=2,
    learning_rate=3e-5, num_train_epochs=5, fp16=True, evaluation_strategy="epoch"
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()
```

---

## When to pick what (fast guidance)

* **Few labels / small GPU:** *VideoMAE-B fine-tune* on your dataset; start with `T=16`, 224², mixed precision.
* **Need zero-shot taxonomy:** *X-CLIP* with prompt templates; optionally calibrate with a handful of labeled clips.
* **Latency-sensitive deployment:** *X3D-M* / *R(2+1)D-18* via PyTorchVideo; export to ONNX/TensorRT.
* **Longer actions / rich motion:** Increase `T` (e.g., 32–64) and use *TimeSformer* or *VideoMAE-L* if VRAM allows.
* **Audio cues matter:** Add an audio branch (log-mel) and fuse late (concat) or via cross-attention.
