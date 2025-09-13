---
title: "Object Detection - Model layer"
css: styles.css
author: "Maria A"
description: "Model architectures and methods for object detection."
tags: ["deep learning", "computer vision", "object detection", "research"]
---
# Object Detection - Model layer
This section covers the **model layer** for object detection, focusing on architectures, key innovations, and practical implementations.

### Canonical DETR family (Transformer-based, set prediction)

* **DETR (ResNet backbone)** — `facebook/detr-resnet-50`, `facebook/detr-resnet-101`
  *Bipartite matching (Hungarian), NMS-free, strong with multi-scale aug; slower to converge vs one-stage.*
* **DETR (ViT backbone, compact)** — `facebook/detr-resnet-50` (swap backbone in fine-tune) or **YOLOS** (see below).
* **Deformable DETR** — `SenseTime/deformable-detr`
  *Multi-scale deformable attention → much faster convergence and small-object gains.*
* **DN-/DAB-/DINO-DET** — (community ports) e.g., `IDEA-Research/dino-5scale`
  *Query denoising, anchor refinement, stronger training signals → SOTA-ish mAP on COCO.*

### ViT-style single-stage

* **YOLOS** — `hustvl/yolos-tiny`, `hustvl/yolos-small`, `hustvl/yolos-base`
  *ViT adapted for detection; light and easy to fine-tune via 🤗 Transformers.*

### Open-Vocabulary / Language-Grounded Detectors

* **OWL-ViT (zero-shot OD)** — `google/owlvit-base-patch32`, `google/owlvit-large-patch14`
  *Text queries → detect novel categories without box supervision.*
* **Grounding DINO** — `IDEA-Research/grounding-dino-base`, `groundingdino/swint-ogc`
  *Phrase grounding + detection; strong zero-shot and promptable OD.*
* **GLIP / OWLv2 (if needed)** — community checkpoints exist on HF Hub for open-vocab detection.

### High-throughput one-stage (non-Transformer backbones, widely used)

* **YOLO family** — (Ultralytics exports on Hub; inference via `ultralytics` or ONNX) e.g., `ultralytics/yolov8n`, `ultralytics/yolov8l`
  *Real-time, strong engineering; train outside Transformers API or via custom loaders.*
* **RT-DETR** — `PaddlePaddle/RT-DETR-R50` (ports available)
  *Real-time DETR variant balancing accuracy/latency.*

### Domain / Task-specific

* **Oriented/Rotated** — (DOTA/xView ports on Hub; e.g., Rotated-YOLO, Oriented-RCNN)
  *Adds angle to boxes; aerial/remote sensing.*
* **Instance Segmentation (box + mask)** — `facebook/mask2former-swin-large-coco-instance` (if boxes + masks needed).
* **Video OD (per-frame baseline)** — use above image detectors frame-wise; trackers (e.g., ByteTrack) add IDs externally.

---

## Architectural Innovations (cheat-sheet)

* **Two-stage CNNs (Faster/Mask R-CNN):** region proposal → per-ROI heads; accurate, heavier, mature ecosystem.
* **One-stage CNNs (YOLO/RetinaNet):** dense predictions, focal loss; real-time, excellent engineering & tools.
* **Anchor-free (FCOS/CenterNet/YOLOX head):** predict centers/boxes directly; simpler label assignment.
* **Transformers for detection (DETR):** set prediction with **Hungarian matching**, global attention, **NMS-free**.
* **Deformable attention:** sparse, multi-scale sampling → faster training, better small-object recall.
* **Query tricks (DN-, DAB-, DINO-DETR):** denoising, anchor refinement, better query initialization → big mAP bumps.
* **Open-vocab / grounded:** align vision with text (CLIP-like) → **zero-shot** detection from prompts (OWL-ViT, Grounding DINO).
* **Real-time optimizations:** lightweight necks/heads, quantization, dynamic shapes, knowledge distillation.
* **Rotated boxes / oriented heads:** regress angle for aerial/OCR/logistics.
* **Video extensions:** temporal features or simple per-frame detect + tracker for strong baselines.

---

## Short code snippets (🤗 Transformers / pipelines)

### 1) Quick inference — classic object detection (DETR)

```python
from transformers import pipeline
detector = pipeline("object-detection", model="facebook/detr-resnet-50")
preds = detector("image.jpg")  # [{'score':0.99,'label':'person','box':{...}}, ...]
```

### 2) Zero-shot object detection — text-prompted (OWL-ViT)

```python
from transformers import pipeline
zs_detector = pipeline("zero-shot-object-detection", model="google/owlvit-base-patch32")
preds = zs_detector(
    "image.jpg",
    candidate_labels=["handbag", "coat", "sneaker", "traffic light"]
)
```

### 3) Grounded phrase detection (Grounding DINO)

```python
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image

model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

image = Image.open("street.jpg").convert("RGB")
inputs = processor(images=image, text="a person riding a bicycle, a red bag", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
results = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids, box_threshold=0.3, text_threshold=0.25, target_sizes=[image.size[::-1]]
)
# results[0]['boxes'], ['labels'], ['scores']
```

### 4) Fine-tuning DETR on COCO-style data

```python
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainingArguments
from datasets import load_dataset

proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)

ds = load_dataset("coco", "2017", split={"train":"train", "val":"validation"})
def preprocess(example):
    return proc(images=example["image"], annotations={"annotations": example["annotations"]}, return_tensors="pt")
train = ds["train"].with_transform(lambda e: preprocess(e))
val   = ds["val"].with_transform(lambda e: preprocess(e))

args = TrainingArguments(output_dir="detr-ft", per_device_train_batch_size=2, num_train_epochs=12,
                         lr_scheduler_type="cosine", weight_decay=0.05, fp16=True, evaluation_strategy="steps")
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val)
# trainer.train()
```

### 5) YOLOS (ViT-style single-stage) inference

```python
from transformers import AutoImageProcessor, YolosForObjectDetection
from PIL import Image
import torch

proc = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
image = Image.open("image.jpg").convert("RGB")

inputs = proc(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
results = proc.post_process_object_detection(outputs, target_sizes=[image.size[::-1]], threshold=0.3)[0]
```

### 6) Per-frame video detection (baseline)

```python
from transformers import pipeline
import cv2

detector = pipeline("object-detection", model="facebook/detr-resnet-50")
cap = cv2.VideoCapture("clip.mp4")
while True:
    ok, frame = cap.read()
    if not ok: break
    preds = detector(frame[:, :, ::-1])  # BGR→RGB
    # draw boxes / write to file...
```

---

## When to pick what (quick guide)

* **Strong baseline & easy training:** *Deformable DETR* (SenseTime) or *DINO-DETR* ports.
* **Fast prototyping & inference:** *DETR pipeline* or *YOLOS* for pure 🤗 workflow; *YOLOv8* via Ultralytics for speed/tooling.
* **Zero-shot / new labels without annotation:** *OWL-ViT* (simple) → *Grounding DINO* (stronger grounding).
* **Aerial/rotated:** use oriented detectors or YOLO variants with angle heads.
* **Video:** per-frame detect + tracker (ByteTrack/OC-SORT) for robust baselines; add temporal models later if needed.
