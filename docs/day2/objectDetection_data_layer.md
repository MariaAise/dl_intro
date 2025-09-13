---
title: "Object Detection - Data layer"
css: styles.css
author: "Maria A"
description: "Datasets and preprocessing for object detection."
tags: ["deep learning", "computer vision", "object detection", "research"]
---
# Object Detection - Data layer

## Datasets — Benchmarks & Sources

### **COCO 2017 (Object Detection)**

* **What it is:** \~118k train / 5k val / 20k test images with \~80 object classes, crowd labels, instance masks, keypoints.
* **Why it matters:** The de-facto general-purpose benchmark; nearly all modern detectors report mAP here.
* **Quirks:** Many small objects; “iscrowd” regions; multiple annotations per image; long-tail frequency even within 80 classes.
* **Where:** Hugging Face (`coco`, config `2017` → subsets `train`, `validation`); also via torchvision’s `CocoDetection`.

### **Pascal VOC 2007/2012**

* **What it is:** \~20 object classes; simpler images; VOC07 (\~5k trainval / 5k test), VOC12 (\~11k trainval).
* **Why it matters:** Lightweight baseline; great for quick iterations and educational demos.
* **Quirks:** Older annotation style; fewer small objects; AP computed at IoU=0.5 in classic protocol.
* **Where:** Hugging Face (`pascal_voc`).

### **LVIS v1 (Long-tail Visual Instance Segmentation)**

* **What it is:** \~1k+ categories with extreme class imbalance; instance segmentation + boxes.
* **Why it matters:** Tests open-vocabulary/long-tail capability and rare class generalization.
* **Quirks:** Highly skewed label distribution; requires special sampling/reweighting.
* **Where:** Hugging Face (`lvis`).

### **Open Images V6 (Detection)**

* **What it is:** Millions of images with \~600 categories; image-level + box labels; hierarchical ontology.
* **Why it matters:** Scale and label hierarchy; good for pretraining and robustness.
* **Quirks:** Noisy labels; partial annotations; class hierarchy requires careful mapping.
* **Where:** Hugging Face (`open_images_v6` with `object_detection` subset).

### **Objects365**

* **What it is:** \~365 categories; \~600k images richly annotated with boxes.
* **Why it matters:** Large-scale pretraining to boost downstream COCO/LVIS performance.
* **Quirks:** Licensing/hosting can be heavier; class names differ from COCO.
* **Where:** Hugging Face (`objects365`) or official site (account needed).

### **Cityscapes (Boxes from Polygons)**

* **What it is:** Urban street scenes; fine pixel labels for 8 categories (19 for segmentation). Boxes can be derived or use detection splits from community repos.
* **Why it matters:** Driving domain; medium-scale; consistent viewpoint.
* **Quirks:** Predominantly large objects; strong class bias (cars/persons).
* **Where:** Hugging Face (`cityscapes`) for segmentation polygons; convert to boxes or use detection versions from forks.

### **BDD100K (Detection)**

* **What it is:** 100k driving images with detection, tracking, and lane/seg labels.
* **Why it matters:** Broad driving conditions (night, weather); good for domain robustness.
* **Quirks:** Class set differs from COCO; time-of-day imbalance.
* **Where:** Hugging Face (`bdd100k`).

### **KITTI (Detection)**

* **What it is:** On-road images labeled for Car/Pedestrian/Cyclist in camera view.
* **Why it matters:** Classic autonomous driving detection benchmark.
* **Quirks:** Small dataset; strict evaluation protocol; depth cues from stereo.
* **Where:** Hugging Face (`kitti`).

### **CrowdHuman**

* **What it is:** \~15k train images focusing on crowded human boxes (head/full body/visible body).
* **Why it matters:** Tests NMS/duplicate handling in crowded scenes.
* **Quirks:** Heavy occlusion; dense overlaps stress post-processing.
* **Where:** Hugging Face (`crowdhuman`).

### **Aerial/Remote Sensing (DOTA / xView)**

* **What it is:** DOTA: oriented boxes over aerial imagery; xView: large-scale overhead boxes.
* **Why it matters:** Small, rotated objects; domain shift from natural images.
* **Quirks:** Rotated boxes (DOTA); extreme small object ratios; tiling needed.
* **Where:** Hugging Face (`dota`, `xview`).

### **Video Detection/Tracking (YouTube-BB, TAO, BDD-Tracking)**

* **What it is:** Frame-wise boxes over videos (YouTube-BB); long-tail, open-world (TAO); driving videos (BDD-Tracking).
* **Why it matters:** Temporal consistency and motion robustness.
* **Quirks:** Label sparsity per frame (YT-BB); identity switches (tracking); domain drift.
* **Where:** Hugging Face (`youtube_bounding_boxes`, `tao`, `bdd100k` with tracking splits).

---

## Preprocessing (what to do and why)

### Normalization

*We adjust pixel values so they’re centered and scaled, making training stable.*

* **ImageNet stats:** `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
  → Matches the preprocessing expected by most pretrained backbones (ResNet/ViT/ConvNeXt).
* **From scratch:** `Standardize per dataset`
  → If no pretrained weights, compute dataset mean/std to reduce covariate shift.

### Resizing & Aspect Strategy

*We resize while keeping aspect ratio to fit model input limits without distorting boxes.*

* **COCO/DETR style:** `Resize shortest_side∈{480…800}; max_size=1333`
  → Multi-scale training improves scale invariance; cap long side to control memory.
* **YOLO style:** `Letterbox to 640×640 (or 1024×1024)`
  → Pads to square without stretch; consistent batch shapes increase throughput.
* **Driving/Aerial:** `Tile or long-side=2048 + sliding windows`
  → Preserves tiny objects; tiling prevents shrinking small targets into oblivion.

### Geometric Augmentations

*We perturb geometry to improve invariance while updating boxes/masks accordingly.*

* **RandomHorizontalFlip(p=0.5)**
  → Cheap diversity; must flip x-coords of boxes.
* **RandomAffine / RandomPerspective (small ranges)**
  → Adds viewpoint variance; keep boxes clipped and valid.
* **Mosaic/MixUp (YOLO-family)**
  → Combines images to enrich context and small-object exposure; tune to avoid label noise.

### Photometric Augmentations

*We vary color/lighting so the model learns robust features instead of memorizing illumination.*

* **ColorJitter / HSV jitter / RandomBrightnessContrast**
  → Simulates different sensors/time-of-day; keep within moderate bounds.
* **Gaussian noise/blur (light)**
  → Models sensor noise or motion blur without hiding tiny objects.

### Label Encoding & Filtering

*We convert annotations to the target detector’s format and drop unusable boxes.*

* **Anchor-based (RetinaNet/YOLOv3):** encode to per-feature-map anchors
  → Precompute anchor boxes; match via IoU thresholds.
* **Anchor-free (FCOS/YOLOX/DETR):** center/point or set-based encoding
  → Simpler assignment; ensure correct class indices and \[x\_min, y\_min, x\_max, y\_max] order.
* **Filter tiny/invalid boxes (e.g., area < 4 px²)**
  → Reduces label noise; prevents unstable gradients.

### Evaluation Transforms

*We disable stochastic augments and use a single, reproducible resize for fair metrics.*

* **COCO/common:** `Resize shortest_side=800, max_size=1333` + `Normalize(...)`
  → Mirrors widely reported settings; no flips/crops at eval.
* **YOLO inference:** `Letterbox to train size` + `Normalize(...)`
  → Keeps NMS behavior consistent with training shape.

---

### Dataloading tips

*We prepare the dataset so training is fast, reproducible, and efficient.*

* **Aspect-ratio batching:** group images by similar h/w before batching
  → Cuts padding waste and speeds up training.
* **Custom `collate_fn` (variable targets):**
  → Batches images while keeping a list of per-image dictionaries (`boxes`, `labels`, `areas`, `iscrowd`).
* **Prefetch & pin memory:** `DataLoader(pin_memory=True, prefetch_factor>1, persistent_workers=True)`
  → Ensures GPUs aren’t starved while waiting for data.
* **Worker init functions:** `worker_init_fn=seed_all`
  → Makes sure random augmentations are reproducible across runs.
* **Cache decoded images / memmaps:**
  → Avoids repeated JPEG decode cost on large corpora.
* **COCO quirks:** respect `iscrowd` masks during training/eval
  → Use them to ignore regions in loss/evaluation; aligns with COCO mAP.

---

## Short Python pseudo-code (Hugging Face Datasets + Transformers/Torch)

## 1) Object detection (DETR-style, COCO)

```python
from datasets import load_dataset
from transformers import AutoImageProcessor
import torch, torchvision
from torchvision.transforms import v2 as T

# 1) Load COCO 2017
ds = load_dataset("coco", "2017", split="train")  # "validation" for val

# 2) Image processor for DETR (handles resizing/normalization/label encoding)
proc = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 3) Albumentations or torchvision v2 augments (geometric & photometric)
train_tf = T.Compose([
    T.ToImage(),
    T.RandomHorizontalFlip(p=0.5),
    # Multi-scale: handled by processor (size) or add RandomResize here if desired
])

def to_target(ann):
    # Convert HF COCO dict to DETR target format
    boxes = []
    classes = []
    for a in ann["annotations"]:
        if a["iscrowd"] == 1:  # optional: skip crowd in training
            continue
        x, y, w, h = a["bbox"]
        boxes.append([x, y, x+w, y+h])
        classes.append(a["category_id"])
    return {"boxes": torch.tensor(boxes, dtype=torch.float32),
            "class_labels": torch.tensor(classes, dtype=torch.int64)}

def preprocess(example):
    image = example["image"]
    target = to_target(example)
    image = train_tf(image)
    # Processor will resize to shorter_side=800, max_size=1333 by default for DETR
    out = proc(images=image, annotations={"image_id": example["image_id"],
                                          "annotations": example["annotations"]},
               return_tensors="pt")
    # Replace labels with our tensors (optional; proc may already build them)
    out["labels"] = [{
        "class_labels": target["class_labels"],
        "boxes": target["boxes"]
    }]
    return {"pixel_values": out["pixel_values"].squeeze(0),
            "labels": out["labels"][0]}

ds = ds.map(preprocess, remove_columns=ds.column_names)

def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = [b["labels"] for b in batch]  # list of dicts
    return {"pixel_values": pixel_values, "labels": labels}

from torch.utils.data import DataLoader
train_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=8,
                          pin_memory=True, collate_fn=collate_fn)

# 4) Model
from transformers import AutoModelForObjectDetection
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
for batch in train_loader:
    out = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
    loss = out.loss
    loss.backward()
    break
```

## 2) YOLO-style loading (letterbox + custom aug)

```python
# Pseudocode: implement letterbox resize, mosaic/mixup with Albumentations,
# return tensors: images [B,3,H,W], targets per image as (cls, x1,y1,x2,y2) after augment.
# Use a custom collate_fn to stack images and concatenate targets with image indices.
```

## 3) Domain-specific (Aerial with tiling)

```python
# Given very large images, tile into 1024x1024 windows with stride 768.
# Map/clip boxes belonging to each tile; drop boxes with <20% area kept to reduce noise.
# Keep metadata to merge predictions back to original coords at eval.
```

---

### Dataset-specific quirks (fast checklist)

* **COCO:** many small objects → use multi-scale, stronger aug; respect `iscrowd`; evaluate with standard 800/1333 resize.
* **LVIS:** extreme class imbalance → class-balanced sampling, focal loss/LA loss, frequency-aware reweighting.
* **Open Images:** hierarchical labels & noisy boxes → label cleaning, class mapping to COCO when needed.
* **Objects365:** class names differ → build consistent label map; good for pretraining then fine-tuning on COCO.
* **Driving (BDD/KITTI/Cityscapes):** lighting/weather domain shift → photometric aug and nighttime upweighting; anchor sizes tuned to large objects.
* **CrowdHuman:** dense overlaps → higher NMS IoU thresholds; consider soft-NMS/cluster-NMS.
* **Aerial (DOTA/xView):** tiny/rotated objects → tiling; consider rotated boxes or oriented detectors; higher input resolution.
* **Video (TAO/YT-BB):** temporal redundancy → sample frames with stride; optional motion-blur aug; tracking-aware eval.