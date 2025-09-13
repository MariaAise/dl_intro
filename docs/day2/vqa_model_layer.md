---
title: "Visual Question Answering (VQA) - Model layer"
css: styles.css
author: "Maria A"
description: "Model architectures and methods for visual question answering."
tags: ["deep learning", "computer vision", "visual question answering", "research"]
---

# Visual Question Answering (VQA) - Model layer

## Model zoo (Hugging Face) — practical picks

### **BLIP (VQA head)**

* **Checkpoints:** `Salesforce/blip-vqa-base`, `Salesforce/blip-vqa-capfilt-large`
* **Why:** Strong open-ended answers; clean processor API; works well without heavy tricks.
* **Core idea:** Dual-encoder pretraining (ITC/ITM/LM) + VQA head; end-to-end on pixels (no region features).

---

### **BLIP-2 (ViT + Q-Former + LLM)**

* **Checkpoints:** `Salesforce/blip2-flan-t5-xl`, `Salesforce/blip2-opt-2.7b`, `Salesforce/blip2-flan-t5-xxl`
* **Why:** Bridges vision features to an LLM via **Q-Former** → strong reasoning, good few-shot.
* **Core idea:** Freeze ViT + LLM, learn a small **querying transformer** (Q-Former) for efficient alignment.

---

### **InstructBLIP (instruction-tuned BLIP-2)**

* **Checkpoints:** `Salesforce/instructblip-vicuna-7b`, `Salesforce/instructblip-flan-t5-xl`
* **Why:** Better follows prompts (“Answer concisely”, “use units …”); robust on diverse VQA styles.
* **Core idea:** Instruction tuning on mixed VQA/vision-lang corpora to improve controllability.

---

### **OFA (Unified Sequence-to-Sequence)**

* **Checkpoints:** `OFA-Sys/ofa-base`, `OFA-Sys/ofa-large`
* **Why:** One seq2seq framework for many vision-language tasks (captioning, VQA, grounding).
* **Core idea:** Everything is text generation conditioned on visual tokens; multitask pretraining.

---

### **ViLT (Vision-and-Language Transformer)**

* **Checkpoints:** `dandelin/vilt-b32-finetuned-vqa`
* **Why:** **No region detector** — patches + text tokens in a single transformer; lightweight & fast.
* **Core idea:** Early fusion of image patches + subword tokens; end-to-end pretraining objectives (MLM/ITM).

---

### **LLaVA / MiniGPT-4 (community MLLMs)**

* **Checkpoints:** `liuhaotian/llava-v1.5-7b`, `liuhaotian/llava-v1.6-vicuna-7b`, `OpenGVLab/minigpt-4-v1_7b`
* **Why:** Chat-style VQA (multi-turn, chain-of-thoughty answers), strong zero-/few-shot on open images.
* **Core idea:** **CLIP/ViT visual encoder → projection → LLM** with visual-instruction tuning.

---

### **OCR-aware VQA (TextVQA / DocVQA)**

* **Checkpoints:** `naver-clova-ix/donut-base-finetuned-docvqa`, `microsoft/layoutlmv3-base` (+ heads)
* **Why:** When reading text in images is essential (menus, receipts, signs).
* **Core idea:** End-to-end OCR-free (Donut) or layout-aware encoders (LayoutLMv3) + QA decoding.

---

### (Legacy but notable) **LXMERT / UNITER / ViLBERT**

* **Checkpoints:** `unc-nlp/lxmert-base-uncased` (others often require conversion)
* **Why:** Classic **region-feature** (Faster R-CNN) + text fusion baselines; still useful for ablations.
* **Core idea:** Late-fusion with pre-extracted object regions (“bottom-up attention”).

---

## Architectural innovations to know

* **Region features → End-to-end pixels:** Older VQA used Faster R-CNN region proposals; newer (ViLT/BLIP) learn directly from patches.
* **Q-Former bridging (BLIP-2):** A small trainable transformer queries frozen vision features and speaks to a frozen LLM → efficiency + strong reasoning.
* **Instruction tuning for vision-language:** InstructBLIP/LLaVA align outputs to natural prompts and constraints.
* **Multitask seq2seq (OFA):** Unifies many tasks as text generation with visual conditioning.
* **OCR-aware pathways:** Either OCR-free (Donut) or OCR+layout (LayoutLMv3) for TextVQA/DocVQA.
* **Contrastive & matching pretraining:** ITC/ITM/MLM to align modalities (BLIP family, ViLT).

---

## Short code snippets (HF Transformers)

### 1) **BLIP (VQA) — direct answers**

```python
import torch
from transformers import BlipForQuestionAnswering, BlipProcessor
from PIL import Image

model_id = "Salesforce/blip-vqa-base"
processor = BlipProcessor.from_pretrained(model_id)
model = BlipForQuestionAnswering.from_pretrained(model_id)

img = Image.open("demo.jpg").convert("RGB")
question = "What color is the car?"

inputs = processor(images=img, text=question, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10)
answer = processor.decode(out[0], skip_special_tokens=True)
print(answer)
```

### 2) **BLIP-2 (Q-Former + LLM) — stronger reasoning**

```python
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

model_id = "Salesforce/blip2-flan-t5-xl"  # or blip2-opt-2.7b
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).eval()

img = Image.open("chart.png").convert("RGB")
prompt = "Answer concisely: How many bars are blue?"

inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=20)
print(processor.decode(out[0], skip_special_tokens=True))
```

### 3) **InstructBLIP — instruction-following VQA**

```python
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image

model_id = "Salesforce/instructblip-vicuna-7b"
processor = InstructBlipProcessor.from_pretrained(model_id)
model = InstructBlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).eval()

img = Image.open("street.jpg").convert("RGB")
prompt = "You are a VQA assistant. Q: How many pedestrians are crossing? A:"

inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=15)
print(processor.decode(out[0], skip_special_tokens=True))
```

### 4) **ViLT — fast, detector-free baseline**

```python
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

model_id = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(model_id)
model = ViltForQuestionAnswering.from_pretrained(model_id)

img = Image.open("kitchen.jpg").convert("RGB")
question = "Is the stove on?"

enc = processor(img, question, return_tensors="pt")
with torch.no_grad():
    logits = model(**enc).logits  # classification over answer vocab
answer_idx = logits.argmax(-1).item()
print(model.config.id2label[answer_idx])
```

### 5) **OFA — seq2seq VQA**

```python
import torch
from transformers import OFATokenizer, OFAProcessor, OFAForConditionalGeneration
from PIL import Image

model_id = "OFA-Sys/ofa-base"
tokenizer = OFATokenizer.from_pretrained(model_id)
processor = OFAProcessor.from_pretrained(model_id)
model = OFAForConditionalGeneration.from_pretrained(model_id)

img = Image.open("menu.jpg").convert("RGB")
question = "What is the price of the soup?"

enc = processor(text=question, images=img, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=16)
print(tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip())
```

### 6) **OCR-aware (DocVQA / TextVQA) — Donut**

```python
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image

model_id = "naver-clova-ix/donut-base-finetuned-docvqa"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

img = Image.open("receipt.jpg").convert("RGB")
prompt = "<s_docvqa><question>What is the total?</question><image>"

inputs = processor(img, prompt, return_tensors="pt")
with torch.no_grad():
    out_ids = model.generate(**inputs, max_new_tokens=32)
print(processor.batch_decode(out_ids)[0].replace(processor.tokenizer.eos_token, "").strip())
```

### 7) **LLaVA (chat-style VQA) — minimal**

```python
# Note: requires appropriate GPU + bitsandbytes/accelerate; API varies by release.
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "liuhaotian/llava-v1.5-7b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")

img = Image.open("scene.jpg")
prompt = "USER: What time is shown on the clock? ASSISTANT:"
inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=32)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

---

## When to pick what (quick guide)

* **General VQA, strong accuracy:** *BLIP-2 / InstructBLIP*
* **Fast, lightweight baseline / ablations:** *ViLT*
* **Instruction-following / multi-turn:** *InstructBLIP / LLaVA*
* **Text-heavy images (menus, receipts, signs):** *Donut / LayoutLMv3 pipelines*
* **Unified multitask research setups:** *OFA*
* **Legacy comparisons on region features:** *LXMERT/UNITER (research baselines)*

