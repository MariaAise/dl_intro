---
title: "Audio Classification - Model layer"    
css: styles.css
author: "Maria A"
description: "How to choose and use models for audio classification tasks."
tags: ["deep learning", "audio", "research"]
---
# Audio Classification - Model layer

## Hugging Face model zoo options (good starting checkpoints)

### Transformers on spectrograms (patch-based ViTs)

* **AST (Audio Spectrogram Transformer)** — `MIT/ast-finetuned-audioset-10-10-0.4593`
  *Multi-label Audioset head; strong general audio tagging baseline.*
* **PaSST (Patchout Spectrogram Transformer)** — e.g., `kkoutini/passt_s_kd_ast10_10`
  *Patchout regularization → efficient + robust on long clips.*
* **HTS-AT (Token-Semantic Transformer for Audio Tagging)** — `TencentGameMate/chinese-htsat` (family)
  *Strong music/environmental tagging; pairs well with CLAP text heads.*
* **Audio-MAE (Masked Autoencoding pretrain on spectrograms)** — e.g., `MIT/ssast-base-patch400` (SSAST/AudioMAE family)
  *Self-supervised pretraining → label-efficient fine-tuning.*

### Self-supervised waveform encoders (SSL)

* **Wav2Vec2 / HuBERT / WavLM** — `facebook/wav2vec2-base`, `superb/hubert-large-superb-er`, `microsoft/wavlm-base-plus`
  *Encode raw wave; add small classifier head for KWS/ASC/emotion.*
* **BEATs** — `microsoft/BEATs` / `microsoft/BEATs-iter3`
  *SSL optimized for non-speech acoustic events; strong ESC-50/ASC results.*

### CNN baselines (fast, small, deployable)

* **PANNs (CNN14/ResNet22)** — `qiuqiangkong/panns_cnn14`
  *Mature CNN family; good speed/quality trade-off.*
* **YAMNet (MobileNet-V1)** — ports on HF (search “yamnet”)
  *Tiny, mobile-friendly; handy for real-time tagging.*

### Audio–text contrastive (zero-shot & tagging)

* **CLAP (Contrastive Language–Audio Pretraining)** — `laion/clap-htsat-unfused` / `laion/clap-htsat-fused`
  *Zero-shot label search via text prompts; also fine-tunes to tags.*
* **AudioCLIP** — community ports (search “audioclip”)
  *Image/audio/text shared space; useful for multimodal label spaces.*

### Task-specific heads

* **Keyword spotting (Speech Commands)** — `superb/wav2vec2-base-superb-ks`
* **Acoustic scenes** — `dcase2020/task1a-baseline-*` (various community ports)
* **Music tagging** — `M-A-P/MERT-v1-95M` (music spectral transformer), `mtg-jamendo-*` adapters

> Pick by use-case: **Audioset-style multi-label** → AST/PaSST/HTS-AT/BEATs; **KWS** → Wav2Vec2/WavLM small; **On-device** → YAMNet/PANNs; **Zero-shot** → CLAP.

---

## Architectural innovations (what actually moves the needle)

* **Spectrogram ViTs (AST/PaSST/HTS-AT):**
  Convert log-mels to **patch tokens**, use **self-attention** to model long contexts; **patchout** (PaSST) randomly drops patches for regularization & speed. Attention **pooling** over time/freq replaces fixed global pooling for better temporal localization.

* **Self-supervised waveform encoders (Wav2Vec2/HuBERT/WavLM/BEATs):**
  Pretrain on raw audio with contrastive/masked objectives → **strong universal features**; downstream = a small **classification head**. **BEATs** tailors SSL to **non-speech acoustic** semantics.

* **Weak-label learning & MIL pooling (AudioSet/FSD50K):**
  Use **segment-level tokens** with **MIL/attention pooling** to aggregate clip-level predictions; handles noisy/weak labels and long clips.

* **Audio–text contrastive (CLAP/AudioCLIP):**
  Align audio and natural-language labels → **zero-shot** classification; lets you “prompt” new classes without retraining.

* **Learnable front-ends (LEAF, SincNet):**
  Replace fixed mel filters with **learnable filterbanks**, improving robustness across domains/devices.

* **Multi-scale & efficient designs:**
  **Dilated CNNs**, **temporal pooling pyramids**, and **patch subsampling** (PaSST) keep **latency + memory** low for deployment.

---

## Short Hugging Face code snippets

### 1) One-liner inference with a pretrained multi-label tagger (AST)

```python
from transformers import pipeline
clf = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
# Returns top labels with scores; for multi-label keep more results:
preds = clf("example.wav", top_k=20)  # filter by score threshold yourself
```

### 2) AutoModel for fine-tuning a spectrogram transformer (single/multi-label)

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForAudioClassification, TrainingArguments, Trainer
import numpy as np

id2label = {i: c for i, c in enumerate(class_names)}
label2id = {c: i for i, c in id2label.items()}

ds = load_dataset("ashraq/esc50").cast_column("audio", Audio(16000))
proc = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def preprocess(batch):
    audio = batch["audio"]["array"]
    inputs = proc(audio, sampling_rate=16000, return_tensors="pt")
    batch["input_values"] = inputs["input_values"][0]
    batch["labels"] = label2id[batch["category"]]
    return batch

ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names)
model = AutoModelForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(class_names), label2id=label2id, id2label=id2label,
    problem_type="single_label_classification"  # or "multi_label_classification"
)

args = TrainingArguments(output_dir="out", fp16=True, per_device_train_batch_size=8, num_train_epochs=10)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
```

### 3) Multi-label head (e.g., FSD50K) — logits → sigmoid

```python
import torch
logits = model(**batch).logits              # [B, C]
probs  = torch.sigmoid(logits)              # multi-label probabilities
preds  = (probs > 0.5).int()
```

### 4) Fine-tune a waveform SSL encoder (WavLM) for keyword spotting

```python
from transformers import AutoProcessor, AutoModelForAudioClassification

model_id = "microsoft/wavlm-base-plus"  # strong SSL backbone
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(
    model_id, num_labels=len(kw_classes), label2id=label2id, id2label=id2label
)

# processor will handle resampling/feature extraction from raw wave arrays
inputs = processor(audio=wave_array, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

### 5) Zero-shot audio classification with CLAP (text prompts as labels)

```python
from transformers import AutoProcessor, AutoModel
import torch, torchaudio

model_id = "laion/clap-htsat-unfused"
proc = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).eval()

audio, sr = torchaudio.load("example.wav")
audio = torchaudio.functional.resample(audio, sr, 48000).mean(0).unsqueeze(0)  # mono 48k

labels = ["dog bark", "siren", "vacuum cleaner", "rain"]
with torch.no_grad():
    a = model.get_audio_features(**proc(audio=audio, sampling_rate=48000, return_tensors="pt"))
    t = model.get_text_features(**proc(text=labels, return_tensors="pt"))
    sims = (a / a.norm(dim=-1, keepdim=True)) @ (t / t.norm(dim=-1, keepdim=True)).t()
    top = sims.squeeze(0).softmax(-1)
print({lab: float(score) for lab, score in zip(labels, top)})
```

### 6) Lightweight CNN baseline (PANNs CNN14) feature extractor + linear head

```python
import torch.nn as nn
from transformers import AutoModel

backbone = AutoModel.from_pretrained("qiuqiangkong/panns_cnn14")  # returns pooled embeddings
class Head(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x): return self.fc(x)

# combine backbone + head in your training loop
```

---

### Practical pairing guide

* **Small data (ESC-50, US8K):** PaSST/AST with **heavy augmentation** or **BEATs/WavLM** with frozen backbone + linear probe.
* **Large, weak labels (AudioSet/FSD50K):** HTS-AT/PaSST + **MIL/attention pooling**, class-balanced sampling or focal loss.
* **On-device/real-time:** YAMNet/PANNs; consider quantization/distillation.
* **Zero-shot / open-vocab:** CLAP (label text prompts), optionally few-shot tune with prototypes.

