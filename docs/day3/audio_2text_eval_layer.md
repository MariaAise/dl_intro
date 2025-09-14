---
title: "Audio-to-Text Conversion - Evaluation layer"    
css: styles.css
author: "Maria A"
description: "How to evaluate ASR model outputs and interpret results."
tags: ["deep learning", "audio", "research"]
---
# Audio-to-Text Conversion - Evaluation layer

### Core metrics (what/when/why)

* **WER (Word Error Rate)** = (S + D + I) / N
  *Use for:* most Latin-script languages and word-segmented scripts.
  *Insight:* overall transcript correctness; sensitive to word insertions/deletions.
* **CER (Character Error Rate)**
  *Use for:* languages without whitespace (zh, ja), noisy text normalization, or when tokenization is tricky.
  *Insight:* fine-grained errors; correlates with readability for non-segmented scripts.
* **Segment WER vs Concatenated WER**
  *Use for:* long-form audio evaluated by chunks.
  *Insight:* concatenated WER reveals stitch/overlap issues hidden by per-chunk scoring.
* **Entity/Number Accuracy (custom slots)**
  *Use for:* domains heavy in numerals, IDs, names (finance/medical).
  *Insight:* business-critical correctness beyond overall WER.
* **RTF (Real-Time Factor) & Latency (p50/p90)**
  *Use for:* streaming/production.
  *Insight:* deployment feasibility; RTF < 1 means faster-than-real-time offline decoding.
* **DER (Diarization Error Rate) & JER**
  *Use for:* multi-speaker ASR with speaker labels.
  *Insight:* speaker attribution quality (miss/false alarm/confusion).
* **LID Accuracy / Code-switch WER**
  *Use for:* multilingual pipelines.
  *Insight:* language routing quality; per-language WER comparisons.
* **Calibration (ECE/Brier/NLL on confidence)**
  *Use for:* post-ASR confidence scoring.
  *Insight:* how well scores reflect true correctness (useful for human-in-the-loop).

> Practical rule: **WER/CER** for core model progress, **DER** when speakers matter, **entity/number accuracy** for domain usefulness, **RTF/latency** for deployability.

---

### Visualization methods (to diagnose + explain)

* **Alignment heatmaps**

  * *Seq2seq attention maps:* decoder-to-encoder attention over time (token ↔ frame).
  * *CTC alignments:* frame-level best path / forced alignment overlay on spectrogram.
* **Word-timeline plots**

  * Show predicted words with start/end times over a waveform or log-Mel spectrogram.
* **Error overlays**

  * Color Levenshtein operations (S/D/I) along the timeline; spotlight where/why WER arises.
* **Entity highlighting**

  * Highlight numbers/dates/tickers in reference vs hypothesis (correct/incorrect).
* **Diarization ribbons**

  * Horizontal bars per speaker with ASR text above; quickly reveals overlap/confusions.
* **Saliency on spectrogram (Integrated Gradients)**

  * Attribute which time–freq regions influenced a token; great for noise/debugging.

---

## Python snippets

> Minimal, drop-in examples. Replace file paths/datasets as needed.

### 1) WER/CER with 🤗 `evaluate` + detailed jiwer report

```python
import evaluate, jiwer

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

ref = ["we will meet at nine thirty"]
hyp = ["we will meat at 9 30"]

print("WER:", wer_metric.compute(references=ref, predictions=hyp))
print("CER:", cer_metric.compute(references=ref, predictions=hyp))

# Detailed analysis: substitutions/deletions/insertions
transforms = jiwer.Compose([
    jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.Strip(),
    jiwer.ReduceToListOfWords()
])
report = jiwer.compute_measures(ref, hyp, truth_transform=transforms, hypothesis_transform=transforms)
print(report)  # keys: wer, mer, wil, hits, subs, dels, ins
```

### 2) Entity/number accuracy (custom slots)

```python
import re
def extract_numbers(s): return re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", s.replace(",", ""))
def entity_accuracy(ref, hyp):
    r, h = set(extract_numbers(ref)), set(extract_numbers(hyp))
    return {"num_acc": len(r & h) / max(1, len(r)), "missed": list(r - h), "hallucinated": list(h - r)}

print(entity_accuracy(ref[0], hyp[0]))
```

### 3) Real-Time Factor (RTF) & latency

```python
import soundfile as sf, time
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
audio, sr = sf.read("long_meeting.wav")
t0 = time.time(); _ = asr("long_meeting.wav"); t1 = time.time()

duration_s = len(audio) / sr
rtf = (t1 - t0) / duration_s
print({"audio_sec": duration_s, "wall_sec": t1 - t0, "RTF": rtf})
```

### 4) Word-timeline plot over spectrogram

```python
import matplotlib.pyplot as plt, numpy as np, librosa, librosa.display
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
               return_timestamps=True, chunk_length_s=30)
out = asr("clip.wav")  # {"text":..., "chunks":[{"text":"...","timestamp":[s,e]}, ...]}

y, sr = librosa.load("clip.wav", sr=16000)
S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80), ref=np.max)
plt.figure(figsize=(12,4))
librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="mel")
for c in out.get("chunks", []):
    s, e = c["timestamp"]
    if s is None or e is None: continue
    plt.axvspan(s, e, alpha=0.15)
    plt.text((s+e)/2, S.shape[0]*0.9, c["text"], ha="center", va="top", fontsize=8, rotation=0)
plt.title("Word timeline over log-Mel"); plt.tight_layout(); plt.show()
```

### 5) Levenshtein error overlay (S/D/I) per token

```python
from rapidfuzz.distance import Levenshtein
from rapidfuzz import process

def colored_ops(ref_tokens, hyp_tokens):
    # Simple alignment path (ops): 0=equal, 1=sub, 2=del, 3=ins
    ops = Levenshtein.editops(ref_tokens, hyp_tokens)
    marks = ["="]*len(hyp_tokens)
    for op in ops:
        if op.tag == "replace": marks[op.dst_pos] = "S"
        elif op.tag == "insert": marks[op.dst_pos] = "I"
    dels = [ref_tokens[op.src_pos] for op in ops if op.tag=="delete"]
    return marks, dels

ref_tokens = ref[0].split(); hyp_tokens = hyp[0].split()
marks, deletions = colored_ops(ref_tokens, hyp_tokens)
print(list(zip(hyp_tokens, marks)), "DELETIONS:", deletions)
```

### 6) Diarization Error Rate (DER) with `pyannote.metrics`

```python
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# reference and hypothesis as pyannote Annotations
ref = Annotation()
ref[Segment(0, 5)] = "spk1"; ref[Segment(5, 10)] = "spk2"

hyp = Annotation()
hyp[Segment(0, 4.5)] = "A"; hyp[Segment(4.5, 10)] = "B"

der = DiarizationErrorRate()
print("DER:", der(ref, hyp))
```

### 7) Attention heatmap (seq2seq models that expose attentions)

```python
import torch, matplotlib.pyplot as plt
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", output_attentions=True).eval()
proc  = WhisperProcessor.from_pretrained("openai/whisper-small")

import torchaudio
wav, sr = torchaudio.load("clip.wav"); wav = torchaudio.functional.resample(wav, sr, 16000)
inputs = proc.feature_extractor(wav.mean(0), sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    out = model.generate(inputs.input_features, output_attentions=True, return_dict_in_generate=True)
# cross-attentions for the last layer (list[tgt_len x src_len])
attn = torch.stack(out.cross_attentions[-1]).squeeze(0).mean(0).cpu().numpy()
plt.imshow(attn, aspect="auto", origin="lower"); plt.xlabel("encoder time"); plt.ylabel("decoder steps")
plt.title("Decoder→Encoder attention"); plt.colorbar(); plt.show()
```

### 8) Saliency on spectrogram (Integrated Gradients with Captum)

```python
import torch, numpy as np, matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").eval()
proc  = WhisperProcessor.from_pretrained("openai/whisper-small")

def forward_melspec(input_features):
    # score of the first generated token (logit for e.g. BOS→first text token)
    out = model.generate(input_features, max_new_tokens=1, output_scores=True, return_dict_in_generate=True)
    return out.scores[0][:, :].max(dim=-1).values  # scalar-like per batch

ig = IntegratedGradients(forward_melspec)

# Prepare features
import torchaudio
wav, sr = torchaudio.load("noisy.wav"); wav = torchaudio.functional.resample(wav, sr, 16000)
features = proc.feature_extractor(wav.mean(0), sampling_rate=16000, return_tensors="pt").input_features

attr, _ = ig.attribute(features, baselines=torch.zeros_like(features), return_convergence_delta=True)
sal = attr.squeeze().abs().sum(0).numpy()  # time x mel
plt.imshow(sal, aspect="auto", origin="lower"); plt.title("Saliency (IG) on log-Mel"); plt.colorbar(); plt.show()
```

### 9) Concatenated WER for chunked decoding

```python
from itertools import chain
import evaluate
wer = evaluate.load("wer")

def concat_text(chunks):
    return " ".join([c["text"].strip() for c in chunks])

ref_full = open("reference.txt").read().strip()
hyp_chunks = asr("long.wav")["chunks"]  # from pipeline with return_timestamps=True
hyp_full = concat_text(hyp_chunks)

print("Chunk-avg WER:", sum(wer.compute(references=[r], predictions=[h["text"]]) for r,h in zip(ref_full.splitlines(), hyp_chunks)) / len(hyp_chunks))
print("Concatenated WER:", wer.compute(references=[ref_full], predictions=[hyp_full]))
```

---

### Interpreting results (quick guide)

* **WER↓ improves** but **number accuracy flat** → add LM biasing or custom normalization.
* **Good CER but poor WER** → tokenization/word-segmentation issue; revisit text normalization.
* **DER high, WER good** → diarization, not acoustics; improve VAD/overlap handling.
* **RTF < 1 offline but p90 latency high online** → I/O or chunk/stride settings; reduce chunk length, enable batched streaming.
* **Attention/CTC alignment off** around music/noise → add RIR/noise augments; tighten VAD; consider speech enhancement.

