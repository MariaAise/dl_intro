---
title: "Image Captioning - Evaluation layer"
css: styles.css
author: "Maria A"
description: "Evaluation metrics and methods for image captioning."
tags: ["deep learning", "computer vision", "image captioning", "evaluation"]
---
# Image Captioning - Evaluation layer

### Core metrics (what they measure & when to use)

* **BLEU-1/2/3/4 (n-gram precision)**

  * *Use when:* You want quick, legacy comparability (older papers, small ablations).
  * *Insight:* Precision-oriented; penalizes missing common n-grams; weak on synonyms/paraphrases.

* **METEOR (unigram F, stem/synonym matching)**

  * *Use when:* You care about recall and soft-matching (morphology, WordNet synonyms).
  * *Insight:* More tolerant to paraphrase than BLEU; historically correlates better with humans than BLEU.

* **ROUGE-L (longest common subsequence)**

  * *Use when:* You want order-aware recall; often reported alongside BLEU/METEOR.
  * *Insight:* Captures sequence overlap without strict n-gram windows.

* **CIDEr / CIDEr-D (tf-idf weighted n-grams, consensus-based)**

  * *Use when:* **Primary COCO leaderboard metric**; multiple references available.
  * *Insight:* Rewards phrases common among human captions; down-weights generic words.

* **SPICE (scene-graph F1)**

  * *Use when:* You want **semantic quality** (objects, attributes, relations).
  * *Insight:* High correlation with human judgment on semantics; slower to compute.

* **SPIDEr (SPICE + CIDEr average)**

  * *Use when:* Balanced single score mixing **semantic** (SPICE) and **consensus fluency** (CIDEr).
  * *Insight:* Good overall selection metric for model picking.

* **BERTScore (semantic similarity via contextual embeddings)**

  * *Use when:* Paraphrases are common; domain captions vary lexically.
  * *Insight:* Token-level cosine similarity; robust to wording differences.

* **CLIPScore / RefCLIPScore (reference-free / reference-aware)**

  * *Use when:* You need **reference-light** evaluation or to detect image–caption alignment.
  * *Insight:* Measures vision–text alignment in a joint embedding space; complements text-only metrics.

* **Diversity/Novelty (Distinct-n, Self-BLEU)**

  * *Use when:* Checking mode collapse or repetitive outputs across a dataset.
  * *Insight:* Higher distinct-n → more lexical diversity; low Self-BLEU across corpus → diverse set.

> **Rule of thumb**
>
> * Report **CIDEr, SPICE (or SPIDEr)** as primary; include **BLEU-4**, **METEOR**, **ROUGE-L** for comparability; add **BERTScore/CLIPScore** for semantic/alignment checks. Always evaluate against **all references** per image.

---

### Visualization methods (to understand *why* a caption was produced)

* **Grad-CAM / Grad-CAM++ on the vision encoder**

  * *What:* Localizes image regions influencing the encoder output.
  * *Why:* Verify that nouns/attributes mentioned are grounded in visual evidence.

* **Attention rollouts / cross-attention maps (encoder–decoder or Q-Former)**

  * *What:* Aggregate attention to visualize **token ↔ region** focus.
  * *Why:* Inspect which image patches support each generated word.

* **Token-level relevance overlays**

  * *What:* Show heatmap per generated token (e.g., “dog”, “red ball”).
  * *Why:* Helpful for error analysis (hallucinations vs grounded mentions).

* **Corpus-level dashboards**

  * *What:* Length histograms, novelty (distinct-n), per-category CIDEr (COCO categories), and hard-set breakdowns (e.g., NoCaps in/near/out-of-domain).
  * *Why:* Exposes brittleness and domain shift.

---

## Python snippets — metrics & visualization

> These are practical, minimal recipes. 

> `pip install evaluate torchmetrics pytorch-grad-cam pycocotools`

> For CIDEr/SPICE via COCO: `pycocoevalcap` implementations are bundled with some repos; a maintained route is to use **`evaluate`** (BLEU/METEOR/ROUGE/BERTScore) + **TorchMetrics CLIPScore** and a standalone SPICE/CIDEr script, or the official COCO caption eval toolkit if available in your environment.

### 1) Evaluate with Hugging Face `evaluate` (BLEU/METEOR/ROUGE/BERTScore)

```python
import evaluate

# references: list[list[str]]  (multiple refs per image)
# predictions: list[str]       (one caption per image, aligned by index)

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

bleu_out = bleu.compute(predictions=predictions, references=[[r] for r in references])  # BLEU expects list of list
meteor_out = meteor.compute(predictions=predictions, references=references)
rouge_out = rouge.compute(predictions=predictions, references=[" ".join(rs) for rs in references])
bert_out = bertscore.compute(predictions=predictions, references=[rs[0] for rs in references],
                             lang="en")

print({
  "BLEU": bleu_out["bleu"],
  "METEOR": meteor_out["meteor"],
  "ROUGE-L": rouge_out["rougeL"],
  "BERTScore_F1": sum(bert_out["f1"])/len(bert_out["f1"]),
})
```

### 2) CLIPScore (image–text alignment) with TorchMetrics

```python
import torch
from PIL import Image
from torchvision import transforms as T
from torchmetrics.multimodal.clip_score import CLIPScore

preprocess = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                                    [0.26862954, 0.26130258, 0.27577711])])  # CLIP stats

clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").cuda()

def images_to_tensor(img_paths):
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in img_paths]).cuda()
    return batch

imgs = images_to_tensor(img_paths)          # list of file paths in same order as predictions
texts = predictions                          # list[str] captions
score = clip_score(imgs, texts)              # average CLIPScore over the batch
print("CLIPScore:", score.item())
```

### 3) COCO CIDEr/SPICE (using the COCO Caption toolkit — sketch)

```python
# Assumes you have the COCO caption evaluation code available in PYTHONPATH.
# Data formats:
#  - gts: {"annotations": [{"image_id": id, "caption": str}, ...]}
#  - res: [{"image_id": id, "caption": str}, ...]

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

coco = COCO("captions_val2017.json")  # ground truth annotations (Karpathy/COCO format)
coco_res = coco.loadRes("results.json")  # your model outputs
coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.evaluate()

print({
  "CIDEr": coco_eval.eval["CIDEr"],
  "SPICE": coco_eval.eval.get("SPICE", None),
  "BLEU-4": coco_eval.eval["BLEU_4"],
  "METEOR": coco_eval.eval["METEOR"],
  "ROUGE-L": coco_eval.eval["ROUGE_L"],
})
```

### 4) Grad-CAM over a ViT/ResNet encoder (using `pytorch-grad-cam`)

```python
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Example: visualize CLIP's image encoder attention to a generated caption keyword
from transformers import CLIPModel, CLIPProcessor
model_id = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(model_id).eval().cuda()
processor = CLIPProcessor.from_pretrained(model_id)

img = Image.open("demo.jpg").convert("RGB")
rgb = np.asarray(img).astype(np.float32) / 255.0

inputs = processor(text=[pred_caption], images=img, return_tensors="pt", padding=True).to("cuda")
with torch.no_grad():
    out = clip_model(**inputs)
    score = out.logits_per_image[0,0]  # image-text similarity

# Target the last visual block for Grad-CAM
target_layers = [clip_model.vision_model.encoder.layers[-1].layer_norm2]  # layer choice may vary per model
cam = GradCAM(model=clip_model.vision_model, target_layers=target_layers, use_cuda=True)

# Define a target that encourages higher image-text similarity
class SimilarityTarget:
    def __call__(self, model_out):
        # create a pseudo target: increase pooled image feature dot pooled text feature
        return model_out.pooler_output[:,0].sum()

grayscale_cam = cam(input_tensor=inputs.pixel_values, targets=[SimilarityTarget()])[0]
vis = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
plt.imshow(vis); plt.axis("off"); plt.title("Grad-CAM over image encoder for caption alignment")
plt.show()
```

### 5) Cross-attention map visualization (encoder–decoder)

```python
# Works with models exposing cross-attention (e.g., VisionEncoderDecoder, BLIP).
# We hook the decoder's cross-attention weights and visualize for a chosen token.

import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from PIL import Image

mid = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(mid, output_attentions=True).eval().to("cuda")
tok = AutoTokenizer.from_pretrained(mid)
proc = AutoImageProcessor.from_pretrained(mid)

img = Image.open("demo.jpg").convert("RGB")
pix = proc(img, return_tensors="pt").pixel_values.to("cuda")

gen = model.generate(pix, max_new_tokens=32, output_attentions=True, return_dict_in_generate=True)
caption_ids = gen.sequences[0]
caption = tok.decode(caption_ids, skip_special_tokens=True)

# Cross-attentions: list[dec_layers] of (num_heads, tgt_len, src_len)
cross_attns = [a.cpu().numpy() for a in gen.decoder_attentions]  # may be tuple per layer

# Pick a token (e.g., the word most related to an object)
token_idx = np.argmax((caption_ids != tok.pad_token_id).cpu().numpy())  # or search for specific token id
attn_layers = []
for layer in cross_attns:
    # average heads
    attn_layers.append(layer.mean(axis=0)[token_idx])  # (src_len,)
attn_map = np.mean(np.stack(attn_layers, 0), 0)

# Reshape src_len (ViT patches) to grid (e.g., 14x14 for 224/16)
grid = int(np.sqrt(attn_map.shape[-1]))
attn_grid = attn_map[:grid*grid].reshape(grid, grid)

plt.imshow(attn_grid, cmap="jet")
plt.colorbar(); plt.title(f"Cross-attention to patches for token: `{caption.split()[token_idx]}`")
plt.show()
```

### 6) Diversity & length diagnostics

```python
import numpy as np

def distinct_n(captions, n=2):
    ngrams = set()
    total = 0
    for c in captions:
        toks = c.lower().split()
        total += max(0, len(toks)-n+1)
        for i in range(len(toks)-n+1):
            ngrams.add(tuple(toks[i:i+n]))
    return len(ngrams) / (total + 1e-8)

lens = [len(c.split()) for c in predictions]
print({
  "len_mean": float(np.mean(lens)),
  "len_std": float(np.std(lens)),
  "distinct-1": distinct_n(predictions, 1),
  "distinct-2": distinct_n(predictions, 2)
})
```

---

### Practical guidance

* **Always** evaluate with **all references** (COCO has 5 per image). Some libraries need a list of refs per example.
* Use **SPIDEr (CIDEr+SPICE avg)** for single best model selection; it balances consensus fluency with semantic correctness.
* Add **CLIPScore** to catch **hallucinations** (high text-only metrics but poor image alignment).
* Visualize **per-token cross-attention** when captions go wrong: if heat isn’t on the mentioned region, it’s likely hallucinated.
* Keep **val transforms deterministic** and seed all randomness so metric changes reflect model changes, not preprocessing noise.
