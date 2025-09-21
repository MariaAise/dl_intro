---
title: "Introduction to Computer Vision"
css: styles.css
author: "Maria A"
description: "Introduction to CV tasks and architectures."
tags: ["deep learning", "computer vision", "research"]
---


# Audio for emotions

Speech-based emotion recognition involves analyzing vocal features to identify emotions expressed in speech. This task can be approached using various techniques, including:

1. **Feature Extraction**: Extracting relevant features from audio signals, such as Mel-frequency cepstral coefficients (MFCCs), pitch, and energy.

2. **Modeling**: Training machine learning models (e.g., SVM, Random Forest) or deep learning architectures (e.g., CNNs, RNNs) on labeled datasets to classify emotions.

3. **Evaluation**: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.

4. **Applications**: Implementing emotion recognition systems in areas like customer service, healthcare, and entertainment.

## Key Models and Architectures

1. **Convolutional Neural Networks (CNNs)**: Effective for capturing local patterns in spectrograms or raw audio signals.
2. **Recurrent Neural Networks (RNNs)**: Suitable for modeling sequential data and capturing temporal dependencies in audio signals.
3. **Transformers**: Emerging as powerful models for audio processing tasks, leveraging self-attention mechanisms to capture long-range dependencies.
4. **Hybrid Models**: Combining CNNs and RNNs to leverage both spatial and temporal features in audio data.

## Datasets
1. **RAVDESS**: A widely used dataset for emotion recognition in speech and song. Linked [here](https://zenodo.org/record/1188976) also at HF [here](https://huggingface.co/datasets/EmotionRAVDESS).
2. **CREMA-D**: Contains audio-visual recordings of actors expressing various emotions. Linked [here](https://github.com/CheyneyComputerScience/CREMA-D), also at HF [here](https://huggingface.co/datasets/crema_d).
3. **IEMOCAP**: A multimodal dataset with audio, video, and text annotations for emotion recognition. Linked [here](https://sail.usc.edu/iemocap/), also at HF [here](https://huggingface.co/datasets/iemocap).
4. **TESS**: Toronto emotional speech set, designed for emotion recognition tasks. Linked [here](https://tspace.library.utoronto.ca/handle/1807/24487), also at HF [here](https://huggingface.co/datasets/TESS).
5. **Emo-DB**: Berlin Database of Emotional Speech, containing recordings of actors expressing different emotions. Linked [here](https://tudelft.nl/en/ewi/research/ai/ai-lab/emo-db), also at HF [here](https://huggingface.co/datasets/emo_db).

## Challenges and Considerations
1. **Data Quality**: Ensuring high-quality audio recordings and accurate emotion labels.
2. **Variability**: Addressing variability in speech due to factors like speaker differences, accents, and recording conditions.
3. **Context**: Considering the context in which speech occurs, including situational and cultural factors that may influence emotional expression. 
4. **Real-time Processing**: Developing models that can operate in real-time for applications like virtual assistants and customer service bots.
5. **Ethical Considerations**: Addressing privacy concerns and potential biases in emotion recognition systems.

## Data and setup

```python
from datasets import load_dataset

ds = load_dataset("MahiA/RAVDESS", cache_dir="/Volumes/Crucial X9/data")
print(ds)
```

You will see the dataset has two splits: train and test. Each example contains an audio file and its corresponding emotion label.`classname`.

```python
print(ds["train"][0])
```

The `classname` feature contains the emotion label. To see all possible labels:

```python
print(ds["train"][0])
labels=set(ds["train"]["classname"])

print(f"The labels in the dataare: {labels}")
```  

>*Hint*: we use `set()` to get unique labels which is a data structure in Python that stores unordered unique items.

To see the distribution of labels in the training set:

```python
from collections import Counter
label_counts = Counter(ds["train"]["classname"])
for label, count in label_counts.items():
    print(f"{label}: {count}")
```

>*Hint*: `Counter` is a convenient way to count occurrences of items in a list. `Counter` returns a dictionary-like object where keys are the unique items and values are their counts.


To visualize the distribution of labels, you can use a bar chart:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
bars=plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel("Emotion labels")
plt.ylabel("Number of samples")
plt.title("Distribution of Emotion Labels in RAVDESS Training Set")
plt.xticks(rotation=45)

for bar in bars:
    height=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2, height+0.1, 
    f'{int(height)}', ha="center", va="bottom")
    
plt.tight_layout()
plt.show()
```

Now, let's listen to an example audio clip using `Gradio`:

```python
!pip install Gradio
```

If you noticed, our dataset does not contain the actual audio waveform data, but rather file paths to the audio files. We can use the `huggingface_hub` library to download and load these audio files. `hf_hub_download` function allows us to download files from a Hugging Face repository.

We also going to use the `soundfile` library to read audio files. If you don't have it installed, you can do so using pip:

```python
!pip install soundfile huggingface_hub
```
Now, we can define a function to load and return the audio data given a file path:

```python
from huggingface_hub import hf_hub_download
import soundfile as sf

def load_audio(file_path):
    audio_path=hf_hub_download(
        repo_id="MahiA/RAVDESS",
        filename=file_path,
        repo_type="dataset"
    )
    audio_data, samplerate=sf.read(audio_path)
    return audio_data, samplerate
```

Now, we can load the actual audio file 

```python
import gradio as gr

def generate_audio():
    example=ds["train"].shuffle(seed=42)[0]
    audio_data, samplerate=load_audio(example["path"])
    label=example["classname"]
    return label, (audio_data, samplerate)

```

and listen to it using Gradio:

```python
import gradio as gr



import gradio as gr



```python
import Gradio as gr

## Future Directions
1. **Multimodal Approaches**: Integrating audio with other modalities (e.g., facial expressions, physiological signals) for more robust emotion recognition.
2. **Transfer Learning**: Leveraging pretrained models on large audio datasets to improve performance on emotion recognition tasks.
3. **Explainability**: Developing methods to interpret model decisions and understand how emotions are inferred from audio features.
4. **Personalization**: Tailoring emotion recognition systems to individual users for improved accuracy and user experience.
5. **Cross-cultural Studies**: Investigating how cultural differences impact emotional expression and recognition in speech.    

## Takeaway
Speech-based emotion recognition is a complex yet promising field with significant applications across various domains. By leveraging advanced machine learning and deep learning techniques, researchers and practitioners can develop systems that accurately identify emotions from speech, enhancing human-computer interactions and providing valuable insights into human behavior.
