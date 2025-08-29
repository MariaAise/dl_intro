---

---

# 🚀 Day 1 – LLM Platforms & API Setup

This session introduces the key **LLM platforms** used throughout the course and walks you through how to **interact with them via GUI and API**.

By the end of this session, you'll be able to:
- Understand the landscape of LLM providers (OpenAI, Google, Hugging Face, DeepSeek, Anthropic)
- Navigate playground environments like Google AI Studio, OpenAI Playground, Hugging Face Spaces
- Generate API keys and configure Python access
- Run code examples across Gemini, GPT-4, and HF models
- Compare outputs, strengths, and deployment considerations

---

## 🗺️ Platform Landscape

Start here for an overview of the ecosystem and conceptual differences between models:

📘 [`platform_comparison.md`](./platform_comparison.md)

Includes:
- Feature comparison table
- Model access workflows
- Conceptual divergence between providers

---

## 🧪 Walkthroughs

| Platform      | UI Walkthrough Page                             |
|---------------|--------------------------------------------------|
| Hugging Face  | [`huggingface_walkthrough.md`](./huggingface_walkthrough.md)     |
| OpenAI        | [`openai_playground_walkthrough.md`](./openai_playground_walkthrough.md) |
| Google AI     | [`gemini_studio_walkthrough.md`](./gemini_studio_walkthrough.md) |

Each page includes screenshots, UI tips, and code export guidance.

---

## 🔐 API Key Setup & Security

Learn what APIs are, how to register, and best practices for storing your tokens:

📘 [`api_key_setup.md`](./api_key_setup.md)

Includes:
- What is an API (in depth)
- Where to generate keys for each platform
- Security do’s and don’ts

---

## 📦 Code-Based Access

Jumpstart with Colab/Python-ready snippets:

📘 [`api_inference_quickstart.md`](./api_inference_quickstart.md)

Covers:
- Minimal working examples for each provider
- Output parsing tips
- Common troubleshooting steps

---

## 🔁 Modular Code Blocks

Reusable function wrappers from `codebook/apis/`:

| Module                        | Description                                |
|------------------------------|--------------------------------------------|
| [`openai_api_basic_call.md`](../../codebook/apis/openai_api_basic_call.md)     | OpenAI `ChatCompletion` function            |
| [`gemini_api_basic_call.md`](../../codebook/apis/gemini_api_basic_call.md)     | Gemini SDK wrapper                         |
| [`hf_inference_api_call.md`](../../codebook/apis/hf_inference_api_call.md)     | Hugging Face REST call                     |
| [`hf_local_transformer_pipeline.md`](../../codebook/apis/hf_local_transformer_pipeline.md) | Local Transformers via `pipeline()`        |

---

## 🤖 Try It: Multi-Provider Prompt Comparison

Use this sandbox notebook to run a **side-by-side prompt test** with Gemini, OpenAI, and Hugging Face:

📓 [`multi_provider_sandbox.ipynb`](../../notebooks/Day1_Session2/multi_provider_sandbox.ipynb)

---

## 📊 Suggested Visuals (Optional)

- API flow diagrams for each platform
- Model selection and prompting architecture
- Platform comparison matrix (Figma)
- Error handling flowcharts

---

## 🧠 Coming Up Next

You’ll apply these tools in:
- Prompt engineering experiments (Session 3)
- Embedding-based workflows (Day 2)
- Agent pipelines and RAG (Day 3)

---

## 🔗 Related Sessions

- [`day1_intro_to_llms.md`](./day1_intro_to_llms.md)
- [`day1_prompt_engineering.md`](./day1_prompt_engineering.md)
- [`day2_embeddings_basics.md`](../day2/day2_embeddings_basics.md)

---