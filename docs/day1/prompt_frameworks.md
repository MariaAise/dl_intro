---

---


# 🧠 Prompt Engineering Frameworks

> “Frameworks turn intuition into structure. Prompting is no different — a framework lets you reason systematically with language models.”

This page introduces the most widely used prompt engineering frameworks and shows how to apply them across OpenAI, Gemini, Claude, Hugging Face, and product-level interfaces.

Each section includes:

- ✅ **Core concept**
- 🎯 **Best use cases**
- ❌ **Limitations**
- 💡 **Pro tips**
- 🧪 **Implementation example**
- 🛠 **Hugging Face Example**
- 🔁 **Sample Workflow**

---

## 🔹 Zero-shot Prompting

**Core idea:** Ask the model to perform a task directly, without any examples. Relies on the model’s general training.

✅ **Best For:**

- Simple tasks (summarisation, classification, Q&A)
- Rapid prototyping
- When examples are not available

❌ **Limitations:**

- Prone to hallucination or vague output
- Highly dependent on phrasing
- May not align to task expectations without formatting hints

💡 **Pro Tips:**

- Add structure: “Respond in three bullet points.”
- Use role priming: “You are a...”
- Be direct with verbs: “Summarise, Classify, Extract...”

🧪 **Example:**

```text
Summarise this paragraph in three bullet points:

"The new policy proposal aims to reduce urban emissions by 40% by 2030..."
```

🛠 **Hugging Face Implementation:**

```python
from transformers import pipeline
summarizer = pipeline("summarization")
text = "The new policy proposal aims to reduce urban emissions by 40% by 2030..."
summary = summarizer(text, max_length=60, min_length=10, do_sample=False)
```

🔁 **Sample Workflow:**

1. Copy/paste text from article, report, policy.
2. Load `summarization` pipeline from HF.
3. Adjust length and sampling.
4. Review and refine format if needed.

---

## 🔹 Few-shot Prompting

**Core idea:** Provide 1–5 example input-output pairs before the actual input to guide the model.

✅ **Best For:**

- Style imitation
- Classification with nuance
- Custom logic where few examples clarify pattern

❌ **Limitations:**

- Takes more tokens (context limit concerns)
- Quality of examples heavily affects outcome
- Fragile if formatting is inconsistent

💡 **Pro Tips:**

- Keep format exactly consistent across examples
- Use diverse edge cases if task is complex
- Anchor outputs with clear labels (e.g., → support)

🧪 **Example:**

```text
Decide if the following quotes express support or opposition:

- "This change is long overdue." → support
- "This will increase our costs unfairly." → oppose

Statement: "This proposal seems rushed and underdeveloped."
```

🛠 **Hugging Face Implementation:**

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="facebook/bart-large-mnli")
classifier("This proposal seems rushed and underdeveloped.")
```

🔁 **Sample Workflow:**

1. Manually define few-shot prompt if using OpenAI/Gemini.
2. For HF, simulate few-shot by running examples separately.
3. Use labels for consistent evaluation.
4. Export structured labels for later training.

---

## 🔹 Chain-of-Thought (CoT)

**Core idea:** Ask the model to show intermediate reasoning steps before answering. This improves performance on complex reasoning tasks.

✅ **Best For:**

- Logic or math problems
- Policy justification
- Anything requiring rationale or step-by-step explanation

❌ **Limitations:**

- Output can be verbose
- Reasoning may not be logically valid — just fluent

💡 **Pro Tips:**

- Use explicit trigger phrases: “Let’s think step by step.”
- Ask for rationale *before* final answer
- Combine with formatting constraints for control

🧪 **Example:**

```text
Question: A conference has 120 attendees. Half are from government, and a quarter are from academia. How many are from other sectors?

Let’s think step by step:
1. Half of 120 = 60 (government)
2. A quarter of 120 = 30 (academia)
3. 120 - 60 - 30 = 30 (other sectors)
Final Answer: 30
```

🛠 **Hugging Face Implementation:**

```python
prompt = """
Q: A conference has 120 attendees. Half are from government, and a quarter are from academia. How many are from other sectors?
A: Let's think step by step:
"""

from transformers import pipeline
cot = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")
cot(prompt, max_new_tokens=100)
```

🔁 **Sample Workflow:**

1. Embed reasoning scaffold in prompt.
2. Choose model with instruction-tuned weights.
3. Postprocess for final answer extraction.

---

## 🔹 ReAct (Reasoning + Acting)

**Core idea:** Combine thought traces with external tool usage (e.g., retrieval, calculations, search). Originates from agent-based prompting.

✅ **Best For:**

- Agents / tool use (LangChain, OpenAI functions)
- Retrieval-Augmented Generation (RAG)
- Multi-hop logic

❌ **Limitations:**

- Requires framework support (e.g., function calling or memory loop)
- Not universally supported in playgrounds

💡 **Pro Tips:**

- Follow exact format: Thought → Action → Observation → Repeat
- Can chain multiple tools if memory is handled properly

🧪 **Example:**

```text
Question: What’s the population of the capital city of the country where the 2022 FIFA World Cup was held?

Thought: The 2022 World Cup was in Qatar.
Action: Lookup("Capital of Qatar")
Observation: Doha
Action: Lookup("Population of Doha")
Observation: 2.3 million
Final Answer: 2.3 million
```

🛠 **Hugging Face Implementation:** Use LangChain with Hugging Face model and tools:

```python
from langchain.llms import HuggingFacePipeline
from langchain.agents import Tool, initialize_agent
from transformers import pipeline

llm = HuggingFacePipeline(pipeline("text-generation", model="tiiuae/falcon-7b-instruct"))
tools = [Tool.from_function(name="Lookup", func=my_search_function)]
agent = initialize_agent(tools, llm, agent_type="react-description")
agent.run("What’s the population of the capital city of the country where the 2022 FIFA World Cup was held?")
```

🔁 **Sample Workflow:**

1. Format prompt as ReAct trace
2. Wrap HF model into LangChain agent
3. Bind to external tools (search, DB)

---

## 🔹 ToT (Tree of Thought)

**Core idea:** Explore multiple reasoning paths in parallel and select the best. Used in advanced orchestration with model voting or branching logic.

✅ **Best For:**

- High-stakes reasoning (medical, legal, finance)
- Planning or optimization problems

❌ **Limitations:**

- Not supported in single call; requires orchestrator
- Slower, more compute-heavy

💡 **Pro Tips:**

- Use when multiple “candidate” solutions need comparison
- Combine with prompt templates + scoring heuristics

🧪 **Example (simplified):**

```text
Task: Suggest the best relocation city based on cost, safety, and opportunity.

Option A: ...
Option B: ...
Option C: ...

Evaluate pros and cons. Choose best.
```

🛠 **Hugging Face Implementation:** Currently requires custom orchestration:

- Generate 3 completions using `num_return_sequences=3`
- Score or filter using a follow-up ranking prompt

🔁 **Sample Workflow:**

1. Generate N options with sampling or top-p
2. Prompt LLM again: “Evaluate and choose best”
3. Use scoring template to extract winning response

---

## 🧱 Framework Summary Table

| Framework | Best For                | Limitations               | Trigger Phrase                   |
| --------- | ----------------------- | ------------------------- | -------------------------------- |
| Zero-shot | Simple tasks            | Prone to vague answers    | “Summarise...”                   |
| Few-shot  | Classification, mimicry | Token cost, brittle       | “Example: ...”                   |
| CoT       | Logic & math            | Verbose, not always valid | “Let’s think step by step.”      |
| ReAct     | Tools + steps           | Needs orchestration       | “Thought → Action → Observation” |
| ToT       | Complex eval            | Not standalone            | Tree or branch logic             |

---

## 📎 Related Modules

- [Prompt Anatomy →](prompt_anatomy.md)
- [Prompt Failures →](prompt_failures.md)
- [Evaluation Methods →](prompt_evaluation.md)
- [Mini Projects →](mini_project_templates.md)

---

