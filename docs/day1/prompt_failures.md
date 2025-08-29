---

---


# ❌ Prompt Failure Modes and Fixes

> "A failed prompt is a diagnostic opportunity, not a dead end."

Prompting doesn’t always go right — especially as tasks get more complex. This module helps you recognize common failure types, debug them methodically, and apply rewrites or framework shifts that recover performance.

Each failure mode includes:

- 🔍 **Symptoms**
- ⚠️ **Root Cause**
- 🧪 **Example**
- 🔁 **Recovery Pattern**
- 🛠 **Fix in Hugging Face / OpenAI / Gemini**

---

## 🔸 Failure Type 1: Vague or Off-Topic Output

🔍 **Symptoms:**

- The output wanders from the topic
- The answer feels generic or boilerplate

⚠️ **Cause:**

- Instruction is too vague (e.g. "analyze this")
- No constraints on topic or tone

🧪 **Example:**

```text
Prompt: Analyze the document below.

Output: "This document discusses various important aspects and should be considered carefully."
```

🔁 **Recovery Pattern:**

- Add tighter scope: "Summarize the document in 3 key risks."
- Specify audience: "As a legal analyst, extract the risk exposure."

🛠 **Fix Tip (HF):** Use structured summarization:

```python
summarizer("Summarize this policy in 3 bullet points")
```

---

## 🔸 Failure Type 2: Hallucination / Fabricated Facts

🔍 **Symptoms:**

- Cites nonexistent laws, people, or data
- Fabricates URLs or references

⚠️ **Cause:**

- Prompt asks for information beyond training
- Model is forced to "guess"

🧪 **Example:**

```text
Prompt: Who was the Chief Economist at the IMF in 2024?
Output: "Dr. Stephanie Kingsley held that position."
[Note: Person doesn't exist.]
```

🔁 **Recovery Pattern:**

- Reframe as conditional: "If unknown, respond 'unknown'."
- Add role: "You are a careful researcher. Do not guess."

🛠 **Fix Tip (OpenAI):** Use `temperature=0.0` to reduce creative guesses

---

## 🔸 Failure Type 3: Format Drift / Output Doesn’t Match Template

🔍 **Symptoms:**

- Output varies in format
- Model skips or reorders fields

⚠️ **Cause:**

- Prompt lacks strong examples or rigid schema
- Model is unsure what formatting matters

🧪 **Example:**

```text
Prompt: Extract JSON:
{
  "policy_theme": "",
  "risk": "",
  "recommendation": ""
}

Output:
"The policy is focused on economic growth and suggests improving tax rates."
```

🔁 **Recovery Pattern:**

- Show full example in correct format
- Phrase clearly: “Return in *exact* format below”

🛠 **Fix Tip (Gemini):** Use system prompt to force JSON (model name: `gemini-pro`)

---

## 🔸 Failure Type 4: Prompt Misinterpreted / Wrong Task

🔍 **Symptoms:**

- Model answers a different question
- Misses goal completely (e.g., generates a title instead of summary)

⚠️ **Cause:**

- Instruction unclear or implies wrong task
- Example phrasing differs from goal

🧪 **Example:**

```text
Prompt: Provide a one-sentence overview.
Output: "Overview: The topic discussed is immigration law."
[Expected: full sentence summary]
```

🔁 **Recovery Pattern:**

- Rewrite instruction to match cognitive load: “Summarise in one fluent sentence.”
- Use example pairs to reinforce task

🛠 **Fix Tip:** Use few-shot prompting to anchor correct interpretation.

---

## 🔸 Failure Type 5: Verbosity / Overlong Output

🔍 **Symptoms:**

- Output is too long or includes disclaimers

⚠️ **Cause:**

- No token/word limits
- Role framing too generic (e.g., “You are helpful”)

🧪 **Example:**

```text
Prompt: Briefly explain the key benefits of this program.
Output: "As an AI language model developed by OpenAI, I cannot provide..."
```

🔁 **Recovery Pattern:**

- Add strict constraints: “Answer in <50 words” or “3 bullet points max”
- Remove “You are helpful...” defaults

🛠 **Fix Tip:** Use OpenAI `max_tokens` + post-trim filter if needed.

---

## 🧭 Next → [Prompt Modularity](prompt_modularity.md)

-

## 📎 Related Modules

- [Prompt Anatomy →](prompt_anatomy.md)
- [Prompt Frameworks →](prompt_frameworks.md)
- [Evaluation Methods →](prompt_evaluation.md)
- [Mini Projects →](mini_project_templates.md)

