---

---


# 📏 Prompt Evaluation & Scoring Frameworks

> “If you can’t evaluate it, you can’t improve it.”

This module introduces structured methods to evaluate LLM-generated responses, especially when prompts are reused, shared, or productized.

You’ll learn:

- ✅ What makes an output ‘good’ or ‘bad’?
- 🧠 Rubric-based scoring patterns
- 🛠 Implementation via notebooks or app UIs
- 🔁 Self-grading and auto-evaluation techniques

---

## 🧠 When & Why to Evaluate

Use evaluation when:

- Comparing different prompts or models
- Testing outputs across formats or tones
- Automating client- or publication-facing outputs
- Measuring consistency over time

---

## 🧰 Evaluation Rubric (Generic)

| Criterion         | Description                                 | Scale |
| ----------------- | ------------------------------------------- | ----- |
| Relevance         | Output matches the task and prompt goals    | 1–5   |
| Completeness      | All parts of the instruction are addressed  | 1–5   |
| Clarity           | Writing is clear, unambiguous, and readable | 1–5   |
| Reasoning Quality | Any logic used is valid, well-structured    | 1–5   |
| Format Fidelity   | Output matches formatting constraints       | 1–5   |

**Use Case:** Attach this to any mini project task as an evaluation layer.

---

## 🔍 Self-Grading Prompt Example

```text
Evaluate the following LLM output based on: Relevance, Clarity, and Format (1–5 each). Justify each rating.

Output:
"This study discusses several topics. It is important."
```

**Expected Output:**

```json
{
  "relevance": [2, "Does not answer specific task"],
  "clarity": [2, "Too vague to be meaningful"],
  "format": [3, "Roughly resembles instruction but lacks structure"]
}
```

---

## 🧪 Implementation in Notebooks

1. Load generated outputs from LLM task
2. Apply rubric programmatically
3. Ask another LLM to score (meta-model evaluation)
4. Compare across prompt versions

Example Cell:

```python
scores = evaluate_output(prompt_id="summary_v1", response=text)
display(scores)
```

---

## 🧩 Related: Evaluation Interface Pattern

As we move from theory into real-world workflows, you’ll begin to test different prompt versions — and you’ll need a way to score their outputs systematically.

To support this, we introduce a **lightweight evaluation interface** that allows you to:
- Select a prompt version
- Paste a generated response
- Rate it using sliders (e.g. relevance, completeness, clarity)
- Get a total score and brief summary
- Export the results to CSV or Notion

👉 [View the Interface Pattern →](../../codebook/day1/prompt_eval_interface.md)

---

## 🔁 Comparative Prompt A/B Testing

**What it is:** A/B testing allows you to compare two different versions of the same prompt — or model outputs — and choose which one performs better.

This is useful when:

- You’ve rewritten a prompt and want to validate if the new version improves quality
- You're choosing between zero-shot and few-shot formatting
- You want human or LLM feedback on clarity, tone, or reasoning

**Example format:**

```yaml
prompt_a: [response1]
prompt_b: [response2]
criteria: [clarity, relevance, format]
```

Then:

- Ask users or a model to judge which response better meets the criteria and explain why.
- Use scores to improve future prompt iterations or select defaults in your UI.

A/B testing can be as simple as a Google Form or as complex as a Streamlit app.

---

## 🧠 Metrics to Track Over Time

| Metric         | Why It Matters                |
| -------------- | ----------------------------- |
| Average score  | General quality level         |
| Variance       | Consistency of results        |
| Format errors  | Prompt failure detection      |
| Time to answer | Latency in UX or API settings |

These can be logged per prompt or per project in a prompt registry.

---

## 🕵️‍♀️ Error Analysis Toolkit

### 1. **Common Failure Modes**

- Vague completions
- Hallucinated citations
- Format drift
- Misaligned tone or role

```python
# Top error categories
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eval_outputs.csv")
df['error_type'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Failure Mode Distribution")
plt.show()
```

### 2. **Recovery Protocols**

- **Immediate:** Retry with fallback prompt (e.g., add structure)
- **Mid-term:** Prompt refinement using examples or role alignment
- **Long-term:** Retrain embedding classifier or scoring heuristic

---

## 📊 Quantitative Metrics Table

| Metric Name    | Code Snippet                         | Description                  |
| -------------- | ------------------------------------ | ---------------------------- |
| Output length  | `len(output.split())`                | Detect verbosity             |
| JSON valid     | `json.loads(output)`                 | Format reliability           |
| Readability    | `textstat.flesch_reading_ease(text)` | Clarity measure              |
| Score variance | `np.var(scores)`                     | Evaluate scoring consistency |

---

## 🧭 Domain-Specific Rubrics

### Health Policy

| Criterion    | Description                              |
| ------------ | ---------------------------------------- |
| Risk Framing | Is health risk framed accurately?        |
| Equity       | Does output reflect diverse populations? |

### Legal Summary

| Criterion      | Description                               |
| -------------- | ----------------------------------------- |
| Precision      | Are legal terms used appropriately?       |
| Interpretation | Is statutory language simplified clearly? |

### Academic Rewrites

| Criterion        | Description                        |
| ---------------- | ---------------------------------- |
| Jargon Reduction | Is technical language reduced?     |
| Audience Fit     | Does output match stated audience? |

---

## 🎯 Strategic Extensions

### 👥 Human-in-the-Loop Evaluation

```markdown
- Create shared Google Sheet rubric
- Invite team to score outputs across rows
- Use Cohen’s Kappa for inter-rater reliability
```

### 💸 Cost-Performance Tradeoffs

```markdown
| Model          | Cost/1k | Avg. Score | Speed |
|----------------|---------|------------|-------|
| gpt-4          | $0.30   | 4.2        | 1.5s  |
| gpt-3.5        | $0.004  | 3.5        | 0.7s  |
| Claude Sonnet  | —       | 4.1        | 0.8s  |
```

### 🛡️ Regulatory Compliance

```markdown
### FDA-GPT Evaluation Checklist
- 21 CFR Part 11 logging requirements
- Output must include audit trail (timestamp, version, reviewer)
- Secure retention of logs for 5+ years
```

---

## ✅ Summary

Prompt evaluation is more than scoring — it's about **building confidence** in your pipeline.

By applying rubrics, metrics, A/B testing, and human review patterns, you can:

- Identify where and how your prompt fails
- Improve quality with data-driven revisions
- Track change over time and justify output reliability

These tools are essential for **MVPs, research prototypes, and production systems** alike.

---

📎 [Prompt Failures →](prompt_failures.md) 📎 [Mini Projects →](mini_project_templates.md) 📎 [Prompt Frameworks →](prompt_frameworks.md) 📎 [Prompt Modularity →](prompt_modularity.md)

