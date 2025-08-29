---

---

# 🧪 Mini Project Templates (Prompting)

> “The best way to master prompting is to *apply* it on a problem you understand.”

This page offers plug-and-play **prompt templates** that participants can use to:

- Apply prompting frameworks to real-world tasks
- Adapt templates to their own research or sector
- Practice on their own after the live session

Each template includes:

- 🧠 Task
- 🧱 Prompt structure
- 🛠️ Example (in plain text)
- 🔁 Variant paths to try (Zero-shot, Few-shot, CoT, Structured)
- 🔬 Technical or research implementation pattern

---

## 🧠 Research Hypothesis Generator

**Task:** Generate testable research hypotheses based on literature or qualitative insights.

**Prompt Template:**

```yaml
system: "You are a PhD-level research assistant."
task: "Generate 3 research hypotheses based on the following excerpt."
format: Bullet list, each < 30 words
input: [Paste interview summary, paper excerpt, or report finding]
```

**Example Output:**

- Users with chronic illness are more likely to use online forums for emotional support.
- Privacy concerns reduce likelihood of AI adoption in regional clinics.
- Teacher burnout correlates with perceived policy instability.

**Variants:**

- Add method suggestion: “Suggest method to test each hypothesis”
- Switch from bullet to structured table (theme | hypothesis | method)
- Use this to build surveys or protocols

**Technical Path:**

- Chain to a JSON-to-CSV converter for survey builders
- Build prompt registry under `prompts/research/hypothesis_builder.md`
- Use in CoT format for literature synthesis-based generation

---

## 📊 Research Summary Comparator

**Task:** Compare two article abstracts or study summaries across dimensions (e.g. theory, data, region).

**Prompt Template:**

```yaml
system: "You are a comparative research analyst."
task: "Compare these two abstracts across topic, method, region, and findings."
format: Markdown table
input: [Abstract A] [Abstract B]
```

**Example Output:**

| Category | Abstract A                  | Abstract B                        |
| -------- | --------------------------- | --------------------------------- |
| Topic    | climate risk                | adaptation planning               |
| Method   | interview                   | panel regression                  |
| Region   | Australia                   | ASEAN                             |
| Findings | social trust moderates risk | migration patterns shape response |

**Variants:**

- Include rating for strength of method
- Extend into automatic synthesis: “What is the policy implication of both combined?”

**Technical Notes:**

- Prompt lives in `prompts/research/compare_articles.yaml`
- Use in paired dataset workflow: compare hundreds of abstracts programmatically

---

## 💼 Business Strategy Evaluator

**Task:** Evaluate a business strategy case (e.g. startup plan, public document, press release) across viability, innovation, and risk.

**Prompt Template:**

```yaml
system: "You are a strategic analyst evaluating a business plan."
task: "Score this strategy across 3 dimensions: viability, innovation, execution risk."
format: JSON
input: [Paste strategy, executive summary, or pitch deck text]
```

**Example Output:**

```json
{
  "viability": [4, "Target market is well-defined, pricing model tested"],
  "innovation": [3, "Moderately novel, but competitors exist"],
  "execution_risk": [2, "Team has prior experience and capital"]
}
```

**Variants:**

- Use multiple roles: analyst, investor, CTO
- Add CoT rationale: “Explain each score before providing”
- Use slider interface in Streamlit to manually adjust scores

**Research Application:**

- Compare business models in case studies
- Automate pre-screening of MBA venture projects
- Link prompt output to rubric/learning outcome system

**Technical Pattern:**

- Store prompt as `prompts/business/strategy_eval.md`
- Call via Streamlit + plot radar chart
- Export JSON to Airtable or Notion

---

📎 [Download Starter Notebook →](link) 📎 [See Prompt Anatomy →](prompt_anatomy.md) 📎 [See Prompt Frameworks →](prompt_frameworks.md) 📎 [Try Prompt Evaluation →](prompt_evaluation.md)

