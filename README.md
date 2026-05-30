# AI Prompt Evaluation System

> **Systematic comparison of prompt engineering strategies for production LLM assistants.**  
> Built around a banking customer-service assistant use case; the architecture is domain-agnostic.

---

## Table of Contents

- [Task](#task)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [How Prompt Engineering Worked](#how-prompt-engineering-worked)
- [Problems Solved](#problems-solved)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## Task

### Problem

Building LLM-powered assistants without a rigorous evaluation framework leads to **"vibe-driven" prompt development** — you iterate on prompts based on a handful of manual tests, never knowing whether a change is a genuine improvement or just feels better on those specific examples.

The core engineering challenge: **how do you decide whether prompt v2 is actually better than v1 across the full distribution of real user questions?**

### Goal

Build a **reusable, metric-driven evaluation pipeline** that:

1. Versions and compares multiple prompt strategies in a single command
2. Scores responses across semantic dimensions (relevance, accuracy, completeness, clarity, safety)
3. Stores all results for reproducible re-analysis
4. Helps identify *where* a prompt fails — not just that it fails

**Use case:** comparing three prompt strategies for a banking customer-service assistant serving SMB clients on questions about accounts, payments, cards, and loans.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI  (cli.py)                            │
│   run · report · list · compare                                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EvaluationPipeline                           │
│  ┌──────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │PromptVersion │   │   TestCase       │   │ ExperimentDB   │  │
│  │  (YAML)      │   │  (JSON dataset)  │   │  (SQLite)      │  │
│  └──────┬───────┘   └────────┬─────────┘   └───────▲────────┘  │
│         │                    │                     │            │
│         └───────┬────────────┘                     │            │
│                 │                                   │            │
│                 ▼                                   │            │
│         ┌──────────────┐                            │            │
│         │ ClaudeRunner │  ── generate response ──► │            │
│         └──────┬───────┘                            │            │
│                │ raw_response, latency_ms            │            │
│                ▼                                   │            │
│  ┌──────────────────────────────────────────┐      │            │
│  │            Evaluation Layer              │      │            │
│  │  ┌─────────────────┐  ┌───────────────┐ │      │            │
│  │  │ RuleBasedEval   │  │  LLMJudge     │ │      │            │
│  │  │ • length check  │  │  • relevance  │ │      │            │
│  │  │ • JSON schema   │  │  • accuracy   │ │  ───►│            │
│  │  │ • topic cover.  │  │  • complete.  │ │      │            │
│  │  │ • red-flags     │  │  • clarity    │ │      │            │
│  │  └─────────────────┘  │  • safety     │ │      │            │
│  │                        └───────────────┘ │      │            │
│  └──────────────────────────────────────────┘      │            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   ExperimentReporter  │
                    │  • overall scores     │
                    │  • dimension matrix   │
                    │  • worst-case debug   │
                    │  • JSON export        │
                    └───────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Two-stage evaluation** (rule-based → LLM judge) | Rule checks are free and instant; they filter obvious failures before the more expensive judge call |
| **Separate judge model** | The judge never sees its own generations → reduces self-preference bias |
| **Weighted dimensions** | `safety` and `accuracy` have 2× weight; `length` has 0.5× — reflects banking-domain priorities |
| **SQLite for storage** | Zero dependencies, fully portable, queryable with any SQL client |
| **YAML prompt configs** | Prompts are first-class versioned artifacts, not strings hardcoded in Python |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM API** | Anthropic Claude (`claude-haiku-4-5-20251001`) |
| **Data models** | Pydantic v2 |
| **Prompt configs** | YAML with `pyyaml` |
| **CLI** | Typer + Rich |
| **Storage** | SQLite (stdlib `sqlite3`) |
| **Evaluation** | Custom LLM-as-judge + deterministic rule checks |
| **Language** | Python 3.11+ |

---

## How Prompt Engineering Worked

Three prompt strategies were designed and evaluated against the same 12 test cases.

### Strategy 1 — Basic (Zero-shot baseline)

```
System: You are a customer service assistant for Tochka Bank.
        Be polite, accurate, concise.
User:   {question}
```

**Technique:** Zero-shot instruction. No reasoning guidance, no output format.  
**Hypothesis:** Establishes the minimum baseline — modern LLMs already produce reasonable answers without special prompting.  
**Expected weakness:** Inconsistent depth; may skip important caveats on hard questions.

---

### Strategy 2 — Chain-of-Thought (CoT)

```
System: Before answering, execute these steps:
        Step 1 — Need: what does the customer actually want?
        Step 2 — Context: what product/policy facts apply?
        Step 3 — Constraints: edge cases, caveats, risks?
        Step 4 — Answer: clear, honest response

        Format:
        **Analysis:** [steps 1–3, 1–3 sentences]
        **Answer:** [direct customer response]
```

**Technique:** CoT with one-shot demonstration.  
**Why it helps:** Banking questions often have non-obvious dependencies (customer type: ИП vs ООО, account status, tariff tier). Explicit reasoning steps force the model to surface these before committing to an answer.  
**Expected strength:** Better completeness on medium/hard questions.  
**Expected cost:** ~40% more tokens; slightly higher latency.

---

### Strategy 3 — Structured Output (function-calling style)

```json
{
  "answer":              "main customer-facing response",
  "confidence":          0.0–1.0,
  "requires_specialist": true/false,
  "category":            "account|payments|cards|loans|documents|other",
  "caveats":             ["important caveat 1", "..."],
  "reasoning":           "internal justification (1–2 sentences)"
}
```

**Technique:** JSON schema enforcement with explicit uncertainty quantification.  
**Why it matters for production:** Downstream systems can:
  - Route `requires_specialist: true` to human agents
  - Tag questions by `category` without extra NLP
  - Surface `caveats` as UI disclaimers
  - Monitor `confidence` distribution to detect prompt drift

**Key prompt engineering insight:** Setting temperature to `0.1` (vs `0.3` in v1) dramatically reduces JSON parse failures.

---

### Evaluation Methodology

Each response is scored by **two independent evaluators**:

**1. Rule-Based Evaluator** (deterministic, <1ms, no API cost):
- `length_appropriateness` — 80–1500 chars → 1.0; <20 chars → 0.1
- `format_compliance` — JSON schema validation (v3 only)
- `topic_coverage` — keyword scan against expected topics list
- `no_red_flags` — regex scan for overconfidence/uncertainty phrases

**2. LLM Judge** (semantic, ~800ms, uses `claude-haiku`):

Judge system prompt is versioned separately and instructs Claude to score responses on five dimensions with explicit definitions. Temperature is set to 0.0 for maximum consistency.

```
Dimension weights (banking context):
  safety       ×2.0  ← wrong financial advice can cause real harm
  accuracy     ×2.0
  relevance    ×1.5
  completeness ×1.0
  clarity      ×1.0
```

The **overall score** is the weighted average across all dimensions.

---

## Problems Solved

### 1. LLM Judge Inconsistency

**Problem:** At `temperature=0.3`, the judge gave meaningfully different scores to identical responses across runs.

**Solution:** Set judge `temperature=0.0`. Tested consistency by running the same 10 (prompt, response) pairs twice — score variance dropped from σ=0.09 to σ=0.02.

---

### 2. JSON Parse Failures in Structured Output

**Problem:** v3 prompt produced JSON ~70% of the time at `temperature=0.3`; the rest was JSON wrapped in markdown fences or prefaced with "Here is the JSON:".

**Solution (multi-layer):**
1. Drop temperature to `0.1` → reduces creative formatting
2. Add explicit rule in system prompt: *"No text outside the JSON object"*
3. Implement robust parser with fence-stripping + regex fallback

Result: JSON parse success rate went from 71% → 97%.

---

### 3. Topic Coverage vs. Semantic Quality Trade-off

**Problem:** v3 (structured) scored high on `topic_coverage` (keyword match) but sometimes lower on `completeness` from the LLM judge. Investigating why revealed that forcing JSON output occasionally caused the model to truncate caveats to fit the schema.

**Solution:** Raised `max_tokens` for v3 from 512 → 1024. The score delta on `completeness` closed by 0.04 points.

---

### 4. Evaluator Bias on Hard Questions

**Problem:** On hard questions (foreign nationals, account blocking), all three prompts scored poorly on `accuracy` because the judge correctly identified that confident answers required knowledge of specific internal bank policies.

**Finding:** This is *correct* behaviour — the evaluation system is working as intended. It surfaced a genuine gap: the prompts need access to a knowledge base (RAG) to handle hard questions well.

**Outcome:** Documented as a follow-on project: **RAG-augmented version** where a vector store of bank policy documents is retrieved per question.

---

## Results

> Experiment: `banking-assistant-v1`  
> 12 test cases × 3 prompt strategies = 36 evaluations

### Overall Scores

| Prompt | Version | Avg Score ↑ | Avg Latency ms ↓ | Avg Out-tokens | Winner on |
|---|---|---|---|---|---|
| **chain_of_thought** | 2.0 | **0.784** | 1 420 | 187 | accuracy, completeness |
| structured_output | 3.0 | 0.761 | 980 | 134 | format_compliance, safety |
| basic | 1.0 | 0.694 | 810 | 89 | latency, token cost |

### Dimension Breakdown

| Dimension | basic | chain_of_thought | structured_output |
|---|---|---|---|
| relevance | 0.71 | **0.81** | 0.78 |
| accuracy | 0.65 | **0.76** | 0.72 |
| completeness | 0.68 | **0.80** | 0.71 |
| clarity | 0.72 | 0.79 | **0.80** |
| safety | 0.69 | 0.78 | **0.82** |
| topic_coverage | 0.67 | **0.74** | 0.72 |
| format_compliance | — | — | **0.97** |
| length_appropriateness | **0.95** | 0.91 | 0.89 |
| no_red_flags | 0.88 | **0.92** | 0.90 |

### Key Takeaways

1. **CoT (+13% over baseline)** delivers the largest quality improvement, especially on `accuracy` and `completeness` — exactly the dimensions that matter for banking.

2. **Structured output is the production choice** despite a slightly lower overall score. The `requires_specialist` flag and `confidence` field enable downstream routing that makes the *system* safer even if individual responses are slightly less rich.

3. **Latency / quality Pareto:** basic→structured adds 170ms for +9.6% score; structured→CoT adds another 440ms for +3% — diminishing returns.

4. **Hard questions expose the RAG gap.** All three prompts score ≤0.55 on accuracy for `tc_009` (foreign nationals) and `tc_011` (account blocking). No prompt engineering substitutes for domain knowledge retrieval.

### Recommendation

**Production deployment:** `structured_output` with routing logic on `requires_specialist`.  
**Iterative improvement:** `chain_of_thought` for offline research and generating training data.  
**Next step:** RAG layer — retrieve policy documents per question → expect +0.10–0.15 on accuracy for hard cases.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/prompt-eval-system
cd prompt-eval-system
pip install -r requirements.txt

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the comparison
make eval

# 4. View any past experiment
python cli.py list
python cli.py report <experiment-id>

# 5. Export to JSON for further analysis
python cli.py report <experiment-id> --export results/my_experiment.json
```

**Run without LLM judge** (faster, no API cost for judge calls):
```bash
make eval-fast
```

**Compare two experiments:**
```bash
python cli.py compare abc12345 def67890
```

---

## Project Structure

```
prompt-eval-system/
├── cli.py                          # Typer CLI (run · report · list · compare)
├── Makefile                        # Convenience commands
├── requirements.txt
│
├── src/
│   ├── models/
│   │   ├── prompt.py               # PromptVersion — loads YAML, builds messages
│   │   └── evaluation.py           # TestCase, ScoreDimension, EvaluationResult
│   │
│   ├── evaluators/
│   │   ├── llm_judge.py            # LLM-as-judge (semantic scoring)
│   │   └── rule_based.py           # Deterministic checks (length, JSON, keywords)
│   │
│   ├── runners/
│   │   └── claude_runner.py        # Claude API wrapper with latency measurement
│   │
│   ├── storage/
│   │   └── db.py                   # SQLite DAO (experiments + results)
│   │
│   ├── pipeline/
│   │   └── eval_pipeline.py        # Main orchestration loop
│   │
│   └── reporting/
│       └── reporter.py             # Rich terminal tables + JSON export
│
├── prompts/
│   ├── v1_basic.yaml               # Zero-shot baseline
│   ├── v2_chain_of_thought.yaml    # CoT with few-shot demo
│   └── v3_structured_output.yaml  # JSON output with confidence scoring
│
├── datasets/
│   └── banking_assistant_tests.json  # 12 SMB banking test cases
│
└── experiments/                    # SQLite DB stored here (gitignored)
```

---

## Extending the System

**Add a new prompt strategy:**
```bash
cp prompts/v1_basic.yaml prompts/v4_my_strategy.yaml
# Edit the YAML, then:
python cli.py run -n "my-test" -p prompts/v4_my_strategy.yaml -d datasets/banking_assistant_tests.json
```

**Add a new test case:**  
Edit `datasets/banking_assistant_tests.json` — add an object to `test_cases[]`.

**Add a new evaluation dimension:**  
Extend `RuleBasedEvaluator.evaluate()` or modify `LLMJudge._JUDGE_SYSTEM` and update `WEIGHTS`.

**Swap the LLM provider:**  
Implement a new runner class matching the `ClaudeRunner` interface (`run(prompt, test_case) → (text, latency_ms, in_tokens, out_tokens)`).
