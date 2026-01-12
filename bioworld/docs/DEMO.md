# BioWorld OS — Demo Walkthrough

A script for running and presenting the POC. Two modes: **interactive UI**
(recommended for a live demo) and **headless** (for a quick reproducibility
check or CI-style smoke test).

---

## 1. Prerequisites

- Python 3.11 (Anaconda) on the POC machine (i5, 4 GB RAM, no GPU).
- Dependencies installed (see README §Setup).
- `bioworld/.env` present with at least `OPENROUTER_API_KEY`
  (and `TAVILY_API_KEY` if you want evidence search). Write it via Python to
  avoid a UTF-8 BOM.
- Default provider: `openrouter` → `openai/gpt-oss-20b:free`.

Verify the provider one-liner works before the demo:

```powershell
$env:BIOWORLD_PROVIDER = "openrouter"
python -c "from bioworld.agent import build_model; m = build_model(); print(type(m).__name__, 'OK')"
```

---

## 2. Option A — Interactive UI (recommended)

```powershell
cd E:\ai-safety
streamlit run bioworld/app.py
```

Browser opens (default `http://localhost:8501`). You see four areas:

1. **Question input** — pre-filled with the default DR question.
2. **Dataset path** — pre-filled with `data\debrecen\messidor_features.arff`.
3. **▶ Run agent** button.
4. On run: **Agent activity** (left) and **Sandbox console + Leaderboard**
   (right); after the run: **Artifacts** and **Final report**.

### What to say / do

Keep the default question for a clean 3-model comparison, or try:

- *"Profile the dataset and tell me about class imbalance."*
- *"Train logistic regression and random forest, then tell me which is better
  and why."*
- *"Explain the model with SHAP and run one what-if: what happens if the
  patient's glucose-equivalent feature is extreme?"*
- *"Write a report with limitations and evidence links."*

### Narrate while it runs

- **"Watch the reasoning pane — the agent is thinking out loud before each
  tool call."**
- **"That `run_code` call executed in an isolated subprocess: single-threaded,
  10-minute timeout, and every line of stdout comes straight back to us."**
- **"See the leaderboard filling? Every run appends a row — the agent reads it
  back and decides what to try next. Nothing is hidden."**
- When a model needs a GPU: **"It's about to skip a transformer and tell us
  why. Honesty is a feature."**
- At the end: **"`report.md` just appeared in artifacts — that's the
  deliverable, and it's required to state its own limitations."**

### Expected output (verified)

~150 s run on the free tier, ending with a leaderboard like:

```
logreg   AUC 0.804 ± 0.040
rf       AUC 0.752 ± 0.041
histgb   AUC 0.772 ± 0.039
```

---

## 3. Option B — Headless smoke test

```powershell
cd E:\ai-safety
python -c @"
from bioworld.agent import build_agent
from bioworld.config import DEFAULT_DATASET
a = build_agent()
r = a.invoke({'messages':[{'role':'user','content':
    'Profile ' + str(DEFAULT_DATASET) + ', train 3 model families with run_code, then report the leaderboard.'}]})
print(r['messages'][-1].content)
"@
```

Expect: `profile_dataset` → `starter_script` → `run_code` → final message
summarizing the leaderboard. Wall time ≈ 2–3 minutes on free OpenRouter.

---

## 4. Switching providers live

```powershell
$env:BIOWORLD_PROVIDER = "gemini"      # needs GEMINI_API_KEY (free at aistudio.google.com/apikey)
$env:BIOWORLD_PROVIDER = "groq"        # fast; free tier ~8-12k TPM may 413 on the full loop
$env:BIOWORLD_PROVIDER = "ollama"      # fully local; needs a tool-capable model pulled
```

Also `BIOWORLD_MODEL` overrides the model id, `BIOWORLD_SUBAGENTS=1` re-enables
the three subagents (only sensible on large-context models), and
`BIOWORLD_TIMEOUT` tunes the sandbox wall clock.

---

## 5. Demo script (60–90 s spoken pitch)

1. **"This is BioWorld OS — executable biomedical intelligence on a laptop."**
2. **"One question in, and a deep agent runs the whole experiment loop:
   profile, train, compare, explain, simulate, cite, report."**
3. **"The constraint is the machine — i5, 4 GB, no GPU. The agent is *told* it
   can't do transformers and it *admits* it when it has to skip one."**
4. **"Every step is visible: reasoning, tool calls, live sandbox stdout, the
   append-only leaderboard, and the final Markdown report."**
5. **"It runs on free-tier models today — OpenRouter or Gemini — and fully
   local with Ollama tomorrow."**
6. **"This is the governance story too: honest uncertainty, bounded
   execution, nothing hidden — a first step toward an auditable AI research
   scientist."**

---

## 6. Troubleshooting the demo

| Symptom                        | Fix                                                            |
|--------------------------------|----------------------------------------------------------------|
| `OPENROUTER_API_KEY missing`   | add key to `bioworld/.env` (BOM-free)                          |
| OpenRouter 402 / credit error  | confirm `max_tokens=4096` is used (see `agent.build_model`)    |
| Groq 413 "Request too large"   | provider free TPM too low → use openrouter/gemini, or `BIOWORLD_SUBAGENTS=0` |
| Sandbox prints `[TIMEOUT]`     | raise `BIOWORLD_TIMEOUT`; agent sees partial stdout and retries |
| lightgbm access-violation      | expected in this env; starter uses HistGradientBoosting instead |
| `.env` keys "not loading"      | rewrite `.env` via Python (no BOM)                             |
| No report.md at end            | ask explicitly: *"write the report to report.md"*              |
