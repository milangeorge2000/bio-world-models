# BioWorld OS — Executable Biomedical Intelligence on a Laptop

A long-running **deep research agent** (LangChain `deepagents`) that answers a
biomedical question end-to-end: it profiles a dataset, trains CPU-only ML
models inside a **hardware-honest sandbox**, iterates a **leaderboard**,
explains with **SHAP**, runs **what-if simulations**, links evidence via
**web search (Tavily)**, and writes a **final Markdown report** — with every
step **streamed live** to a multi-pane Streamlit UI.

It is built to run on a modest machine (**Intel i5, 4 GB RAM, no GPU**), which
is the whole point: prove an *executable biomedical intelligence pipeline* that
runs anywhere, not a giant model.

> POC dataset: **Diabetic Retinopathy Debrecen** (UCI) — 1,151 rows, 114 KB,
> direct download. `data/debrecen/messidor_features.arff`.

---

## 1. Quick start

```bash
# from the project root (E:\ai-safety)
pip install -r requirements.txt     # see §Setup

streamlit run bioworld/app.py
```

Type a research question, keep the dataset path, click **▶ Run agent**.
Watch the agent think, call tools, train models, and stream output live.
See `docs/DEMO.md` for a full walkthrough.

### Headless (no UI)

```python
from bioworld.agent import build_agent
from bioworld.config import DEFAULT_DATASET

agent = build_agent()
reply = agent.invoke({
    "messages": [{"role": "user", "content":
        f"Profile {DEFAULT_DATASET}, train 3 model families with run_code, "
        "then report the leaderboard."}],
})["messages"][-1].content
print(reply)
```

### MCP surface

Expose the same ML tools over the Model Context Protocol:

```bash
python -m bioworld.mcp_server
```

---

## 2. Architecture (short)

```
┌────────────────────────── Streamlit UI (app.py) ──────────────────────────┐
│  Agent activity │ Sandbox console │ Leaderboard │ Artifacts & report      │
└──────────────▲───────────────────┬───────────────────────────────────────┘
               │ ui.stream_agent(...) yields structured events
┌──────────────┴───────────────────▼───────────────────────────────────────┐
│                        deep agent (agent.py)                              │
│   create_deep_agent(model, tools, system_prompt[, subagents])             │
│      │  tools: profile_dataset, run_code, read_artifact, list_artifacts,  │
│      │         starter_script, web_search  (tools.py)                     │
└──────┬──────────────────────────┬─────────────────────────────────────────┘
       │                          │
┌──────▼───────────────┐   ┌──────▼──────────────────────────────────────┐
│  build_model()       │   │   CPU-only sandbox (sandbox.py)              │
│  provider dispatch   │   │   subprocess + single-threaded BLAS env      │
│  openrouter|gemini|  │   │   10-min wall timeout, artifact discovery,   │
│  groq|ollama         │   │   metrics.json / leaderboard.csv capture     │
└──────────────────────┘   └──────────────────────────────────────────────┘
```

Full detail: **`docs/ARCHITECTURE.md`** · component map in §6.

---

## 3. Dataflow (short)

1. **User** → question (+ dataset path) → `stream_agent(agent, question, path)`.
2. **Agent** restates/analyzes, then calls `profile_dataset(path)` →
   schema, missing %, target distribution.
3. **`starter_script(path)`** → baseline training script → **`run_code(code)`**:
   the sandbox writes `run_NNN.py`, executes it in a subprocess, and returns
   stdout/stderr + new artifacts + `metrics.json`.
4. **Leaderboard** accumulates as the agent iterates model families
   (`append_leaderboard({model, auc_mean, auc_std, n_folds, cpu_only, ...})`).
5. **Winner** → SHAP explainer (`run_code`) → **what-if** perturbations.
6. **`web_search`** (Tavily) links top features to literature.
7. **Final `report.md`** written to `workspace/artifacts/` and streamed to the UI.

Full detail with a worked example: **`docs/DATAFLOW.md`**.

---

## 4. Setup

POC machine: **Windows 10, Intel i5, 4 GB RAM, Python 3.11 (Anaconda), CPU-only**.

```bash
# core
pip install langchain langchain-core langchain-openai langchain-groq
pip install langchain-google-genai langchain-ollama langchain-tavily
pip install deepagents langgraph streamlit python-dotenv mcp
pip install pandas numpy scikit-learn scipy joblib shap
pip install lightgbm xgboost pyarrow
```

> **Environment pinning (important).** This project pins `numpy==1.26.4`,
> `pandas==2.2.3` because `shap`/`deepagents`'s transitive `torch` import breaks
> under the stock Anaconda numpy. In practice we **uninstalled torch entirely**
> (a broken torch DLL blocked `deepagents` import); deepagents works without it.
> `lightgbm` can crash in this env (`access violation` OSError), so the starter
> baseline deliberately uses scikit-learn `HistGradientBoostingClassifier`.

### Environment variables (`bioworld/.env`)

Create `bioworld/.env` with your keys:

```
GROQ_API_KEY=...
TAVILY_API_KEY=...
OPENROUTER_API_KEY=...
GEMINI_API_KEY=            # optional, for the gemini provider
```

> **BOM gotcha.** `.env` must be written without a UTF-8 BOM. Write it with a
> small Python script (`open(p,'w',encoding='utf-8')`), not PowerShell
> `Set-Content` (which adds a BOM and silently corrupts key loading).

### Provider switch

| Env var                 | Default                | Values                               |
|-------------------------|------------------------|--------------------------------------|
| `BIOWORLD_PROVIDER`     | `openrouter`           | `openrouter` \| `gemini` \| `groq` \| `ollama` |
| `BIOWORLD_MODEL`        | per-provider default   | any model id the provider exposes    |
| `BIOWORLD_TIMEOUT`      | `600` (seconds)        | sandbox wall-clock timeout           |
| `BIOWORLD_SUBAGENTS`    | `0`                    | `1` to re-enable data_quality/modeler/explainer subagents |
| `OPENROUTER_BASE_URL`   | `https://openrouter.ai/api/v1` | custom gateway                |

Provider → default model:

| Provider     | Default model                  | Context | Why                                                  |
|--------------|--------------------------------|---------|------------------------------------------------------|
| `openrouter` | `openai/gpt-oss-20b:free`      | 131k    | Free, request-capped (not TPM-capped), tool-calling  |
| `gemini`     | `gemini-2.5-flash`             | 1M      | Very generous free tier (≈1M TPM)                    |
| `groq`       | `openai/gpt-oss-20b`           | —       | Fast, but free tier ≈8–12k TPM (deepagents prompt ~19–21k tokens overflows → 413) |
| `ollama`     | `llama3-groq-tool-use:8b`      | 8k      | Fully local/offline                                  |

Switch with a plain env var (Windows):

```powershell
$env:BIOWORLD_PROVIDER = "gemini"   # or openrouter | groq | ollama
```

---

## 5. Verified run

End-to-end **verified on the free `openai/gpt-oss-20b:free` (OpenRouter)**:

```
profile_dataset  →  starter_script  →  run_code (10s, 3 models, 5-fold CV)
→  leaderboard  →  final reply

leaderboard:
  logreg   AUC 0.804  ± 0.040
  histgb   AUC 0.772  ± 0.039
  rf       AUC 0.752  ± 0.041
```

The sandbox returned stdout, artifacts (`best_model.joblib`, `leaderboard.csv`),
and wall time for every run — nothing hidden.

---

## 6. Component map

| File                              | Role                                                            |
|-----------------------------------|-----------------------------------------------------------------|
| `bioworld/config.py`              | paths, `.env` loading, provider/model switch, workspace safety  |
| `bioworld/sandbox.py`             | CPU-only subprocess sandbox, artifact discovery, metrics capture|
| `bioworld/mltasks.py`             | dataset profiler + starter training-script generator            |
| `bioworld/tools.py`               | LangChain tools the agent calls                                 |
| `bioworld/agent.py`               | `create_deep_agent` + system prompt + sub-agent definitions     |
| `bioworld/mcp_server.py`          | FastMCP server exposing the same ML tools over MCP              |
| `bioworld/ui.py`                  | converts the LangGraph stream into UI events                    |
| `bioworld/app.py`                 | Streamlit multi-pane streaming app                              |
| `workspace/skills/biomedical-ml/SKILL.md` | the biomedical-ML skill (loop + honesty rules)           |
| `workspace/artifacts/`            | live output: `leaderboard.csv`, `best_model.joblib`, `report.md`|

---

## 7. Design principles

1. **Hardware honesty.** The sandbox pins single-threaded BLAS and a timeout;
   if the agent proposes a GPU/transformer model the system prompt tells it to
   SKIP and say why out loud. The machine is the constraint — and the pitch.
2. **Nothing hidden.** Every byte the sandbox prints returns to the agent *and*
   to the UI console. Model artifacts, metrics, and reports land in the
   workspace for inspection.
3. **Iterative, evidence-linked research.** Profiling → baseline → iterate →
   explain (SHAP) → what-if → literature search → honest report, in one loop.
4. **Provider-agnostic.** Four providers behind one `build_model()` switch, so
   the same agent runs free (OpenRouter/Gemini), fast (Groq), or fully local
   (Ollama).
5. **Governed by prompts, bounded by sandbox.** No GPU access, no internet from
   the sandbox, wall-clock-capped execution, workspace-escape checks on every
   artifact read.
