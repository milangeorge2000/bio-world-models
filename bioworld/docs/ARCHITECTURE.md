# BioWorld OS — Architecture

This document describes every component, its responsibilities, the key design
decisions, and the trade-offs that shape the POC.

## 0. Goals and non-goals

**Goals**
- Prove an *executable biomedical intelligence* loop on a normal laptop
  (i5, 4 GB RAM, no GPU).
- Let an LLM agent actually drive ML experiments, not just describe them.
- Make every step observable: reasoning, tool calls, sandbox stdout, artifacts.
- Stay provider-agnostic (free cloud, paid cloud, or fully local).
- Be honest about hardware limits and statistical uncertainty.

**Non-goals**
- Training large models — CPU-only by contract.
- Production MLOps (no distributed training, no model registry).
- Clinical-grade accuracy claims — the report is required to state limitations.

## 1. System context diagram

```
                 ┌──────────────────────────────┐
                 │        User / Researcher     │
                 └──────────────┬───────────────┘
                                │ question + dataset path
                                ▼
                 ┌──────────────────────────────┐
                 │       Streamlit UI           │  bioworld/app.py
                 │  (4 panes, live streaming)   │
                 └──────────────┬───────────────┘
                                │ ui.stream_agent()  ─ structured events
                                ▼
                 ┌──────────────────────────────┐
                 │   LangGraph StateGraph       │
                 │   (created by deepagents)    │  bioworld/agent.py
                 │    orchestrator node + tool  │
                 │    nodes; optional subagents │
                 └───────┬──────────────┬───────┘
                         │              │
              model calls│              │ LangChain tools
                         ▼              ▼
              ┌─────────────────┐  ┌────────────────────────────┐
              │  build_model()  │  │  tools.py                  │
              │  ChatOpenAI     │  │  profile_dataset  run_code │
              │  ChatGroq       │  │  starter_script  read_..   │
              │  Gemini         │  │  list_artifacts web_search │
              │  ChatOllama     │  └───────┬────────────────────┘
              └─────────────────┘          │
                                           ▼
                              ┌─────────────────────────────┐
                              │  sandbox.py (subprocess)    │
                              │  → workspace/run_NNN.py     │
                              │  → python run_NNN.py        │
                              │  → artifacts/ + stdout      │
                              └─────────────────────────────┘
```

## 2. Component breakdown

### 2.1 `config.py` — configuration and path safety
- Loads `bioworld/.env` via `python-dotenv`; exports the API keys.
- Resolves canonical paths: `PKG_DIR`, `PROJECT_DIR`, `WORKSPACE_DIR`,
  `ARTIFACTS_DIR`, `DATA_DIR`.
- Provider selection (`BIOWORLD_PROVIDER`) and per-provider default model
  (`BIOWORLD_MODEL`).
- `workspace_path(name)` — the single guard that refuses any path escaping the
  workspace (used by the sandbox's artifact reader).

### 2.2 `sandbox.py` — the hardware-honest execution boundary
The sandbox is the heart of the pitch. `run_code(code)`:
1. Writes the agent's code to `workspace/run_NNN.py` (counter-incremented),
   prepending a prologue that injects helpers:
   - `WORKSPACE` / `ARTIFACTS` absolute paths,
   - `save_metrics(d)` → `artifacts/metrics.json`,
   - `append_leaderboard(row)` → `artifacts/leaderboard.csv` (append + header).
2. Snapshots the workspace file set *before* the run.
3. Launches `[sys.executable, run_NNN.py]` as a subprocess with:
   - `cwd = workspace`,
   - a single-threaded BLAS environment (`OMP_NUM_THREADS=1`, etc.) so a 4 GB
     machine never explodes thread count,
   - `PYTHONPATH=PROJECT_DIR` (so scripts can `import bioworld.mltasks`),
   - `timeout = BIOWORLD_TIMEOUT` (default 600 s).
4. Captures stdout/stderr (text, utf-8), measures wall time, discovers *new*
   artifacts by diffing the file snapshot, reads any `metrics.json`, then
   deletes that per-run metrics file so the next run starts clean.
5. Truncates long outputs (`MAX_RETURN = 12000` chars) and returns a
   `RunResult` dataclass.

`read_artifact(name)` returns workspace file contents (resolves + escapes
check, truncation). `leaderboard()` parses the CSV. `clean()` strips ANSI codes.

**Why a subprocess and not `exec`?** A child process isolates crashes, memory,
and hangs from the agent's Python process; a 10-minute timeout kills runaway
training; stdout is captured byte-for-byte and shown in the UI.

### 2.3 `mltasks.py` — ML helpers
- `load_dataframe(path)` — loads ARFF (via `scipy.io.arff`) or CSV; decodes
  byte columns (the Debrecen label arrives as bytes `b'0'`/`b'1'`) to plain
  strings.
- `profile(path)` — returns rows, columns, dtypes, missing %, head rows,
  `describe()`, target-candidate detection (`class`/`label`/`target`/`outcome`,
  else last column), and the target distribution.
- `starter_train_code(path)` — a complete, working baseline script:
  LogisticRegression + RandomForest + HistGradientBoosting, 5-fold stratified
  CV, `roc_auc`, `append_leaderboard` rows, saves `best_model.joblib`,
  `save_metrics(...)`. Deliberately **does not** use lightgbm (crashes in this
  env with an access-violation OSError).

### 2.4 `tools.py` — the agent's tools
Each `@tool` returns a trimmed string so the model's context stays small.

| Tool                | What it does                                             | Output cap |
|---------------------|----------------------------------------------------------|-----------|
| `profile_dataset`   | `mltasks.profile`, slimmed to the fields the agent needs | 1500 chars|
| `run_code`          | `sandbox.run_code` + status line + stdout/stderr/artifacts/metrics | 3500 chars|
| `read_artifact`     | read a workspace file                                    | 12000     |
| `list_artifacts`    | recursive file listing of the workspace                  | —         |
| `starter_script`    | `starter_train_code(path)`                               | —         |
| `web_search`        | Tavily, top-4 results, truncated content                 | 4000      |

Tool groups also define what each subagent may touch:
`ORCHESTRATOR_TOOLS` (all six), `MODELING_TOOLS`, `DATA_QUALITY_TOOLS`,
`EXPLAINER_TOOLS`.

### 2.5 `agent.py` — the deep agent
- `SYSTEM_PROMPT` — an 8-step research loop (restate → profile → sandbox ML →
  hardware awareness → SHAP explain → what-if → evidence → report), plus
  honesty rules (wide CIs are a feature; never fake a GPU run).
- `build_model()` — provider dispatch:
  - `groq` → `ChatGroq(model, temperature=0)`
  - `gemini` → `ChatGoogleGenerativeAI(model, temperature=0)`
  - `openrouter` → `ChatOpenAI(model, temperature=0, max_tokens=4096,
    openai_api_key, base_url)` (max_tokens 4096 is important — 65535 exceeds
    some free-credit balances and OpenRouter rejects it)
  - `ollama` → `ChatOllama(model, temperature=0, num_ctx=8192)`
  - missing keys raise a clear `RuntimeError` with a fix hint.
- `_subagents()` — `data_quality`, `modeler`, `explainer` `SubAgent`s.
- `build_agent()` — `create_deep_agent(model, tools, system_prompt,
  subagents, name="bioworld", debug=False)`.

**Why subagents are off by default:** `deepagents` embeds each subagent's full
system prompt into the `task` tool description, which inflates the
orchestrator prompt to ~19–21k tokens. That fits OpenRouter/Gemini (131k–1M
context) but overflows Groq-free (8–12k TPM) and Ollama's 8k context. Set
`BIOWORLD_SUBAGENTS=1` when using a large-context model.

### 2.6 `ui.py` — stream → events bridge
`stream_agent(agent, user_text, dataset_path, thread_id)` streams with
`stream_mode=["messages", "updates"]` and yields normalized tuples:

```
("thinking",  text)          streamed assistant prose
("tool_call", name, args)    agent invoked a tool
("tool_result", name, txt)   tool returned
("console",   text)          run_code stdout/stderr
("final",     text)          final assistant message
("done",)
```

This keeps `app.py` decoupled from LangGraph/LangChain internals.

### 2.7 `app.py` — Streamlit UI
- Wide layout, title + caption, a research-question input and a dataset-path
  input (defaulted to the Debrecen dataset).
- On **▶ Run agent**: builds the agent, resets the chat buffer, opens two
  columns — left: live agent-activity box inside `st.status`; right: sandbox
  console (`st.code`) and leaderboard (`st.dataframe` from CSV).
- Below: an artifacts listing and, if `report.md` exists, the rendered report.
- Console buffer accumulates `run_code` outputs and shows them as one block at
  the end.

### 2.8 `mcp_server.py` — MCP surface
`FastMCP("bioworld-ml")` exposing `profile_dataset`, `run_code`,
`list_artifacts`, `read_artifact`, `starter_script`. Demonstrates that the same
tools are consumable over the Model Context Protocol (agents can load them via
`langchain-mcp-adapters`).

## 3. Key design decisions (and why)

| Decision | Rationale |
|----------|-----------|
| Subprocess sandbox | crash/memory/hang isolation; honest wall-time + RAM; byte-exact stdout |
| Single-threaded BLAS | keeps a 4 GB machine stable; CPU-only by contract |
| Workspace-escape check | every artifact read must stay under `workspace/` |
| Trimmed tool outputs | keeps deepagents' already-large prompt inside context budgets |
| Providers behind `build_model()` | one agent, four backends (free/paid/local) |
| max_tokens=4096 on OpenRouter | 65535 exceeds free-credit balance → request rejected |
| Starter uses HistGB not lightgbm | lightgbm raises access-violation in this numpy env |
| Subagents off by default | their prompts bloat the orchestrator past small-context models |
| BOM-free `.env` | PowerShell/Write-tool BOMs silently corrupt key loading |
| Honesty rules in system prompt | "SKIP ViT, say why"; wide CIs stated; n<1000 warning |

## 4. Failure modes & handling

- **Model context overflow** (Groq 413 / Ollama): switch provider to
  OpenRouter/Gemini, or disable subagents, or trim tool output.
- **Sandbox timeout**: `BIOWORLD_TIMEOUT` raises; stdout-so-far returned with a
  `[TIMEOUT ...]` banner so the agent can see partial progress.
- **Crash in sandbox**: returncode ≠ 0, stderr returned verbatim → agent
  debugs its own code (self-healing loop).
- **Missing API key**: `build_model()` raises a fixable `RuntimeError`.
- **Dataset not found**: UI blocks with an error; sandbox returns clear stderr.

## 5. Future / evolution

- Re-enable `BIOWORLD_SUBAGENTS=1` by default on large-context providers.
- Memory measurement (`psutil` peak RSS) instead of `"na"`.
- Model registry + versioned artifacts in the workspace.
- Persistent agent memory (checkpointer across sessions) — see the
  companion governance posts in `E:\ai-safety` for the memory-attack-surface
  discussion.
- Plug additional biomedical datasets (single-cell, EHR, imaging-derived
  features) through `profile_dataset`.
