# BioWorld OS — Dataflow

This document traces a single question from the user to the final report, with
a worked example against the Debrecen diabetic-retinopathy dataset.

## 0. The artifacts that persist

Everything the agent produces lands under `bioworld/workspace/`:

```
workspace/
├── run_001.py          # every sandboxed script the agent ran (full history)
├── run_002.py
├── ...
├── skills/
│   └── biomedical-ml/SKILL.md
└── artifacts/
    ├── leaderboard.csv       # append-only model results
    ├── metrics.json          # per-run transient metrics (deleted after read)
    ├── best_model.joblib     # winning fitted model
    ├── shap_summary.png      # (agent-generated, optional)
    └── report.md             # final deliverable
```

## 1. Step-by-step dataflow

### Step 0 — User intent
The user submits a question (and optionally a dataset path) to the Streamlit
app. Default question (from `app.py`):

> "Using the Debrecen diabetic-retinopathy dataset, build a CPU-only predictor
> for signs of DR, compare 3 model families, explain the top features with
> SHAP, run 2 what-if simulations, and write a report."

### Step 1 — Stream bridge
`ui.stream_agent(agent, question, path)` wraps the question in a message
(`"Dataset: <path>\n\n<question>"`) and calls
`agent.stream(payload, config={thread_id}, stream_mode=["messages","updates"])`.
It normalizes LangGraph chunks into `("thinking"|"tool_call"|"tool_result"|
"console"|"final"|"done", ...)` events for the UI.

### Step 2 — Agent reasoning + first tool call
The deep agent reads `SYSTEM_PROMPT` (the 8-step loop) and emits thinking
streamed as `("thinking", ...)`. Its first decision is to **profile**:

```
tool_call: profile_dataset(path=E:\ai-safety\data\debrecen\messidor_features.arff)
```

`tools.profile_dataset` → `mltasks.profile`:
1. `load_dataframe` — `scipy.io.arff.loadarff` → pandas; byte columns
   (`b'0'`, `b'1'`) decoded to `"0"`/`"1"`.
2. Computes rows (1151), columns (19), dtypes, missing %, head, target
   candidates → `Class` (the last column), and distribution `{1: 611, 0: 540}`.
3. Slims to ~1500 chars and returns JSON.

The UI shows the tool call; the agent reads the profile and picks a target.

### Step 3 — Baseline script
The agent asks for a working starting point:

```
tool_call: starter_script(path=...messidor_features.arff)
→ starter_train_code() returns the full baseline script
```

### Step 4 — Sandboxed training (the loop)
The agent (possibly after editing) sends:

```
tool_call: run_code(code="...baseline...")
```

`tools.run_code` → `sandbox.run_code`:
1. Writes `workspace/run_001.py` = prologue + agent code.
2. Snapshots workspace files.
3. Runs `python run_001.py` in a subprocess with single-threaded BLAS env,
   `PYTHONPATH=project`, 600 s timeout.
4. The script prints `Loaded (1151, 19) target dist: {1: 611, 0: 540}`, trains
   logreg/rf/histgb with 5-fold stratified CV, appends 3 rows to
   `leaderboard.csv`, saves `best_model.joblib`, writes `metrics.json`.
5. Returns `RunResult`: `[ok=True rc=0 wall=9.2s artifacts=2]` + full stdout.
6. `metrics.json` is deleted so the next run's metrics are fresh.

UI shows the stdout in the **Sandbox console** pane; leaderboard refreshes.

### Step 5 — Iteration over the leaderboard
The agent tries other CPU-friendly variants (feature selection, calibration,
different folds/seed). Every `run_code` appends rows. The agent can
`read_artifact("leaderboard.csv")` or `list_artifacts()` to see cumulative
results. Verified end state:

```
model,auc_mean,auc_std,n_folds,cpu_only,peak_ram_mb
logreg,0.804,0.0396,5,True,na
rf,0.7517,0.0409,5,True,na
histgb,0.7716,0.0388,5,True,na
```

### Step 6 — Hardware honesty check
If the agent proposes a transformer/ViT/GPU model, the system prompt + skill
say: **SKIP it and state why** ("SKIPPED ViT: CPU-only, no GPU detected").
This surfaces in the streamed reasoning — visible, honest behavior.

### Step 7 — SHAP explanation
The winner (logreg) gets explained:

```
tool_call: run_code(code="...shap.TreeExplainer/LinearExplainer...
                      save shap_summary.png + importance table to ARTIFACTS...")
```

### Step 8 — What-if simulation
The agent perturbs 2–3 clinically meaningful features (e.g., glucose, blood
pressure, feature #2/#3) through the fitted model and prints risk deltas with
a confidence note.

### Step 9 — Evidence search
```
tool_call: web_search(query="top SHAP features ... diabetic retinopathy pathway")
→ Tavily returns titles/URLs/snippets (capped 4000 chars)
```
The agent links top features to literature before writing the report.

### Step 10 — Report
The agent writes `report.md` to `artifacts/`:

```
tool_call: run_code(code="...open(ARTIFACTS/report.md,'w').write('# BioWorld Report'...
                        ... question, data, methods, leaderboard, best model + CI,
                        SHAP findings, what-if, evidence links, limitations ...)")
```

The UI detects `report.md` and renders it under **Final report**; the final
assistant message streams into the activity pane.

### Step 11 — Done
`("done",)` ends the stream; `st.status` flips to "Agent finished".

## 2. What the UI shows, mapped to dataflow

| UI pane      | Data source                              | Event(s) feeding it                  |
|--------------|------------------------------------------|--------------------------------------|
| Agent activity | `agent.stream` AI chunks               | `thinking`, `tool_call`, `tool_result`, `final` |
| Sandbox console | `run_code` return strings             | `console`                            |
| Leaderboard  | `artifacts/leaderboard.csv`              | rendered at end of run               |
| Artifacts    | `workspace/` recursive file listing      | rendered at end of run               |
| Final report | `artifacts/report.md`                    | rendered at end of run               |

## 3. Worked example (the verified run)

On `openai/gpt-oss-20b:free` via OpenRouter (151 s wall, 9 stream events):

```
thinking      → restates the task
tool_call     → profile_dataset(messidor_features.arff)
tool_result   → 1151 rows, 19 cols, target Class {1:611, 0:540}
tool_call     → starter_script(...)
tool_result   → baseline source
tool_call     → run_code(baseline)
tool_result   → [ok=True rc=0 wall=9.2s artifacts=2] + stdout
               (3 models, 5-fold CV AUCs)
read_artifact → leaderboard.csv (found after list_artifacts found it)
final         → summary of the leaderboard
done          → 151.4 s, 9 events
```

Notice one interesting self-healing moment: the first `read_artifact(
"leaderboard.csv")` returned `[not found]` because the training run's file
listing hadn't surfaced it, so the agent called `list_artifacts()` to see what
existed, then read the leaderboard successfully. The loop is genuinely
agent-driven, not scripted.

## 4. Honesty & governance checkpoints

| Where                      | What is enforced                                        |
|----------------------------|---------------------------------------------------------|
| `SYSTEM_PROMPT` + SKILL.md | no GPU/transformer; SKIP out loud; wide CIs; n<1000 warning |
| `sandbox.py` env           | single-threaded BLAS; wall-time cap                     |
| `read_artifact`/`workspace_path` | workspace-escape guard                          |
| tool output trims          | context budget respected across providers               |
