"""LangChain tools the deep agent uses to drive the ML loop."""
from __future__ import annotations

import json

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from . import sandbox
from .config import TAVILY_API_KEY
from .mltasks import profile as _profile, starter_train_code


@tool
def profile_dataset(path: str) -> str:
    """Profile a dataset on disk. Returns rows, columns, dtypes, missing %,
    target distribution, and a head sample. Pass the dataset file path.
    Use this BEFORE writing training code to understand the shape of the data
    and to pick a target column."""
    try:
        info = _profile(path)
        # Keep only the fields the agent needs to pick a target & plan.
        slim = {
            "path": info["path"], "n_rows": info["n_rows"], "n_cols": info["n_cols"],
            "columns": info["columns"], "dtypes": info["dtypes"],
            "missing_pct": info["missing_pct"], "target": info["target"],
            "target_distribution": info["target_distribution"],
            "head": info["head"][:2],
        }
        return json.dumps(slim, indent=2, default=str)
    except Exception as e:  # pragma: no cover
        return f"[profile error] {e!r}"


@tool
def run_code(code: str) -> str:
    """Execute Python ML code in a CPU-only sandbox and return stdout, stderr,
    a list of new artifacts, and any metrics.json the code produced. Helper
    globals available in your script: `WORKSPACE`, `ARTIFACTS`, `save_metrics(d)`,
    `append_leaderboard(row)`.

    The sandbox is single-threaded, CPU-only with a 10-minute timeout, matching
    the POC hardware (i5, 4GB RAM). Prefer LightGBM / RandomForest /
    LogisticRegression / scikit-learn. Use 5-fold stratified CV for small n.
    Always print metrics and call append_leaderboard({model, auc_mean, auc_std,
    n_folds, cpu_only}) so results accumulate. Return ONLY the code to run."""
    r = sandbox.run_code(code)
    parts = [
        f"[ok={r.ok} rc={r.returncode} wall={r.wall_seconds:.1f}s artifacts={len(r.artifacts)}]",
    ]
    if r.stdout:
        parts.append("--- stdout ---\n" + r.stdout)
    if r.stderr:
        parts.append("--- stderr ---\n" + r.stderr)
    if r.artifacts:
        parts.append("--- new artifacts ---\n" + "\n".join(r.artifacts))
    if r.metrics:
        parts.append("--- metrics.json ---\n" + json.dumps(r.metrics, indent=2, default=str))
    return "\n\n".join(parts)


@tool
def read_artifact(name: str) -> str:
    """Read a file from the workspace (e.g., leaderboard.csv, a report md, a
    json). Use to inspect results a previous run_code produced."""
    return sandbox.read_artifact(name)


@tool
def list_artifacts() -> str:
    """List all files currently in the workspace/artifacts."""
    items = [str(p.relative_to(sandbox.WORKSPACE_DIR))
             for p in sandbox.WORKSPACE_DIR.rglob("*") if p.is_file()]
    return "\n".join(sorted(items)) if items else "[empty]"


@tool
def starter_script(path: str) -> str:
    """Return a working baseline training script for the dataset at `path`.
    Edit it (models, folds, features, calibration) then pass to run_code."""
    return starter_train_code(path)


def _tavily():
    return TavilySearch(max_results=4, topic="general")


@tool
def web_search(query: str) -> str:
    """Search the biomedical literature / web for evidence (Tavily). Use to
    link influential features or model choices to known pathways, papers, or
    clinical context before writing the final report."""
    if not TAVILY_API_KEY:
        return "[no TAVILY_API_KEY configured]"
    try:
        res = _tavily().invoke(query)
        if isinstance(res, dict):
            out = []
            if res.get("answer"):
                out.append("ANSWER: " + res["answer"])
            for r in res.get("results", [])[:4]:
                out.append(f"- {r.get('title','')}\n  {r.get('url','')}\n  {r.get('content','')[:300]}")
            return "\n".join(out)
        return str(res)
    except Exception as e:  # pragma: no cover
        return f"[search error] {e!r}"


# Tools grouped for the orchestrator vs subagents.
ORCHESTRATOR_TOOLS = [
    profile_dataset, run_code, read_artifact, list_artifacts,
    starter_script, web_search,
]
MODELING_TOOLS = [run_code, read_artifact, list_artifacts, starter_script]
DATA_QUALITY_TOOLS = [profile_dataset, read_artifact, list_artifacts]
EXPLAINER_TOOLS = [run_code, read_artifact, list_artifacts]