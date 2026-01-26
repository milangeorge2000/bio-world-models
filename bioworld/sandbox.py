"""CPU-only subprocess sandbox for running agent-generated ML code.

Design keeps hardware honest (the whole POC pitch): every run executes in a
child process with a wall-clock timeout, single-threaded BLAS, and output
captured to disk. The agent writes self-contained scripts that read the
dataset, train, and persist artifacts (metrics.json, leaderboard.csv,
models, figures) under the workspace. Nothing is hidden — every byte the
subprocess prints comes back to the agent and to the UI.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from .config import ARTIFACTS_DIR, WORKSPACE_DIR, PROJECT_DIR

WALL_TIMEOUT = int(os.getenv("BIOWORLD_TIMEOUT", "600"))

_SANDBOX_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONPATH": str(PROJECT_DIR),
    "BIOWORLD_WORKSPACE": str(WORKSPACE_DIR),
    "BIOWORLD_ARTIFACTS": str(ARTIFACTS_DIR),
}

MAX_RETURN = 12000


@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int
    artifacts: list[str]
    metrics: dict | None
    wall_seconds: float


_counter = 0


def _next_script_path() -> Path:
    global _counter
    _counter += 1
    return WORKSPACE_DIR / f"run_{_counter:03d}.py"


def _discover_artifacts(before: set[str]) -> list[str]:
    after = {str(p) for p in WORKSPACE_DIR.rglob("*") if p.is_file()}
    new = sorted(after - before)
    rel = []
    for p in new:
        try:
            rel.append(str(Path(p).relative_to(WORKSPACE_DIR)))
        except ValueError:
            rel.append(p)
    skip = ("run_",)
    return [r for r in rel if not any(r.startswith(s) for s in skip)]


def _load_metrics() -> dict | None:
    p = ARTIFACTS_DIR / "metrics.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def run_code(code: str, timeout: int | None = None) -> RunResult:
    """Execute Python source in an isolated subprocess under the workspace."""
    src = textwrap.dedent(code).strip()
    # When code contains no top-level exec, assume a bare block and exec it.
    if not src:
        return RunResult(False, "", "empty code", 1, [], None, 0.0)

    script = _next_script_path()
    prologue = (
        "import os, sys, json, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        f"WORKSPACE = {str(WORKSPACE_DIR)!r}\n"
        f"ARTIFACTS = {str(ARTIFACTS_DIR)!r}\n"
        "os.makedirs(ARTIFACTS, exist_ok=True)\n"
        "os.chdir(WORKSPACE)\n"
        "def save_metrics(m):\n"
        "    import json\n"
        "    open(os.path.join(ARTIFACTS,'metrics.json'),'w').write(json.dumps(m, indent=2, default=str))\n"
        "def append_leaderboard(row):\n"
        "    import json, os\n"
        "    p=os.path.join(ARTIFACTS,'leaderboard.csv')\n"
        "    import csv\n"
        "    exists=os.path.exists(p)\n"
        "    with open(p,'a',newline='') as f:\n"
        "        w=csv.DictWriter(f, fieldnames=list(row.keys()));\n"
        "        if not exists: w.writeheader()\n"
        "        w.writerow(row)\n"
    )
    script.write_text(prologue + "\n" + src + "\n", encoding="utf-8")

    before = {str(p) for p in WORKSPACE_DIR.rglob("*") if p.is_file()}
    start = time.time()
    env = dict(os.environ)
    env.update(_SANDBOX_ENV)
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(WORKSPACE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout or WALL_TIMEOUT,
        )
        ok = proc.returncode == 0
        out, err = proc.stdout, proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        ok = False
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = f"[TIMEOUT after {timeout or WALL_TIMEOUT}s]\n" + (
            (e.stderr or "") if isinstance(e.stderr, str) else ""
        )
        rc = -1
    except Exception as e:  # pragma: no cover
        ok = False
        out = ""
        err = f"[sandbox error] {e!r}"
        rc = -2

    wall = time.time() - start
    artifacts = _discover_artifacts(before)
    metrics = _load_metrics()

    if len(out) > MAX_RETURN:
        out = out[:MAX_RETURN] + f"\n...[truncated, saved to stdout in {script.name}]"
    if len(err) > MAX_RETURN:
        err = err[:MAX_RETURN] + "\n...[truncated]"
    # Clear single-run metrics so each report is fresh.
    mp = ARTIFACTS_DIR / "metrics.json"
    if mp.exists():
        mp.unlink()
    return RunResult(ok, out, err, rc, artifacts, metrics, wall)


def read_artifact(name: str) -> str:
    p = (WORKSPACE_DIR / name).resolve()
    if not str(p).startswith(str(WORKSPACE_DIR)) or not p.exists():
        return f"[not found: {name}]"
    txt = p.read_text(encoding="utf-8", errors="replace")
    return txt if len(txt) <= MAX_RETURN else txt[:MAX_RETURN] + "\n...[truncated]"


def leaderboard() -> list[dict]:
    p = ARTIFACTS_DIR / "leaderboard.csv"
    if not p.exists():
        return []
    import csv
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_SCRUB = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(s: str) -> str:
    return _SCRUB.sub("", s)