"""A FastMCP server exposing BioWorld's ML tools over the Model Context
Protocol. Run standalone: `python -m bioworld.mcp_server`.

The deep agent can ALSO load these same tools over MCP via
langchain-mcp-adapters — demonstrating the plugin/MCP ecosystem. In this POC
the tools are registered directly as LangChain tools as well (see tools.py),
but this server is the canonical, protocol-exposed surface.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .mltasks import profile as _profile, starter_train_code
from . import sandbox

mcp = FastMCP("bioworld-ml")


@mcp.tool()
def profile_dataset(path: str) -> str:
    """Profile a dataset on disk: rows, columns, dtypes, missing %, target."""
    import json
    try:
        return json.dumps(_profile(path), indent=2, default=str)[:6000]
    except Exception as e:
        return f"[error] {e!r}"


@mcp.tool()
def run_code(code: str) -> str:
    """Execute Python ML code in the CPU-only sandbox; return stdout/artifacts."""
    r = sandbox.run_code(code)
    return (
        f"[ok={r.ok} rc={r.returncode} wall={r.wall_seconds:.1f}s "
        f"artifacts={len(r.artifacts)}]\n\n--- stdout ---\n{r.stdout}\n\n"
        f"--- stderr ---\n{r.stderr}\n"
    )[-8000:]


@mcp.tool()
def list_artifacts() -> str:
    items = [str(p.relative_to(sandbox.WORKSPACE_DIR))
             for p in sandbox.WORKSPACE_DIR.rglob("*") if p.is_file()]
    return "\n".join(sorted(items)) if items else "[empty]"


@mcp.tool()
def read_artifact(name: str) -> str:
    return sandbox.read_artifact(name)


@mcp.tool()
def starter_script(path: str) -> str:
    return starter_train_code(path)


if __name__ == "__main__":
    mcp.run()