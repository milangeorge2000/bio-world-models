"""BioWorld OS — Streamlit app. Live, multi-pane, streaming.

Run from the project root:
    streamlit run bioworld/app.py

Panes:
  • Agent activity  — streaming reasoning + tool calls
  • Sandbox console — live ML stdout/stderr
  • Leaderboard     — models seen so far (from artifacts/leaderboard.csv)
  • Artifacts & report
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioworld.config import WORKSPACE_DIR, ARTIFACTS_DIR, DEFAULT_DATASET, PKG_DIR  # noqa: E402
from bioworld.ui import stream_agent  # noqa: E402

st.set_page_config(page_title="BioWorld OS", layout="wide", page_icon="🧬")

st.title("🧬 BioWorld OS — Executable Biomedical Intelligence")
st.caption("A long-running deep agent. Profiles data → trains CPU-only models in a "
           "sandbox → iterates a leaderboard → explains with SHAP → runs what-if → "
           "reports. Everything streamed live.")

col_q, col_d = st.columns([3, 2])
with col_q:
    question = st.text_input(
        "Research question",
        value="Using the Debrecen diabetic-retinopathy dataset, build a CPU-only "
              "predictor for signs of DR, compare 3 model families, explain the top "
              "features with SHAP, run 2 what-if simulations, and write a report.",
    )
with col_d:
    dataset = st.text_input("Dataset path", value=DEFAULT_DATASET or "")

if st.button("▶ Run agent", type="primary"):
    if not dataset or not os.path.exists(dataset):
        st.error(f"Dataset not found: {dataset}")
        st.stop()
    from bioworld.agent import build_agent
    agent = build_agent()
    st.session_state.setdefault("chat", []).clear()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Agent activity")
        activity = st.container()
    with c2:
        st.subheader("Sandbox console")
        console = st.container()
        st.subheader("Leaderboard")
        lb_slot = st.container()

    console_buf: list[str] = []
    with activity:
        box = st.empty()
        with st.status("Agent running…", expanded=True) as status:
            acc = ""
            for ev in stream_agent(agent, question, dataset):
                kind = ev[0]
                if kind == "thinking":
                    acc += ev[1]
                    box.markdown(acc)
                elif kind == "tool_call":
                    args_s = ev[2]
                    if isinstance(args_s, (dict, list)):
                        import json as _json
                        args_s = _json.dumps(args_s, default=str)[:200]
                    arg_preview = str(args_s)[:160]
                    st.markdown(f"🔧 **{ev[1]}**(`{arg_preview}`)")
                    console_buf.append(f"$ {ev[1]}({arg_preview})")
                elif kind == "tool_result":
                    nm = ev[1]
                    head = str(ev[2])[:200]
                    if nm != "run_code":
                        st.markdown(f"↩ _{nm}_ → `{head}`")
                elif kind == "console":
                    console_buf.append(str(ev[2]))
                elif kind == "final":
                    if ev[1]:
                        box.markdown(acc + "\n\n---\n**Final:** " + ev[1][:200])
                elif kind == "done":
                    status.update(label="Agent finished", state="complete")

    with console:
        st.code("\n\n".join(console_buf) or "[no console output yet]", language="bash")

    with lb_slot:
        lb_path = ARTIFACTS_DIR / "leaderboard.csv"
        if lb_path.exists():
            import pandas as _pd
            try:
                df_lb = _pd.read_csv(lb_path)
                st.dataframe(df_lb, use_container_width=True, hide_index=True)
            except Exception:
                st.text(lb_path.read_text(encoding="utf-8"))
        else:
            st.write("_leaderboard will populate as the agent trains…_")

    st.subheader("Artifacts")
    items = sorted([str(p.relative_to(WORKSPACE_DIR))
                    for p in WORKSPACE_DIR.rglob("*") if p.is_file()])
    st.write("\n".join(items) if items else "_none yet_")

    report = ARTIFACTS_DIR / "report.md"
    if report.exists():
        st.subheader("Final report")
        st.markdown(report.read_text(encoding="utf-8"))