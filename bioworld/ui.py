"""Streaming bridge: turns the deep-agent LangGraph event stream into
structured UI events (tokens, tool calls, console output, final message)."""
from __future__ import annotations

import json
from typing import Iterator

from langchain_core.messages import ToolMessage


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                else:
                    out.append(json.dumps(block, default=str)[:200])
            else:
                out.append(str(block))
        return "".join(out)
    return str(content)


def stream_agent(agent, user_text: str, dataset_path: str | None = None,
                 thread_id: str = "bioworld") -> Iterator[tuple]:
    """Yield structured events for the UI.

    Event types:
      ("thinking", text)        — streamed assistant prose / reasoning
      ("tool_call", name, args)  — the agent decided to call a tool
      ("tool_result", name, txt) — tool returned; routed to console if run_code
      ("console", text)          — sandbox stdout/stderr
      ("final", text)            — the final assistant message
      ("done",)                  — stream finished
    """
    content_prefix = (f"Dataset: {dataset_path}\n\n" if dataset_path else "")
    payload = {"messages": [{"role": "user", "content": content_prefix + user_text}]}
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}

    final_text = ""
    for mode, chunk in agent.stream(payload, config=config,
                                   stream_mode=["messages", "updates"]):
        if mode == "messages":
            msg, _meta = chunk
            mtype = getattr(msg, "type", "")
            if mtype == "ai":
                # tool calls first
                tcs = getattr(msg, "tool_call_chunks", None) or getattr(msg, "tool_calls", None)
                if tcs:
                    for tc in tcs:
                        if isinstance(tc, dict):
                            yield ("tool_call", tc.get("name", ""), tc.get("args", ""))
                txt = _to_text(getattr(msg, "content", ""))
                if txt:
                    final_text = txt if not final_text else final_text
                    yield ("thinking", txt)
            elif mtype == "tool":
                name = getattr(msg, "name", "")
                result = _to_text(getattr(msg, "content", ""))
                yield ("tool_result", name, result)
                if name == "run_code":
                    yield ("console", result)
        elif mode == "updates":
            # Surfacing sub-agent / todo updates lightly.
            if isinstance(chunk, dict):
                for key, val in chunk.items():
                    if key == "tasks" and isinstance(val, dict):
                        pass
    yield ("final", final_text)
    yield ("done",)