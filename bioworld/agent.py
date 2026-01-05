"""Build the BioWorld deep agent: a long-running research conductor with
sub-agents and skills, streaming every step."""
from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from deepagents import create_deep_agent
from deepagents.graph import SubAgent

from .config import (GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY,
                     OPENROUTER_BASE_URL, MODEL_NAME, PROVIDER, WORKSPACE_DIR)
from .tools import ORCHESTRATOR_TOOLS, EXPLAINER_TOOLS, DATA_QUALITY_TOOLS, MODELING_TOOLS


SYSTEM_PROMPT = """You are BioWorld OS — the lead biomedical-intelligence research agent.
You run on a CPU-only laptop (Intel i5, 4GB RAM, no GPU). That constraint is the
POINT of this POC: prove the executable pipeline, not a giant model.

For every request you:
1. RESTATE & ANALYZE the question — say what is being asked and what an answer
   would require (data, model family, validation, what-if). Think out loud.
2. PROFILE the dataset with `profile_dataset`. Decide the task type
   (classification / survival / regression) and the target column.
3. SANDBOX-DRIVEN ML — call `starter_script` for a working baseline, then EDIT
   and RUN it via `run_code`. Iterate: try a few CPU-friendly model families
   (LogisticRegression, RandomForest, LightGBM, XGBoost), 5-fold stratified CV,
   with/without feature scaling, calibration for small n. For EVERY candidate,
   call `append_leaderboard({model, auc_mean, auc_std, n_folds, cpu_only,
   peak_ram_mb})`. Report peak RAM and wall-clock when you can.
4. HARDCWARE AWARENESS — if a model would need a GPU/transformer, SKIP it and
   say why out loud ("SKIPPED ViT: CPU-only, no GPU detected"). Never pretend.
5. EXPLAIN — once a winner exists, run a SHAP explainer script via `run_code`
   (TreeExplainer for tree models). Save a summary to artifacts.
6. WHAT-IF SIMULATION — pick 2-3 clinically meaningful feature changes and run
   them through the trained model; report the risk delta with a confidence note.
7. EVIDENCE — call `web_search` to link the top SHAP features to diabetic
   retinopathy literature / pathways before writing the report.
8. REPORT — write a final Markdown report to artifacts using run_code: question,
   data, methods, leaderboard, best model + honest CI, SHAP findings, what-if,
   evidence links, limitations (small n, no external validation). End by
   printing the report path.

Delegate when useful: hand dataset QA to the `data_quality` sub-agent, model
iteration to the `modeler` sub-agent, and explanation to the `explainer`
sub-agent. Always log your reasoning before each tool call. Be honest about
uncertainty — wide CIs and a "statistical power" warning are a FEATURE here.
Be concise in prose but thorough in execution."""


def build_model() -> BaseChatModel:
    """Build the chat model from the configured provider.

    Provider is chosen via BIOWORLD_PROVIDER env var (default openrouter).
    Models default per provider but are overridable via BIOWORLD_MODEL.
    Free-tier-friendly defaults: OpenRouter (no TPM cap, 20 RPM) -> Gemini.
    """
    provider = PROVIDER
    if provider == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing. Put it in bioworld/.env")
        return ChatGroq(model=MODEL_NAME, temperature=0)
    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY missing. Get one free at "
                               "https://aistudio.google.com/apikey and add to .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY missing. Put it in bioworld/.env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=4096,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192)
    raise RuntimeError(f"Unknown BIOWORLD_PROVIDER: {provider!r}. "
                       f"Use one of groq|gemini|openrouter|ollama.")


def _subagents() -> list[SubAgent]:
    return [
        {
            "name": "data_quality",
            "description": "Profile a dataset and report schema, missing data, "
                           "class imbalance, target leakage risk, and outliers. "
                           "Read-only: does not train.",
            "system_prompt": "You inspect biomedical datasets for quality issues "
                             "and report findings concisely. Use profile_dataset. "
                             "Return a short markdown summary: schema, target, "
                             "missingness, imbalance, leakage risks, recommended "
                             "preprocessing. Never train models.",
            "tools": DATA_QUALITY_TOOLS,
        },
        {
            "name": "modeler",
            "description": "Iteratively train and evaluate CPU-friendly ML "
                           "models in the sandbox, logging each to the leaderboard.",
            "system_prompt": "You are a CPU-only ML engineer. Use run_code to "
                             "train LightGBM/RandomForest/LogisticRegression "
                             "with 5-fold CV. Always append_leaderboard rows. "
                             "Report AUC mean +/- std and peak RAM. Do not use "
                             "GPUs/transformers. Be terse; print metrics.",
            "tools": MODELING_TOOLS,
        },
        {
            "name": "explainer",
            "description": "Run SHAP and what-if simulations on a trained "
                           "model artifact and return explanations.",
            "system_prompt": "You explain a trained model. Use run_code to run "
                             "SHAP (TreeExplainer for trees) and what-if feature "
                             "perturbations. Report feature importances, a "
                             "counterfactual, and confidence notes. Be concise.",
            "tools": EXPLAINER_TOOLS,
        },
    ]


def build_agent():
    """Create the streaming deep agent.

    Subagents are disabled by default: deepagents embeds each sub-agent's full
    system prompt into the `task` tool description, which balloons the
    orchestrator prompt past Groq-free / Ollama-8k context budgets. Set
    BIOWORLD_SUBAGENTS=1 to re-enable the data_quality / modeler / explainer
    sub-agents (recommended only with a large-context model).
    """
    model = build_model()
    subs = _subagents() if os.getenv("BIOWORLD_SUBAGENTS", "0") == "1" else []
    return create_deep_agent(
        model=model,
        tools=ORCHESTRATOR_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        subagents=subs,
        name="bioworld",
        debug=False,
    )