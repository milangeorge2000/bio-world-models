"""Configuration: paths, model, and environment loading."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PKG_DIR.parent
WORKSPACE_DIR = PKG_DIR / "workspace"
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"
DATA_DIR = PROJECT_DIR / "data"

load_dotenv(PKG_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

default_dataset = DATA_DIR / "debrecen" / "messidor_features.arff"
DEFAULT_DATASET = str(default_dataset) if default_dataset.exists() else None

# Provider switch. One of: openrouter | gemini | groq | ollama.
PROVIDER = os.getenv("BIOWORLD_PROVIDER", "openrouter").lower()

# Per-provider default models.
_DEFAULT_MODELS = {
    "openrouter": "openai/gpt-oss-20b:free",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-20b",
    "ollama": "llama3-groq-tool-use:8b",
}
MODEL_NAME = os.getenv("BIOWORLD_MODEL", _DEFAULT_MODELS.get(PROVIDER, "openai/gpt-oss-20b:free"))

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

for d in (WORKSPACE_DIR, ARTIFACTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def workspace_path(name: str) -> Path:
    p = (WORKSPACE_DIR / name).resolve()
    if not str(p).startswith(str(WORKSPACE_DIR)):
        raise ValueError(f"Escape from workspace: {name}")
    return p