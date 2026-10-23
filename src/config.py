"""Settings for the RAG pipeline, read from environment variables or a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

# Free model IDs change often. Check them with scripts/check_llm.py --list-free.
# openrouter/free is a router that picks an available free model, so it is the last resort.
DEFAULT_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free",
)


@dataclass(frozen=True)
class Settings:
    # LLM connection
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    models: tuple[str, ...] = DEFAULT_MODELS
    timeout_s: float = 60.0
    max_retries: int = 3          # retries per model before moving to the next one
    backoff_base_s: float = 1.0
    backoff_max_s: float = 20.0
    max_tokens: int = 400
    temperature: float = 0.1

    # Retrieval and out-of-domain gate
    top_k: int = 5
    min_score: float = 0.45        # best chunk must score at least this, or the query is off-topic
    min_chunk_score: float = 0.35  # chunks below this are not shown to the LLM
    max_context_chars: int = 6000

    # Answer checking
    support_threshold: float = 0.6  # share of answer sentences that must be supported by the context
    max_repairs: int = 1            # extra LLM calls allowed to fix a failed answer

    log_path: str | None = "logs/rag_queries.jsonl"

    def __post_init__(self):
        if not self.models:
            raise ValueError("at least one model is required")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_retries < 0 or self.max_repairs < 0:
            raise ValueError("max_retries and max_repairs cannot be negative")
        if not 0 <= self.min_chunk_score <= self.min_score <= 1:
            raise ValueError("need 0 <= min_chunk_score <= min_score <= 1")
        if not 0 <= self.support_threshold <= 1:
            raise ValueError("support_threshold must be between 0 and 1")


def _number(env: Mapping[str, str], name: str, cast, default):
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from None


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from the environment (and .env). Pass env to override, e.g. in tests."""
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        env = os.environ

    defaults = Settings()
    models_raw = env.get("OPENROUTER_MODELS", "")
    models = tuple(m.strip() for m in models_raw.split(",") if m.strip()) or defaults.models

    return Settings(
        api_key=env.get("OPENROUTER_API_KEY") or None,
        base_url=env.get("OPENROUTER_BASE_URL") or defaults.base_url,
        models=models,
        timeout_s=_number(env, "LLM_TIMEOUT_S", float, defaults.timeout_s),
        max_retries=_number(env, "LLM_MAX_RETRIES", int, defaults.max_retries),
        max_tokens=_number(env, "LLM_MAX_TOKENS", int, defaults.max_tokens),
        temperature=_number(env, "LLM_TEMPERATURE", float, defaults.temperature),
        top_k=_number(env, "RAG_TOP_K", int, defaults.top_k),
        min_score=_number(env, "RAG_MIN_SCORE", float, defaults.min_score),
        min_chunk_score=_number(env, "RAG_MIN_CHUNK_SCORE", float, defaults.min_chunk_score),
        max_context_chars=_number(env, "RAG_MAX_CONTEXT_CHARS", int, defaults.max_context_chars),
        max_repairs=_number(env, "RAG_MAX_REPAIRS", int, defaults.max_repairs),
    )
