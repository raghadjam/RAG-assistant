"""Check your LLM setup before running the pipeline.

    python scripts/check_llm.py               # key status + test each configured model
    python scripts/check_llm.py --list-free   # also list the free models available right now

Free model IDs change often. If a configured model shows NOT LISTED or FAIL,
pick one from the --list-free output and put it in OPENROUTER_MODELS in .env.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_settings  # noqa: E402
from src.llm_client import LLMClient, LLMError  # noqa: E402

SENSITIVE = ("key", "label", "token", "secret")


def free_models(models_json: dict) -> list[dict]:
    """Models whose prompt and completion price are both zero, biggest context first."""
    free = []
    for model in models_json.get("data", []):
        pricing = model.get("pricing") or {}
        try:
            is_free = float(pricing.get("prompt", 1)) == 0 and float(pricing.get("completion", 1)) == 0
        except (TypeError, ValueError):
            is_free = False
        if is_free:
            free.append(model)
    return sorted(free, key=lambda m: m.get("context_length") or 0, reverse=True)


def redact(data: dict) -> dict:
    return {k: ("***" if any(s in k.lower() for s in SENSITIVE) else v) for k, v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the LLM configuration.")
    parser.add_argument("--list-free", action="store_true", help="list currently free models")
    args = parser.parse_args()

    settings = load_settings()
    base = settings.base_url.rstrip("/")
    print(f"Endpoint : {base}")
    print(f"API key  : {'set' if settings.api_key else 'NOT SET (add OPENROUTER_API_KEY to .env)'}")
    print(f"Models   : {', '.join(settings.models)}\n")
    if not settings.api_key:
        return 1

    # key status and remaining free-tier requests (OpenRouter only)
    try:
        resp = requests.get(f"{base}/key", headers={"Authorization": f"Bearer {settings.api_key}"}, timeout=15)
        if resp.status_code == 200:
            print("Key status (OpenRouter /key):")
            for name, value in redact(resp.json().get("data", {})).items():
                print(f"  {name}: {value}")
        else:
            print(f"Key status: HTTP {resp.status_code} ({resp.text[:120]})")
    except requests.RequestException as exc:
        print(f"Key status: could not reach the API ({exc.__class__.__name__})")

    # live list of models
    live_ids: set[str] = set()
    try:
        resp = requests.get(f"{base}/models", timeout=20)
        resp.raise_for_status()
        models_json = resp.json()
        live_ids = {m["id"] for m in models_json.get("data", [])}
        if args.list_free:
            print("\nFree models available now (largest context first):")
            for m in free_models(models_json):
                print(f"  {m['id']:<60} context {m.get('context_length')}")
    except (requests.RequestException, ValueError) as exc:
        print(f"\nCould not fetch the model list ({exc.__class__.__name__}); skipping the listing check.")

    # send a tiny request to each configured model
    print("\nTesting each configured model with one tiny request:")
    failed = 0
    for model in settings.models:
        listed = "" if not live_ids or model in live_ids else "  (NOT LISTED on the API)"
        single = LLMClient(replace(settings, models=(model,), max_retries=0))
        try:
            reply = single.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=64)
            print(f"  OK    {model}  {reply.latency_ms:.0f} ms -> {reply.text[:30]!r}{listed}")
        except LLMError as exc:
            failed += 1
            print(f"  FAIL  {model}  {exc}{listed}")

    if failed == len(settings.models):
        print("\nNo configured model works. Run with --list-free and update OPENROUTER_MODELS in .env.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
