"""Chat client for OpenRouter (or any OpenAI-compatible API) with error handling.

What happens when a call fails:

    401 bad key              -> LLMAuthError, stop (retrying cannot help)
    402 no credits           -> LLMCreditsError, stop
    400 context too long     -> LLMContextTooLongError, the caller shrinks the prompt
    408 / 429 / 5xx / timeout / connection error
                             -> wait (exponential backoff, or Retry-After) and retry,
                                then move on to the next model in the list
    403 / 404 / other 4xx / empty answer
                             -> skip straight to the next model

If every model fails, LLMUnavailableError lists what went wrong with each.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field

import requests

from src.config import Settings

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

CONTEXT_ERROR_RE = re.compile(
    r"context (length|window)|maximum context|too many tokens|token limit|"
    r"reduce the length|exceeds? the (model'?s )?(max|maximum)",
    re.IGNORECASE,
)


class LLMError(Exception):
    """Base class for all LLM failures."""


class LLMAuthError(LLMError):
    pass


class LLMCreditsError(LLMError):
    pass


class LLMContextTooLongError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("all models failed: " + "; ".join(reasons))


class _Retry(Exception):
    def __init__(self, reason: str, retry_after: float | None = None):
        self.reason = reason
        self.retry_after = retry_after


class _SkipModel(Exception):
    def __init__(self, reason: str):
        self.reason = reason


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0       # time of the successful HTTP call only
    attempts: int = 1             # HTTP calls made, including failed ones
    truncated: bool = False       # True if the model hit max_tokens mid-answer
    failures: list[str] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Settings, session=None, sleep=time.sleep, jitter=random.uniform):
        self.settings = settings
        self.session = session or requests.Session()
        self._sleep = sleep      # injectable so tests do not really wait
        self._jitter = jitter

    def chat(self, messages: list[dict], max_tokens: int | None = None) -> LLMResponse:
        if not self.settings.api_key:
            raise LLMAuthError("OPENROUTER_API_KEY is not set. Add it to your .env file.")

        failures: list[str] = []
        attempts = 0
        for model in self.settings.models:
            for attempt in range(self.settings.max_retries + 1):
                attempts += 1
                try:
                    response = self._send(model, messages, max_tokens)
                except _Retry as exc:
                    failures.append(f"{model}: {exc.reason}")
                    if attempt == self.settings.max_retries:
                        break
                    delay = self._delay(attempt, exc.retry_after)
                    logger.warning("%s failed (%s); retrying in %.1fs", model, exc.reason, delay)
                    self._sleep(delay)
                except _SkipModel as exc:
                    failures.append(f"{model}: {exc.reason}")
                    logger.warning("%s skipped (%s)", model, exc.reason)
                    break
                else:
                    response.attempts = attempts
                    response.failures = failures
                    return response
        raise LLMUnavailableError(failures)

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.settings.backoff_max_s)
        delay = min(self.settings.backoff_base_s * 2 ** attempt, self.settings.backoff_max_s)
        return delay + self._jitter(0, delay * 0.25)

    def _send(self, model: str, messages: list[dict], max_tokens: int | None) -> LLMResponse:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": self.settings.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        url = self.settings.base_url.rstrip("/") + "/chat/completions"

        start = time.perf_counter()
        try:
            resp = self.session.post(url, json=payload, headers=headers,
                                     timeout=(10, self.settings.timeout_s))
        except requests.exceptions.Timeout:
            raise _Retry("timeout") from None
        except requests.exceptions.ConnectionError:
            raise _Retry("connection error") from None
        latency_ms = (time.perf_counter() - start) * 1000

        status = resp.status_code
        body = _json_or_none(resp)

        # OpenRouter can answer HTTP 200 with an error object instead of choices
        if status == 200 and isinstance(body, dict) and body.get("error") and not body.get("choices"):
            code = body["error"].get("code")
            status = int(code) if str(code).isdigit() else 502

        if status == 200:
            return _parse_success(body, model, latency_ms)

        message = _error_message(body, resp)
        if status == 401:
            raise LLMAuthError(f"invalid or missing API key ({message})")
        if status == 402:
            raise LLMCreditsError(f"insufficient credits or negative balance ({message})")
        if status == 400 and CONTEXT_ERROR_RE.search(message):
            raise LLMContextTooLongError(message)
        if status in RETRYABLE_STATUS:
            raise _Retry(f"HTTP {status}: {message}", retry_after=_retry_after(resp))
        raise _SkipModel(f"HTTP {status}: {message}")


def _json_or_none(resp):
    try:
        return resp.json()
    except ValueError:
        return None


def _error_message(body, resp) -> str:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("message", "unknown error"))[:200]
    return (getattr(resp, "text", "") or "no response body")[:200]


def _retry_after(resp) -> float | None:
    value = (getattr(resp, "headers", None) or {}).get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _strip_think(text: str) -> str:
    """Some reasoning models put their thinking in <think> tags; drop it."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.split("<think>")[0]   # an unclosed tag means the thinking was cut off


def _parse_success(body, model: str, latency_ms: float) -> LLMResponse:
    try:
        choice = body["choices"][0]
        content = (choice.get("message") or {}).get("content")
    except (KeyError, IndexError, TypeError):
        raise _SkipModel("unexpected response format") from None

    text = _strip_think(content or "").strip()
    if not text:
        raise _SkipModel("empty answer (a reasoning model may have used all its tokens)")

    usage = body.get("usage") or {}
    return LLMResponse(
        text=text,
        model=body.get("model") or model,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_ms=latency_ms,
        truncated=choice.get("finish_reason") == "length",
    )
