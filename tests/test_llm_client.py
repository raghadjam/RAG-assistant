"""Tests for src/llm_client.py: retries, backoff, fallback and error handling."""

import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.llm_client import (  # noqa: E402
    LLMAuthError,
    LLMClient,
    LLMContextTooLongError,
    LLMCreditsError,
    LLMUnavailableError,
)
from tests.fakes import FakeResponse, FakeSession, err, ok  # noqa: E402

MESSAGES = [{"role": "user", "content": "hi"}]


def make_client(script, models=("model-a", "model-b"), max_retries=2, api_key="test-key"):
    settings = Settings(api_key=api_key, models=models, max_retries=max_retries)
    session = FakeSession(script)
    sleeps = []
    client = LLMClient(settings, session=session, sleep=sleeps.append, jitter=lambda a, b: 0)
    return client, session, sleeps


class TestSuccess:
    def test_returns_text_model_and_usage(self):
        client, session, sleeps = make_client([ok("Hello [1].", model="real-model")])
        reply = client.chat(MESSAGES)
        assert reply.text == "Hello [1]."
        assert reply.model == "real-model"
        assert (reply.prompt_tokens, reply.completion_tokens) == (12, 7)
        assert reply.attempts == 1
        assert reply.failures == []
        assert sleeps == []

    def test_request_is_built_correctly(self):
        client, session, _ = make_client([ok("x")])
        client.chat(MESSAGES, max_tokens=123)
        call = session.calls[0]
        assert call["url"].endswith("/chat/completions")
        assert call["headers"]["Authorization"] == "Bearer test-key"
        assert call["json"]["model"] == "model-a"
        assert call["json"]["max_tokens"] == 123
        assert call["json"]["messages"] == MESSAGES
        assert call["timeout"][1] == 60.0

    def test_think_tags_are_removed(self):
        client, _, _ = make_client([ok("<think>let me reason</think>The answer [1].")])
        assert client.chat(MESSAGES).text == "The answer [1]."

    def test_unclosed_think_tag_is_cut(self):
        client, _, _ = make_client([ok("Real answer. <think>and then it got cut off")])
        assert client.chat(MESSAGES).text == "Real answer."

    def test_truncated_flag_set_when_model_hits_token_limit(self):
        client, _, _ = make_client([ok("Cut off mid", finish="length")])
        assert client.chat(MESSAGES).truncated is True


class TestRetries:
    def test_rate_limit_then_success(self):
        client, session, sleeps = make_client([err(429), ok("fine")])
        reply = client.chat(MESSAGES)
        assert reply.text == "fine"
        assert reply.attempts == 2
        assert sleeps == [1.0]
        assert len(reply.failures) == 1

    def test_backoff_grows_exponentially(self):
        client, _, sleeps = make_client([err(503), err(503), ok("fine")])
        client.chat(MESSAGES)
        assert sleeps == [1.0, 2.0]

    def test_retry_after_header_is_honoured(self):
        client, _, sleeps = make_client([err(429, headers={"Retry-After": "7"}), ok("fine")])
        client.chat(MESSAGES)
        assert sleeps == [7.0]

    def test_retry_after_is_capped(self):
        client, _, sleeps = make_client([err(429, headers={"Retry-After": "500"}), ok("fine")])
        client.chat(MESSAGES)
        assert sleeps == [20.0]

    def test_timeout_is_retried(self):
        client, _, sleeps = make_client([requests.exceptions.Timeout(), ok("fine")])
        assert client.chat(MESSAGES).text == "fine"
        assert len(sleeps) == 1

    def test_connection_error_is_retried(self):
        client, _, _ = make_client([requests.exceptions.ConnectionError(), ok("fine")])
        assert client.chat(MESSAGES).text == "fine"

    def test_http_200_with_error_body_is_treated_as_error(self):
        body = FakeResponse(200, {"error": {"code": 429, "message": "rate limited upstream"}})
        client, _, sleeps = make_client([body, ok("fine")])
        assert client.chat(MESSAGES).text == "fine"
        assert len(sleeps) == 1

    def test_non_json_error_page_is_retried(self):
        html = FakeResponse(502, None, text="<html>Bad gateway</html>")
        client, _, _ = make_client([html, ok("fine")])
        assert client.chat(MESSAGES).text == "fine"

    def test_no_sleep_after_the_last_attempt(self):
        client, _, sleeps = make_client([err(500)], models=("only",), max_retries=0)
        with pytest.raises(LLMUnavailableError):
            client.chat(MESSAGES)
        assert sleeps == []


class TestFallback:
    def test_moves_to_next_model_after_retries_run_out(self):
        client, session, _ = make_client([err(429), err(429), ok("from b", model="model-b")], max_retries=1)
        reply = client.chat(MESSAGES)
        assert reply.text == "from b"
        assert [c["json"]["model"] for c in session.calls] == ["model-a", "model-a", "model-b"]
        assert reply.attempts == 3
        assert len(reply.failures) == 2

    def test_404_skips_model_without_retrying(self):
        client, session, sleeps = make_client([err(404, "model not found"), ok("from b")])
        assert client.chat(MESSAGES).text == "from b"
        assert [c["json"]["model"] for c in session.calls] == ["model-a", "model-b"]
        assert sleeps == []

    def test_403_skips_model(self):
        client, session, _ = make_client([err(403, "flagged"), ok("from b")])
        client.chat(MESSAGES)
        assert len(session.calls) == 2

    def test_400_that_is_not_about_length_skips_model(self):
        # e.g. a model that rejects system messages
        client, session, _ = make_client([err(400, "Developer instruction is not enabled"), ok("from b")])
        assert client.chat(MESSAGES).text == "from b"

    def test_empty_answer_skips_model(self):
        client, session, _ = make_client([ok(""), ok("from b")])
        assert client.chat(MESSAGES).text == "from b"
        assert len(session.calls) == 2

    def test_all_models_failing_raises_with_reasons(self):
        client, _, _ = make_client([err(404, "gone"), err(404, "gone too")])
        with pytest.raises(LLMUnavailableError) as info:
            client.chat(MESSAGES)
        assert len(info.value.reasons) == 2
        assert "model-a" in info.value.reasons[0]
        assert "HTTP 404" in info.value.reasons[0]


class TestFatalErrors:
    def test_missing_api_key(self):
        client, session, _ = make_client([], api_key=None)
        with pytest.raises(LLMAuthError, match="OPENROUTER_API_KEY"):
            client.chat(MESSAGES)
        assert session.calls == []

    def test_401_stops_immediately(self):
        client, session, sleeps = make_client([err(401, "bad key")])
        with pytest.raises(LLMAuthError):
            client.chat(MESSAGES)
        assert len(session.calls) == 1
        assert sleeps == []

    def test_402_stops_immediately(self):
        client, session, _ = make_client([err(402, "no credits")])
        with pytest.raises(LLMCreditsError):
            client.chat(MESSAGES)
        assert len(session.calls) == 1

    def test_context_too_long_is_reported_for_the_caller_to_handle(self):
        client, session, _ = make_client([err(400, "This model's maximum context length is 8192 tokens")])
        with pytest.raises(LLMContextTooLongError):
            client.chat(MESSAGES)
        assert len(session.calls) == 1
