"""End-to-end RAG: question -> retrieve -> gate -> LLM -> check -> result."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.config import Settings
from src.guardrails import AnswerCheck, check_answer, gate_query
from src.llm_client import LLMContextTooLongError, LLMError, LLMResponse
from src.prompts import build_messages, repair_message
from src.search import semantic_search

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000

INVALID_MESSAGE = "Please ask a question (up to 1,000 characters)."
OOD_MESSAGE = (
    "That question doesn't look related to the document collection (arXiv papers on AI, "
    "machine learning and statistics), so I can't answer it from these sources."
)
UNVERIFIED_MESSAGE = (
    "I couldn't produce an answer that I can verify against the retrieved documents, "
    "so I'm not showing one. Try rephrasing the question."
)
LLM_DOWN_MESSAGE = (
    "The language model is unavailable right now, so I can't write an answer. "
    "The most relevant documents I found are listed under sources."
)

# status values:
#   answered, answered_with_warnings  an answer is shown
#   no_answer                         the model said the documents do not contain the answer
#   out_of_domain                     the question is off-topic, the LLM was not called
#   rejected_ungrounded               the answer failed checking and was not shown
#   llm_unavailable                   the LLM could not be reached
#   invalid_query                     empty or too long


@dataclass
class RAGResult:
    question: str
    status: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    top_score: float | None = None
    timing: dict = field(default_factory=dict)
    llm: dict | None = None
    check: dict | None = None
    draft_answer: str | None = None    # the rejected answer, kept for debugging
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _source(n: int, hit: dict) -> dict:
    meta = hit.get("metadata") or {}
    return {
        "n": n,
        "doc_id": meta.get("doc_id"),
        "title": meta.get("title"),
        "chunk_index": meta.get("chunk_index"),
        "score": round(hit["score"], 3),
    }


def _check_summary(check: AnswerCheck) -> dict:
    return {
        "verdict": check.verdict,
        "citations": check.citations,
        "invalid_citations": check.invalid_citations,
        "unsupported_numbers": check.unsupported_numbers,
        "support_ratio": round(check.support_ratio, 2),
        "hard_failures": check.hard_failures,
        "soft_failures": check.soft_failures,
    }


class RAGPipeline:
    def __init__(self, embedder, store, llm, settings: Settings):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.settings = settings

    def ask(self, question: str) -> RAGResult:
        started = time.perf_counter()
        question = (question or "").strip()

        if not question or len(question) > MAX_QUESTION_CHARS:
            result = RAGResult(question[:80], "invalid_query", INVALID_MESSAGE)
            return self._finish(result, started)

        search = semantic_search(question, self.embedder, self.store, k=self.settings.top_k)
        hits = search["results"]
        timing = {
            "embed_ms": search["timing"]["embed_ms"],
            "search_ms": search["timing"]["search_ms"],
            "retrieval_ms": search["timing"]["total_ms"],
            "generation_ms": 0.0,
        }

        gate = gate_query(hits, self.settings.min_score)
        if not gate.in_domain:
            result = RAGResult(question, "out_of_domain", OOD_MESSAGE, top_score=gate.top_score,
                               timing=timing, error=gate.reason)
            return self._finish(result, started)

        context_hits = [h for h in hits if h["score"] >= self.settings.min_chunk_score]
        gen_start = time.perf_counter()
        try:
            result = self._generate_and_check(question, context_hits, gate.top_score, timing)
        except LLMError as exc:
            logger.error("LLM failed for %r: %s", question[:60], exc)
            result = RAGResult(
                question, "llm_unavailable", LLM_DOWN_MESSAGE,
                sources=[_source(i + 1, h) for i, h in enumerate(context_hits)],
                top_score=gate.top_score, timing=timing, error=str(exc),
            )
        result.timing["generation_ms"] = (time.perf_counter() - gen_start) * 1000
        return self._finish(result, started)

    # ------------------------------------------------------------------ #
    def _generate_and_check(self, question, context_hits, top_score, timing) -> RAGResult:
        response = self._chat_shrinking(question, context_hits)
        messages, used = response["messages"], response["used"]
        reply: LLMResponse = response["reply"]

        draft = reply.text
        calls, attempts = 1, reply.attempts
        tokens_in, tokens_out = reply.prompt_tokens or 0, reply.completion_tokens or 0
        check = check_answer(draft, used, self.settings.support_threshold)

        repairs = 0
        while check.hard_failures and repairs < self.settings.max_repairs:
            repairs += 1
            followup = messages + [
                {"role": "assistant", "content": draft},
                {"role": "user", "content": repair_message(check.hard_failures, len(used))},
            ]
            try:
                reply = self.llm.chat(followup)
            except LLMError as exc:
                logger.warning("repair call failed: %s", exc)
                break
            draft = reply.text
            calls += 1
            attempts += reply.attempts
            tokens_in += reply.prompt_tokens or 0
            tokens_out += reply.completion_tokens or 0
            check = check_answer(draft, used, self.settings.support_threshold)

        llm_info = {
            "model": reply.model,
            "calls": calls,
            "http_attempts": attempts,
            "repairs": repairs,
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "truncated": reply.truncated,
            "fallbacks": reply.failures,
        }

        if check.is_refusal:
            status, answer, draft_answer = "no_answer", draft, None
        elif check.hard_failures:
            status, answer, draft_answer = "rejected_ungrounded", UNVERIFIED_MESSAGE, draft
        elif check.soft_failures or reply.truncated:
            status, answer, draft_answer = "answered_with_warnings", draft, None
        else:
            status, answer, draft_answer = "answered", draft, None

        cited = sorted({c for c in check.citations if 1 <= c <= len(used)})
        numbers = cited or list(range(1, len(used) + 1))
        return RAGResult(
            question, status, answer,
            sources=[_source(n, used[n - 1]) for n in numbers],
            top_score=top_score, timing=timing, llm=llm_info,
            check=_check_summary(check), draft_answer=draft_answer,
        )

    def _chat_shrinking(self, question, hits) -> dict:
        """Call the LLM; if the prompt is too long, halve the context and try again."""
        max_chars = self.settings.max_context_chars
        for _ in range(3):
            messages, used = build_messages(question, hits, max_chars)
            try:
                reply = self.llm.chat(messages)
                return {"reply": reply, "messages": messages, "used": used}
            except LLMContextTooLongError:
                logger.warning("context too long at %d chars, shrinking", max_chars)
                max_chars //= 2
        raise LLMContextTooLongError("prompt is still too long after shrinking the context")

    # ------------------------------------------------------------------ #
    def _finish(self, result: RAGResult, started: float) -> RAGResult:
        result.timing["total_ms"] = (time.perf_counter() - started) * 1000
        for key in ("embed_ms", "search_ms", "retrieval_ms", "generation_ms", "total_ms"):
            result.timing.setdefault(key, 0.0)
            result.timing[key] = round(result.timing[key], 1)
        logger.info("status=%s total=%.0fms retrieval=%.0fms generation=%.0fms",
                    result.status, result.timing["total_ms"],
                    result.timing["retrieval_ms"], result.timing["generation_ms"])
        self._log(result)
        return result

    def _log(self, result: RAGResult) -> None:
        if not self.settings.log_path:
            return
        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": result.question,
            "status": result.status,
            "top_score": result.top_score,
            "timing_ms": result.timing,
            "model": (result.llm or {}).get("model"),
            "llm_calls": (result.llm or {}).get("calls"),
            "tokens": ((result.llm or {}).get("prompt_tokens"), (result.llm or {}).get("completion_tokens")),
            "check": (result.check or {}).get("verdict"),
            "error": result.error,
        }
        try:
            path = Path(self.settings.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("could not write query log: %s", exc)
