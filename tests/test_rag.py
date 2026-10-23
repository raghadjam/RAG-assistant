"""Tests for src/rag.py: gate, generation, answer checking, repair, errors, logging."""

import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document  # noqa: E402
from src.config import Settings  # noqa: E402
from src.llm_client import (  # noqa: E402
    LLMAuthError,
    LLMContextTooLongError,
    LLMResponse,
    LLMUnavailableError,
)
from src.prompts import REFUSAL_SENTENCE  # noqa: E402
from src.rag import OOD_MESSAGE, UNVERIFIED_MESSAGE, RAGPipeline  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402
from tests.fakes import FakeLLM, HashEmbedder  # noqa: E402

PAPERS = {
    "d1": ("Attention", "transformer attention mechanism for language translation. " * 4),
    "d2": ("Vision", "convolutional networks for image classification and object detection. " * 4),
    "d3": ("RL", "reinforcement learning agents maximize reward in game environments. " * 4),
}
ATTENTION_PAPERS = {f"a{i}": (f"Doc {i}", f"attention mechanism variant {i} " * 8) for i in range(1, 5)}

GOOD = "The Transformer relies on attention mechanisms for language translation [1]."


def build_pipeline(replies, docs=PAPERS, **overrides):
    embedder = HashEmbedder()
    store = VectorStore(path=None, collection_name=f"t_{uuid.uuid4().hex}")
    chunks = []
    for doc_id, (title, text) in docs.items():
        chunks.extend(chunk_document(doc_id, text, {"title": title}, 300, 50))
    store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
    options = {"api_key": "k", "min_score": 0.3, "min_chunk_score": 0.2, "log_path": None, **overrides}
    settings = Settings(**options)
    llm = FakeLLM(replies)
    return RAGPipeline(embedder, store, llm, settings), llm


class TestAnswering:
    def test_grounded_answer(self):
        pipe, llm = build_pipeline([GOOD])
        result = pipe.ask("transformer attention")
        assert result.status == "answered"
        assert result.answer == GOOD
        assert result.check["verdict"] == "pass"
        assert result.sources[0]["n"] == 1 and result.sources[0]["title"] == "Attention"
        assert len(llm.calls) == 1

    def test_context_and_question_reach_the_llm(self):
        pipe, llm = build_pipeline([GOOD])
        pipe.ask("transformer attention")
        system, user = llm.calls[0]
        assert system["role"] == "system"
        assert "transformer attention mechanism" in user["content"]
        assert "<question>\ntransformer attention\n</question>" in user["content"]

    def test_model_refusal_becomes_no_answer(self):
        pipe, _ = build_pipeline([REFUSAL_SENTENCE])
        result = pipe.ask("transformer attention")
        assert result.status == "no_answer"
        assert result.answer == REFUSAL_SENTENCE

    def test_only_cited_sources_are_returned(self):
        pipe, _ = build_pipeline(["Attention mechanism variant 2 is described here [2]."], docs=ATTENTION_PAPERS)
        result = pipe.ask("attention mechanism variant")
        assert [s["n"] for s in result.sources] == [2]

    def test_llm_details_are_reported(self):
        pipe, _ = build_pipeline([GOOD])
        llm_info = pipe.ask("transformer attention").llm
        assert llm_info["model"] == "fake-model"
        assert llm_info["calls"] == 1 and llm_info["repairs"] == 0
        assert (llm_info["prompt_tokens"], llm_info["completion_tokens"]) == (10, 5)

    def test_truncated_answer_gets_a_warning_status(self):
        cut = LLMResponse(text=GOOD, model="m", truncated=True)
        pipe, _ = build_pipeline([cut])
        assert pipe.ask("transformer attention").status == "answered_with_warnings"

    def test_low_overlap_answer_gets_a_warning_status(self):
        pipe, _ = build_pipeline(["Quantum blockchain cryptography enables decentralized pizza delivery [1]."])
        result = pipe.ask("transformer attention")
        assert result.status == "answered_with_warnings"
        assert result.check["soft_failures"]


class TestOutOfDomain:
    def test_off_topic_question_never_calls_the_llm(self):
        pipe, llm = build_pipeline([GOOD])
        result = pipe.ask("best pizza recipe with fresh basil")
        assert result.status == "out_of_domain"
        assert result.answer == OOD_MESSAGE
        assert llm.calls == []
        assert result.timing["generation_ms"] == 0

    def test_off_topic_result_still_reports_retrieval_timing(self):
        pipe, _ = build_pipeline([])
        timing = pipe.ask("best pizza recipe with fresh basil").timing
        assert timing["retrieval_ms"] > 0
        assert timing["total_ms"] >= timing["retrieval_ms"]

    def test_empty_database_is_out_of_domain(self):
        pipe, llm = build_pipeline([GOOD], docs={})
        assert pipe.ask("transformer attention").status == "out_of_domain"
        assert llm.calls == []


class TestHallucinationChecks:
    def test_invented_number_triggers_a_repair_that_succeeds(self):
        pipe, llm = build_pipeline(["It reached 99.9 BLEU on translation [1].", GOOD])
        result = pipe.ask("transformer attention")
        assert result.status == "answered"
        assert result.answer == GOOD
        assert result.llm["calls"] == 2 and result.llm["repairs"] == 1
        second_call = llm.calls[1]
        assert second_call[-2] == {"role": "assistant", "content": "It reached 99.9 BLEU on translation [1]."}
        assert "99.9" in second_call[-1]["content"]

    def test_answer_that_stays_bad_is_rejected_not_shown(self):
        bad = "It reached 99.9 BLEU on translation [1]."
        pipe, llm = build_pipeline([bad, bad])
        result = pipe.ask("transformer attention")
        assert result.status == "rejected_ungrounded"
        assert result.answer == UNVERIFIED_MESSAGE
        assert result.draft_answer == bad
        assert len(llm.calls) == 2

    def test_repairs_can_be_switched_off(self):
        pipe, llm = build_pipeline(["No citations in this answer about attention mechanisms."], max_repairs=0)
        result = pipe.ask("transformer attention")
        assert result.status == "rejected_ungrounded"
        assert len(llm.calls) == 1

    def test_repair_that_turns_into_a_refusal_is_accepted(self):
        pipe, _ = build_pipeline(["It reached 99.9 BLEU [1].", REFUSAL_SENTENCE])
        assert pipe.ask("transformer attention").status == "no_answer"

    def test_failed_repair_call_still_rejects_the_draft(self):
        pipe, _ = build_pipeline(["It reached 99.9 BLEU [1].", LLMUnavailableError(["x"])])
        result = pipe.ask("transformer attention")
        assert result.status == "rejected_ungrounded"

    def test_citation_to_missing_source_is_rejected(self):
        pipe, _ = build_pipeline(["Attention mechanisms help translation [7]."] * 2)
        assert pipe.ask("transformer attention").status == "rejected_ungrounded"


class TestFailures:
    def test_llm_down_returns_sources_instead_of_crashing(self):
        pipe, _ = build_pipeline([LLMUnavailableError(["model-a: HTTP 503"])])
        result = pipe.ask("transformer attention")
        assert result.status == "llm_unavailable"
        assert result.sources and result.sources[0]["title"] == "Attention"
        assert "503" in result.error

    def test_bad_api_key_is_reported(self):
        pipe, _ = build_pipeline([LLMAuthError("invalid or missing API key")])
        result = pipe.ask("transformer attention")
        assert result.status == "llm_unavailable"
        assert "API key" in result.error

    def test_too_long_context_is_shrunk_and_retried(self):
        pipe, llm = build_pipeline(
            [LLMContextTooLongError("too long"), "Attention mechanism variant 1 is described [1]."],
            docs=ATTENTION_PAPERS, max_context_chars=600,
        )
        result = pipe.ask("attention mechanism variant")
        assert result.status == "answered"
        assert len(llm.calls) == 2
        assert len(llm.calls[1][1]["content"]) < len(llm.calls[0][1]["content"])

    def test_context_that_never_fits_ends_as_llm_unavailable(self):
        pipe, _ = build_pipeline([LLMContextTooLongError("too long")] * 3)
        assert pipe.ask("transformer attention").status == "llm_unavailable"

    @pytest.mark.parametrize("question", ["", "   ", None, "x" * 1001])
    def test_invalid_questions_are_rejected_without_work(self, question):
        pipe, llm = build_pipeline([GOOD])
        result = pipe.ask(question)
        assert result.status == "invalid_query"
        assert llm.calls == []


class TestLatencyLogging:
    def test_timing_fields_are_present_and_consistent(self):
        pipe, _ = build_pipeline([GOOD])
        t = pipe.ask("transformer attention").timing
        assert set(t) == {"embed_ms", "search_ms", "retrieval_ms", "generation_ms", "total_ms"}
        assert t["total_ms"] >= t["retrieval_ms"]
        assert t["total_ms"] >= t["generation_ms"]

    def test_each_question_is_logged_as_a_json_line(self, tmp_path):
        log = tmp_path / "logs" / "queries.jsonl"
        pipe, _ = build_pipeline([GOOD], log_path=str(log))
        pipe.ask("transformer attention")
        pipe.ask("best pizza recipe with fresh basil")
        lines = [json.loads(line) for line in log.read_text().splitlines()]
        assert [r["status"] for r in lines] == ["answered", "out_of_domain"]
        assert lines[0]["model"] == "fake-model"
        assert lines[0]["timing_ms"]["total_ms"] > 0
        assert lines[1]["model"] is None

    def test_unwritable_log_does_not_break_answering(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("not a directory")
        pipe, _ = build_pipeline([GOOD], log_path=str(blocker / "sub" / "log.jsonl"))
        assert pipe.ask("transformer attention").status == "answered"

    def test_result_converts_to_a_json_friendly_dict(self):
        pipe, _ = build_pipeline([GOOD])
        json.dumps(pipe.ask("transformer attention").to_dict())
