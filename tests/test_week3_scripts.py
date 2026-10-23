"""Tests for scripts/check_llm.py, scripts/calibrate_threshold.py and scripts/evaluate_rag.py."""

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_threshold  # noqa: E402
import check_llm  # noqa: E402
import evaluate_rag  # noqa: E402

from src.chunking import chunk_document  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402
from tests.fakes import HashEmbedder  # noqa: E402
from tests.test_rag import GOOD, PAPERS, build_pipeline  # noqa: E402


class TestCheckLlm:
    def test_free_models_are_filtered_and_sorted_by_context(self):
        listing = {"data": [
            {"id": "paid/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}, "context_length": 999},
            {"id": "free/small:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 32000},
            {"id": "free/big:free", "pricing": {"prompt": "0", "completion": "0"}, "context_length": 1000000},
            {"id": "half/free", "pricing": {"prompt": "0", "completion": "0.5"}, "context_length": 5},
            {"id": "no/pricing"},
            {"id": "bad/pricing", "pricing": {"prompt": "n/a", "completion": "n/a"}},
        ]}
        assert [m["id"] for m in check_llm.free_models(listing)] == ["free/big:free", "free/small:free"]

    def test_empty_listing(self):
        assert check_llm.free_models({}) == []

    def test_sensitive_fields_are_hidden(self):
        shown = check_llm.redact({"label": "sk-or-abc", "api_key": "x", "usage": 3, "limit_remaining": 10})
        assert shown == {"label": "***", "api_key": "***", "usage": 3, "limit_remaining": 10}


class TestCalibration:
    def make_store(self):
        embedder = HashEmbedder()
        store = VectorStore(path=None, collection_name=f"cal_{uuid.uuid4().hex}")
        chunks = []
        for doc_id, (title, text) in PAPERS.items():
            chunks.extend(chunk_document(doc_id, text, {"title": title}, 300, 50))
        store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
        return embedder, store

    def test_separable_queries_give_a_clean_gap_and_sensible_cutoff(self):
        embedder, store = self.make_store()
        report = calibrate_threshold.calibrate(
            embedder, store,
            in_queries=["transformer attention", "image classification"],
            off_queries=["best pizza recipe", "qzxv blorf wibble"],
        )
        assert report["clean_gap"] is True
        assert report["on_topic"]["min"] > report["off_topic"]["max"]
        assert report["off_topic"]["max"] < report["recommended_threshold"] < report["on_topic"]["min"]
        assert len(report["sweep"]) == 13

    def test_sweep_counts_mistakes(self):
        rows = calibrate_threshold.sweep([0.9, 0.5], [0.4, 0.1], [0.45, 0.95])
        low, high = rows
        assert (low["wrongly_refused"], low["wrongly_accepted"]) == (0, 0)
        assert low["balanced_accuracy"] == 1.0
        assert (high["wrongly_refused"], high["wrongly_accepted"]) == (2, 0)
        assert high["balanced_accuracy"] == 0.5

    def test_recommend_takes_the_middle_of_tied_best_cutoffs(self):
        rows = [{"threshold": t, "balanced_accuracy": a}
                for t, a in [(0.2, 0.9), (0.3, 1.0), (0.4, 1.0), (0.5, 1.0), (0.6, 0.8)]]
        assert calibrate_threshold.recommend(rows) == 0.4

    def test_markdown_report_mentions_overlap_when_groups_overlap(self):
        embedder, store = self.make_store()
        report = calibrate_threshold.calibrate(
            embedder, store, in_queries=["pizza"], off_queries=["transformer attention"])
        assert "overlap" in calibrate_threshold.markdown_report(report)


class TestEvaluation:
    QUESTIONS = [
        ("answerable", "transformer attention", evaluate_rag.ANSWERED),
        ("off-topic", "best pizza recipe with fresh basil", {"out_of_domain"}),
        ("empty", "   ", {"invalid_query"}),
    ]

    def test_report_counts_and_latency_paths(self):
        pipe, _ = build_pipeline([GOOD])
        report = evaluate_rag.run_evaluation(pipe, self.QUESTIONS)
        assert report["questions"] == 3
        assert report["as_expected"] == 3
        assert report["status_counts"] == {"answered": 1, "out_of_domain": 1, "invalid_query": 1}
        assert report["latency_with_llm"]["count"] == 1
        assert report["latency_without_llm"]["count"] == 2
        assert report["models_used"] == {"fake-model": 1}
        assert report["llm_calls"] == 1

    def test_unexpected_behaviour_is_flagged(self):
        pipe, _ = build_pipeline([GOOD])
        questions = [("off-topic", "transformer attention", {"out_of_domain"})]
        report = evaluate_rag.run_evaluation(pipe, questions)
        assert report["as_expected"] == 0
        assert report["rows"][0]["as_expected"] is False

    def test_pauses_only_after_questions_that_used_the_llm(self):
        pipe, _ = build_pipeline([GOOD])
        pauses = []
        evaluate_rag.run_evaluation(pipe, self.QUESTIONS, sleep_s=2.0, sleep=pauses.append)
        assert pauses == [2.0]

    def test_markdown_has_tables_and_answers(self):
        pipe, _ = build_pipeline([GOOD])
        report = evaluate_rag.run_evaluation(pipe, self.QUESTIONS)
        text = evaluate_rag.markdown_report(report)
        assert "| Type | Question | Status |" in text
        assert "3 of 3 questions behaved as expected" in text
        assert GOOD in text

    def test_rejected_draft_is_shown_in_the_report(self):
        bad = "It reached 99.9 BLEU on translation [1]."
        pipe, _ = build_pipeline([bad, bad])
        report = evaluate_rag.run_evaluation(pipe, [("answerable", "transformer attention", evaluate_rag.ANSWERED)])
        assert report["rows"][0]["status"] == "rejected_ungrounded"
        assert bad in evaluate_rag.markdown_report(report)
