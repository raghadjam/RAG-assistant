"""Tests for scripts/build_retrieval_testset.py, evaluate_retrieval.py and evaluate_generation.py."""

import json
import socket
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_retrieval_testset  # noqa: E402
import evaluate_generation  # noqa: E402
import evaluate_retrieval  # noqa: E402

from src.chunking import chunk_document  # noqa: E402
from src.evaluation import build_retrieval_testset as build_set  # noqa: E402
from src.evaluation import load_testset, run_retrieval_evaluation  # noqa: E402
from src.llm_client import LLMUnavailableError  # noqa: E402
from src.prompts import REFUSAL_SENTENCE  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402
from tests.fakes import HashEmbedder  # noqa: E402
from tests.test_api import make_api  # noqa: E402
from tests.test_evaluation import CONCEPTS, synthetic_corpus  # noqa: E402


def run_main(module, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", [module.__name__, *args])
    return module.main()


# --------------------------------------------------------------------------- #
class TestBuildTestsetScript:
    def corpus_file(self, tmp_path):
        path = tmp_path / "enriched.parquet"
        synthetic_corpus().to_parquet(path)
        return path

    def test_builds_the_file_and_reports_what_it_made(self, tmp_path, monkeypatch, capsys):
        out = tmp_path / "eval" / "set.jsonl"
        concepts = tmp_path / "concepts.txt"
        concepts.write_text("# comment line\nfederated learning\n\nnever mentioned phrase\n")
        code = run_main(build_retrieval_testset, monkeypatch, "--corpus", str(self.corpus_file(tmp_path)),
                        "--out", str(out), "--n-title", "6", "--n-sentence", "6", "--concepts-file", str(concepts))
        printed = capsys.readouterr().out
        assert code == 0
        items = load_testset(out)
        assert [i["type"] for i in items].count("title") == 6 and [i["type"] for i in items].count("concept") == 1
        assert "federated learning" in printed and "Dropped" in printed and "never mentioned phrase" in printed

    def test_an_existing_test_set_is_not_overwritten_without_force(self, tmp_path, monkeypatch):
        corpus, out = self.corpus_file(tmp_path), tmp_path / "set.jsonl"
        args = ["--corpus", str(corpus), "--out", str(out), "--n-title", "3", "--n-sentence", "3"]
        assert run_main(build_retrieval_testset, monkeypatch, *args) == 0
        first = out.read_text()
        assert run_main(build_retrieval_testset, monkeypatch, *args, "--seed", "7") == 1
        assert out.read_text() == first
        assert run_main(build_retrieval_testset, monkeypatch, *args, "--seed", "7", "--force") == 0
        assert out.read_text() != first

    def test_missing_corpus_points_to_the_enrichment_script(self, tmp_path, monkeypatch, capsys):
        assert run_main(build_retrieval_testset, monkeypatch, "--corpus", str(tmp_path / "none.parquet")) == 1
        assert "enrich_corpus" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def retrieval_report():
    corpus = synthetic_corpus()
    embedder = HashEmbedder()
    store = VectorStore(path=None, collection_name=f"r_{uuid.uuid4().hex}")
    chunks = []
    for row in corpus.itertuples(index=False):
        chunks.extend(chunk_document(row.doc_id, row.text, {"title": row.title, "topic_id": int(row.topic_id)}, 500, 80))
    store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
    testset = build_set(corpus, CONCEPTS, n_title=8, n_sentence=8)
    return run_retrieval_evaluation(testset, embedder, store, ks=(1, 3, 5))


class TestEvaluateRetrievalScript:
    def test_markdown_has_definitions_and_a_row_per_query_type(self, retrieval_report):
        text = evaluate_retrieval.markdown_report(retrieval_report)
        assert "Precision@K" in text and "Recall@K" in text and "MRR" in text
        for label in ("title (known paper)", "sentence (known paper)", "concept (many papers)", "all queries"):
            assert label in text
        assert "| P@1 | P@3 | P@5 | R@1 | R@3 | R@5 |" in text.replace("| Query type | Queries ", "")

    def test_markdown_shows_the_filter_comparison(self, retrieval_report):
        text = evaluate_retrieval.markdown_report(retrieval_report)
        assert "topic as a filter" in text
        assert "title (known paper): no filter" in text and "title (known paper): topic filter" in text

    def test_markdown_lists_queries_that_were_not_found(self, retrieval_report):
        text = evaluate_retrieval.markdown_report(retrieval_report)
        assert "not found in the top 5" in text

    def test_columns_follow_the_requested_ks(self, retrieval_report):
        header = [l for l in evaluate_retrieval.markdown_report(retrieval_report).splitlines()
                  if l.startswith("| Query type |")][0]
        assert header.count("P@") == 3 and "P@10" not in header

    def test_numbers_in_the_table_match_the_report(self, retrieval_report):
        row = [l for l in evaluate_retrieval.markdown_report(retrieval_report).splitlines()
               if l.startswith("| title (known paper) |")][0]
        assert f"{retrieval_report['by_type']['title']['recall'][5]:.2f}" in row

    def test_missing_testset_points_to_the_builder(self, tmp_path, monkeypatch, capsys):
        assert run_main(evaluate_retrieval, monkeypatch, "--testset", str(tmp_path / "none.jsonl")) == 1
        assert "build_retrieval_testset" in capsys.readouterr().out

    def test_invalid_k_is_rejected(self, tmp_path, monkeypatch):
        path = tmp_path / "set.jsonl"
        path.write_text('{"id": "a", "type": "title", "query": "q", "relevant_doc_ids": ["d"]}\n')
        assert run_main(evaluate_retrieval, monkeypatch, "--testset", str(path), "--ks", "0", "5") == 1


# --------------------------------------------------------------------------- #
GOOD = "The Transformer relies on attention mechanisms for language translation [1]."
CNN = "Convolutional networks classify images and detect objects [1]."
RL = "Agents maximize reward in game environments through reinforcement learning [1]."

QUESTIONS = [
    {"id": "doc-00", "category": "doc", "question": "transformer attention mechanism for language translation",
     "expected": ["answered", "answered_with_warnings"], "doc_id": "d1",
     "reference": "The transformer uses attention mechanisms for language translation."},
    {"id": "doc-01", "category": "doc", "question": "convolutional networks for image classification and object detection",
     "expected": ["answered", "answered_with_warnings"], "doc_id": "d2",
     "reference": "Convolutional networks classify images and detect objects."},
    {"id": "answerable-02", "category": "answerable", "question": "reinforcement learning agents maximize reward",
     "expected": ["answered", "answered_with_warnings"]},
    {"id": "unanswerable-03", "category": "unanswerable", "question": "attention mechanism exact accuracy figure",
     "expected": ["no_answer", "rejected_ungrounded", "out_of_domain"]},
    {"id": "off-topic-04", "category": "off-topic", "question": "best pizza recipe with fresh basil",
     "expected": ["out_of_domain"]},
    {"id": "nonsense-05", "category": "nonsense", "question": "qzxv blorf wibble", "expected": ["out_of_domain"]},
]


def ask_body(status="answered", answer="Fine [1].", llm=True, sources=("d1",), timing_ms=10.0):
    return {
        "request_id": "r", "question": "q", "status": status, "answer": answer,
        "sources": [{"n": i + 1, "doc_id": d, "score": 0.9} for i, d in enumerate(sources)], "top_score": 0.9,
        "timing": {"embed_ms": 1.0, "search_ms": 2.0, "retrieval_ms": 3.0, "generation_ms": timing_ms - 3.0
                   if llm else 0.0, "total_ms": timing_ms},
        "llm": {"model": "m", "calls": 1, "http_attempts": 1, "repairs": 0, "prompt_tokens": 10,
                "completion_tokens": 5, "truncated": False, "fallbacks": []} if llm else None,
        "check": {"verdict": "pass", "citations": [1], "invalid_citations": [], "unsupported_numbers": [],
                  "support_ratio": 1.0, "hard_failures": [], "soft_failures": []} if llm else None,
    }


class StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, path, payload, request_id):
        self.calls.append((path, payload, request_id))
        return self.responses.pop(0)


def full_run(tmp_path, replies, questions=QUESTIONS, **kwargs):
    api = make_api(tmp_path, replies=replies)
    client = evaluate_generation.InProcessClient(api.app)
    sleeps = []
    report = evaluate_generation.run_generation_evaluation(client, questions, HashEmbedder(),
                                                           sleep=sleeps.append, **kwargs)
    return report, sleeps, api


class TestGenerationEvaluation:
    REPLIES = [GOOD, CNN, RL, REFUSAL_SENTENCE]

    def test_a_full_run_through_the_real_api(self, tmp_path):
        report, _, api = full_run(tmp_path, self.REPLIES)
        assert report["questions"] == 6 and report["outcomes"] == {"completed": 6}
        assert [r["status"] for r in report["rows"]] == [
            "answered", "answered", "answered", "no_answer", "out_of_domain", "out_of_domain"]
        assert all(r["as_expected"] for r in report["rows"])
        assert len(api.llm.calls) == 4                         # the two refusals never reached the LLM
        assert report["unsafe_answers"] == 0 and report["llm_availability_rate"] == 1.0
        assert report["llm_questions_attempted"] == 4

    def test_behaviour_is_summarised_per_category(self, tmp_path):
        behaviour = full_run(tmp_path, self.REPLIES)[0]["behaviour"]
        assert behaviour["doc"] == {"completed": 2, "as_expected": 2, "statuses": {"answered": 2}}
        assert behaviour["unanswerable"]["statuses"] == {"no_answer": 1}
        assert behaviour["off-topic"]["as_expected"] == 1

    def test_answer_quality_measures(self, tmp_path):
        quality = full_run(tmp_path, self.REPLIES)[0]["quality"]
        assert quality["answered"] == 3 and quality["doc_questions_answered"] == 2
        assert quality["citation_coverage_mean"] == 1.0 and quality["verdict_pass_rate"] == 1.0
        assert quality["all_citations_valid_rate"] == 1.0 and quality["support_ratio_mean"] == 1.0
        assert quality["target_paper_cited_rate"] == 1.0
        assert quality["similarity_to_reference_mean"] > quality["similarity_to_other_papers_mean"]
        assert 0 < quality["rouge1_f1_mean"] <= 1 and quality["relevance_to_question_mean"] > 0

    def test_the_request_id_links_the_evaluation_to_the_api_log(self, tmp_path):
        report, _, api = full_run(tmp_path, self.REPLIES)
        logged = {json.loads(l)["request_id"] for l in api.log_path.read_text().splitlines()}
        assert len(logged) == 6 and all(rid.startswith("eval-") for rid in logged)

    def test_latency_is_split_by_path(self, tmp_path):
        latency = full_run(tmp_path, self.REPLIES)[0]["latency"]
        assert latency["with_llm"]["n"] == 4 and latency["without_llm"]["n"] == 2
        assert latency["without_llm"]["generation_ms_mean"] == 0
        assert latency["with_llm"]["server_total_ms"]["mean"] > 0
        assert latency["with_llm"]["client_total_ms"]["mean"] > 0

    def test_token_and_model_accounting(self, tmp_path):
        report = full_run(tmp_path, self.REPLIES)[0]
        assert report["tokens"] == {"prompt": 40, "completion": 20, "repairs": 0, "failed_attempts_before_success": 0}
        assert report["models_used"] == {"fake-model": 4}

    def test_an_outage_is_retried_after_waiting(self, tmp_path):
        questions = QUESTIONS[:1]
        report, sleeps, _ = full_run(tmp_path, [LLMUnavailableError(["m: HTTP 503"]), GOOD], questions,
                                     retries=1, retry_wait_s=7.0)
        row = report["rows"][0]
        assert row["attempts"] == 2 and row["outcome"] == "completed" and row["status"] == "answered"
        assert 7.0 in sleeps

    def test_an_outage_without_retries_is_counted_and_kept_out_of_quality(self, tmp_path):
        report, _, _ = full_run(tmp_path, [LLMUnavailableError(["m: HTTP 503"]), RL], QUESTIONS[:1] + QUESTIONS[2:3],
                                retries=0)
        first, second = report["rows"]
        assert first["outcome"] == "llm_unavailable" and first["as_expected"] is None and first["quality"] is None
        assert first["http_status"] == 503 and first["attempts"] == 1
        assert second["outcome"] == "completed"
        assert report["outcomes"] == {"llm_unavailable": 1, "completed": 1}
        assert report["llm_availability_rate"] == 0.5 and report["quality"]["answered"] == 1
        assert report["behaviour"].get("doc") is None          # no completed doc question to judge

    def test_persistent_outage_gives_up_after_the_retries(self, tmp_path):
        down = [LLMUnavailableError(["x"])] * 3
        report, sleeps, _ = full_run(tmp_path, down, QUESTIONS[:1], retries=2, retry_wait_s=1.0)
        assert report["rows"][0]["attempts"] == 3 and report["rows"][0]["outcome"] == "llm_unavailable"
        assert sleeps == [1.0, 1.0]
        assert report["llm_availability_rate"] == 0.0

    def test_the_llm_question_cap_skips_questions_but_not_refusal_checks(self, tmp_path):
        report, _, api = full_run(tmp_path, [GOOD], max_llm_asks=1)
        outcomes = [r["outcome"] for r in report["rows"]]
        assert outcomes == ["completed", "skipped", "skipped", "skipped", "completed", "completed"]
        assert len(api.llm.calls) == 1
        assert report["outcomes"] == {"completed": 3, "skipped": 3}

    def test_it_pauses_only_after_questions_that_used_the_llm(self, tmp_path):
        _, sleeps, _ = full_run(tmp_path, self.REPLIES, sleep_s=2.5)
        assert sleeps == [2.5, 2.5, 2.5, 2.5]

    def test_an_unexpected_status_is_flagged(self, tmp_path):
        report, _, _ = full_run(tmp_path, [REFUSAL_SENTENCE], QUESTIONS[:1])
        assert report["rows"][0]["status"] == "no_answer" and report["rows"][0]["as_expected"] is False
        assert report["behaviour"]["doc"]["as_expected"] == 0

    def test_http_level_failures_are_recorded_not_fatal(self):
        stub = StubClient([(500, {"error": {"code": "internal_error", "message": "boom"}}),
                           (0, {"error": {"code": "client_error", "message": "ConnectionError: refused"}}),
                           (200, ask_body())])
        questions = [{"id": f"q{i}", "category": "answerable", "question": "x", "expected": ["answered"]}
                     for i in range(3)]
        report = evaluate_generation.run_generation_evaluation(stub, questions, HashEmbedder())
        assert [r["outcome"] for r in report["rows"]] == ["http_error", "http_error", "completed"]
        assert report["rows"][0]["answer"] == "internal_error: boom"
        assert report["rows"][1]["http_status"] == 0

    def test_answering_an_off_topic_question_counts_as_unsafe(self):
        stub = StubClient([(200, ask_body("answered", "Sure, here is a poem [1]."))])
        questions = [{"id": "o", "category": "off-topic", "question": "poem", "expected": ["out_of_domain"]}]
        report = evaluate_generation.run_generation_evaluation(stub, questions, HashEmbedder())
        assert report["unsafe_answers"] == 1 and report["rows"][0]["as_expected"] is False

    def test_a_503_that_is_not_an_llm_outage_is_an_http_error(self):
        stub = StubClient([(503, {"error": {"code": "index_empty", "message": "empty"}})])
        questions = [{"id": "q", "category": "answerable", "question": "x", "expected": ["answered"]}]
        row = evaluate_generation.run_generation_evaluation(stub, questions, HashEmbedder())["rows"][0]
        assert row["outcome"] == "http_error" and len(stub.calls) == 1

    def test_empty_run_does_not_crash(self):
        report = evaluate_generation.run_generation_evaluation(StubClient([]), [], HashEmbedder())
        assert report["questions"] == 0 and report["llm_availability_rate"] is None
        assert report["quality"]["answered"] == 0
        evaluate_generation.markdown_report(report)


class TestGenerationReport:
    def test_markdown_sections(self, tmp_path):
        report = full_run(tmp_path, TestGenerationEvaluation.REPLIES)[0]
        text = evaluate_generation.markdown_report(report)
        for heading in ("Did the system do the right thing?", "Answer quality", "Latency", "Every question", "## Answers"):
            assert heading in text
        assert "6 questions: 6 completed" in text and "LLM availability: 100%" in text
        assert GOOD in text and "Reference (abstract opening)" in text
        assert "Answered by the LLM" in text and "Stopped before the LLM" in text

    def test_similarity_scores_are_not_shown_as_percentages(self, tmp_path):
        text = evaluate_generation.markdown_report(full_run(tmp_path, TestGenerationEvaluation.REPLIES)[0])
        line = [l for l in text.splitlines() if l.startswith("| Similarity of answer to question")][0]
        assert "%" not in line
        rate = [l for l in text.splitlines() if l.startswith("| Citation coverage")][0]
        assert "100%" in rate

    def test_outages_and_skips_are_visible(self, tmp_path):
        report, _, _ = full_run(tmp_path, [LLMUnavailableError(["x"]), GOOD], QUESTIONS[:2], retries=0)
        text = evaluate_generation.markdown_report(report)
        assert "1 hit an LLM outage" in text and "llm_unavailable" in text

    def test_default_questions(self):
        corpus = synthetic_corpus()
        questions = evaluate_generation.default_questions(corpus, n_doc=5)
        categories = [q["category"] for q in questions]
        assert categories.count("doc") == 5 and "empty" not in categories
        assert all(q["expected"] for q in questions) and all(q["reference"] for q in questions[:5])
        assert len({q["id"] for q in questions}) == len(questions)
        assert "doc" not in [q["category"] for q in evaluate_generation.default_questions(None, n_doc=5)]
        assert "doc" not in [q["category"] for q in evaluate_generation.default_questions(corpus, n_doc=0)]


class ScriptedServer:
    """A throwaway HTTP server for testing the HTTP client."""

    def __init__(self, status=200, body='{"ok": true}'):
        outer = self
        self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"path": self.path, "request_id": self.headers.get("X-Request-ID"),
                                       "json": json.loads(raw)})
                payload = body.encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestHttpClient:
    def test_posts_json_with_a_request_id(self):
        server = ScriptedServer(body='{"status": "answered"}')
        try:
            client = evaluate_generation.HttpClient(server.url)
            client.session.trust_env = False
            status, body = client.post("/ask", {"question": "hi"}, "eval-abc-12345")
        finally:
            server.close()
        assert (status, body) == (200, {"status": "answered"})
        assert server.requests[0] == {"path": "/ask", "request_id": "eval-abc-12345", "json": {"question": "hi"}}

    def test_connection_failure_becomes_status_zero(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        client = evaluate_generation.HttpClient(f"http://127.0.0.1:{port}", timeout=2)
        client.session.trust_env = False
        status, body = client.post("/ask", {}, "eval-abc-12345")
        assert status == 0 and body["error"]["code"] == "client_error"

    def test_a_non_json_reply_is_reported(self):
        server = ScriptedServer(status=502, body="<html>Bad Gateway</html>")
        try:
            client = evaluate_generation.HttpClient(server.url)
            client.session.trust_env = False
            status, body = client.post("/ask", {}, "eval-abc-12345")
        finally:
            server.close()
        assert status == 502 and body["error"]["code"] == "bad_response"

    def test_in_process_client_passes_the_request_id_through(self, tmp_path):
        api = make_api(tmp_path, replies=[GOOD])
        client = evaluate_generation.InProcessClient(api.app)
        status, body = client.post("/ask", {"question": "transformer attention"}, "eval-xyz-98765")
        assert status == 200 and body["request_id"] == "eval-xyz-98765"
