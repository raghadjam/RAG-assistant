"""Tests for the FastAPI service (src/api.py).

The service runs against a real RAGPipeline and a real in-memory ChromaDB, with a hashing
embedder and a scripted LLM, so no model download, network or API key is needed.
"""

import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.api as api_module  # noqa: E402
from src.api import API_VERSION, create_app  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.config import Settings, load_settings  # noqa: E402
from src.llm_client import LLMUnavailableError  # noqa: E402
from src.prompts import REFUSAL_SENTENCE  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402
from tests.fakes import FakeLLM, HashEmbedder  # noqa: E402

GOOD = "The Transformer relies on attention mechanisms for language translation [1]."

# doc_id: (title, text, topic_id, sentiment)
PAPERS = {
    "d1": ("Attention", "transformer attention mechanism for language translation. " * 4, 0, "positive"),
    "d2": ("Vision", "convolutional networks for image classification and object detection. " * 4, 1, "negative"),
    "d3": ("RL", "reinforcement learning agents maximize reward in game environments. " * 4, 1, "neutral"),
}


def make_api(tmp_path, replies=(), enriched=True, docs=None, console_log=False, **settings_kwargs):
    docs = PAPERS if docs is None else docs
    embedder = HashEmbedder()
    store = VectorStore(path=None, collection_name=f"api_{uuid.uuid4().hex}")
    chunks = []
    for doc_id, (title, text, topic, sentiment) in docs.items():
        meta = {"title": title}
        if enriched:
            meta.update(topic_id=topic, topic_name=f"topic {topic}", sentiment_label=sentiment, sentiment_score=0.2)
        chunks.extend(chunk_document(doc_id, text, meta, 300, 50))
    if chunks:
        store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))

    options = {"api_key": "k", "min_score": 0.3, "min_chunk_score": 0.2, "log_path": None,
               "api_log_path": None, **settings_kwargs}
    settings = Settings(**options)
    llm = FakeLLM(list(replies))
    pipeline = RAGPipeline(embedder, store, llm, settings)
    log_path = tmp_path / "api.jsonl"
    app = create_app(pipeline, settings, log_path=str(log_path), console_log=console_log)
    client = TestClient(app, raise_server_exceptions=False)
    return SimpleNamespace(app=app, client=client, pipe=pipeline, llm=llm, store=store, log_path=log_path)


def log_lines(api):
    return [json.loads(line) for line in api.log_path.read_text().splitlines()]


@pytest.fixture
def api(tmp_path):
    return make_api(tmp_path, replies=[GOOD])


# --------------------------------------------------------------------------- #
class TestMonitoring:
    def test_health(self, api):
        r = api.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok" and r.json()["version"] == API_VERSION
        assert r.json()["uptime_s"] >= 0

    def test_ready_reports_the_index_and_llm(self, api):
        body = api.client.get("/ready").json()
        assert body["status"] == "ready"
        assert body["chunks"] == api.store.count() == 3
        assert body["enriched"] is True and body["llm_configured"] is True
        assert body["models"] == list(api.pipe.settings.models)

    def test_ready_says_when_the_index_has_no_nlp_metadata(self, tmp_path):
        api = make_api(tmp_path, enriched=False)
        assert api.client.get("/ready").json()["enriched"] is False

    def test_ready_without_an_api_key(self, tmp_path):
        api = make_api(tmp_path, api_key=None)
        assert api.client.get("/ready").json()["llm_configured"] is False

    def test_empty_index_is_not_ready(self, tmp_path):
        api = make_api(tmp_path, docs={})
        r = api.client.get("/ready")
        assert r.status_code == 503 and r.json()["status"] == "not_ready"
        assert "empty" in r.json()["reason"]
        assert r.json()["llm_configured"] is True and r.json()["models"]      # settings are known even so
        assert api.client.get("/health").status_code == 200       # still alive

    def test_root_points_to_the_docs(self, api):
        assert api.client.get("/").json()["docs"] == "/docs"


class TestStartup:
    def test_pipeline_is_built_at_startup_when_none_is_given(self, tmp_path, monkeypatch):
        donor = make_api(tmp_path, replies=[GOOD])
        monkeypatch.setattr(api_module, "build_pipeline", lambda settings: donor.pipe)
        app = create_app(None, donor.pipe.settings, log_path=None, console_log=False)
        with TestClient(app) as client:
            assert client.get("/ready").json()["status"] == "ready"
            assert client.post("/ask", json={"question": "transformer attention"}).status_code == 200

    def test_failed_startup_keeps_the_service_alive_and_explains(self, tmp_path, monkeypatch):
        def explode(settings):
            raise RuntimeError("cannot open the index")

        monkeypatch.setattr(api_module, "build_pipeline", explode)
        app = create_app(None, Settings(api_log_path=None), log_path=None, console_log=False)
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/health").status_code == 200
            ready = client.get("/ready")
            assert ready.status_code == 503 and "cannot open the index" in ready.json()["reason"]
            r = client.post("/ask", json={"question": "anything"})
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "service_unavailable"
            assert r.headers["retry-after"] == "30"

    def test_empty_index_rejects_questions_with_503(self, tmp_path):
        api = make_api(tmp_path, docs={})
        for path, body in (("/ask", {"question": "hello there"}), ("/search", {"query": "hello"})):
            r = api.client.post(path, json=body)
            assert r.status_code == 503 and r.json()["error"]["code"] == "index_empty"
        assert api.client.get("/topics").status_code == 503


class TestTopics:
    def test_lists_topics_with_sizes_and_sentiments(self, api):
        body = api.client.get("/topics").json()
        assert body["enriched"] is True and body["total_chunks"] == 3
        assert body["topics"] == [
            {"topic_id": 0, "topic_name": "topic 0", "chunks": 1, "documents": 1},
            {"topic_id": 1, "topic_name": "topic 1", "chunks": 2, "documents": 2},
        ]
        assert body["sentiments"] == {"positive": 1, "negative": 1, "neutral": 1}

    def test_index_without_metadata(self, tmp_path):
        body = make_api(tmp_path, enriched=False).client.get("/topics").json()
        assert body["enriched"] is False and body["topics"] == []

    def test_catalog_is_computed_once(self, api, monkeypatch):
        calls = []
        real = api_module._load_catalog
        monkeypatch.setattr(api_module, "_load_catalog", lambda store: calls.append(1) or real(store))
        api.client.get("/topics")
        api.client.get("/topics")
        api.client.post("/search", json={"query": "attention", "filters": {"topic_id": 0}})
        assert len(calls) == 1


class TestSearch:
    def search(self, api, **body):
        return api.client.post("/search", json={"query": "transformer attention", **body})

    def test_returns_ranked_chunks_with_metadata(self, api):
        r = self.search(api, top_k=3)
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "transformer attention"
        top = body["results"][0]
        assert top["doc_id"] == "d1" and top["title"] == "Attention"
        assert top["topic_id"] == 0 and top["sentiment_label"] == "positive"
        assert top["text"] and 0 < top["score"] <= 1
        scores = [h["score"] for h in body["results"]]
        assert scores == sorted(scores, reverse=True)
        assert set(body["timing"]) == {"embed_ms", "search_ms", "retrieval_ms", "generation_ms", "total_ms"}

    def test_top_k_limits_the_results(self, api):
        assert len(self.search(api, top_k=1).json()["results"]) == 1
        assert len(self.search(api, top_k=20).json()["results"]) == 3       # only 3 chunks exist

    def test_topic_filter(self, api):
        hits = self.search(api, top_k=10, filters={"topic_id": 1}).json()["results"]
        assert hits and all(h["topic_id"] == 1 for h in hits)
        assert {h["doc_id"] for h in hits} == {"d2", "d3"}

    def test_sentiment_filter(self, api):
        hits = self.search(api, top_k=10, filters={"sentiment": "negative"}).json()["results"]
        assert [h["doc_id"] for h in hits] == ["d2"]

    def test_combined_filters(self, api):
        hits = self.search(api, top_k=10, filters={"topic_id": 1, "sentiment": "neutral"}).json()["results"]
        assert [h["doc_id"] for h in hits] == ["d3"]

    def test_filters_are_echoed(self, api):
        assert self.search(api, filters={"topic_id": 0}).json()["filters"] == {"topic_id": 0, "sentiment": None}

    def test_valid_filters_matching_nothing_give_an_empty_list_not_an_error(self, api):
        r = self.search(api, filters={"topic_id": 0, "sentiment": "negative"})
        assert r.status_code == 200 and r.json()["results"] == []

    def test_unassigned_topic_id_minus_one_is_accepted_when_it_exists(self, tmp_path):
        api = make_api(tmp_path, docs={"d1": ("A", "transformer attention mechanism. " * 6, -1, "neutral")})
        r = api.client.post("/search", json={"query": "transformer attention", "filters": {"topic_id": -1}})
        assert r.status_code == 200 and len(r.json()["results"]) >= 1

    def test_unknown_topic_is_a_400_that_lists_valid_ids(self, api):
        r = self.search(api, filters={"topic_id": 99})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "unknown_topic"
        assert "[0, 1]" in r.json()["error"]["message"]

    def test_filters_on_an_index_without_metadata_are_a_400(self, tmp_path):
        api = make_api(tmp_path, enriched=False)
        for filters in ({"topic_id": 0}, {"sentiment": "positive"}):
            r = api.client.post("/search", json={"query": "transformer", "filters": filters})
            assert r.status_code == 400 and r.json()["error"]["code"] == "filters_unavailable"
        assert api.client.post("/search", json={"query": "transformer"}).status_code == 200   # no filter: fine

    @pytest.mark.parametrize("query", ["", "   ", "\n\t "])
    def test_blank_query_is_a_400(self, api, query):
        r = api.client.post("/search", json={"query": query})
        assert r.status_code == 400 and r.json()["error"]["code"] == "blank_text"

    @pytest.mark.parametrize("body", [
        {},                                                            # query missing
        {"query": 5},                                                  # wrong type
        {"query": "x" * 1001},                                         # too long
        {"query": "ok", "top_k": 0}, {"query": "ok", "top_k": 21},     # out of range
        {"query": "ok", "top_k": "five"}, {"query": "ok", "top_k": 2.5},
        {"query": "ok", "extra": 1},                                   # unknown field
        {"query": "ok", "filters": {"colour": "red"}},                 # unknown filter
        {"query": "ok", "filters": {"sentiment": "happy"}},
        {"query": "ok", "filters": {"topic_id": "two"}}, {"query": "ok", "filters": {"topic_id": -2}},
        {"query": "ok", "filters": "topic 1"},
        [], "just a string", None,
    ])
    def test_schema_violations_are_422(self, api, body):
        r = api.client.post("/search", json=body)
        assert r.status_code == 422
        error = r.json()["error"]
        assert error["code"] == "validation_error" and error["details"]
        assert all({"field", "message", "type"} <= set(d) for d in error["details"])

    def test_the_failing_field_is_named(self, api):
        details = api.client.post("/search", json={"query": "ok", "top_k": 99}).json()["error"]["details"]
        assert details[0]["field"] == "top_k"
        details = api.client.post("/search", json={"query": "ok", "filters": {"sentiment": "x"}}).json()["error"]["details"]
        assert details[0]["field"] == "filters.sentiment"

    def test_maximum_length_query_is_accepted(self, api):
        assert api.client.post("/search", json={"query": "attention " * 100}).status_code == 200
        assert api.client.post("/search", json={"query": "a" * 1000}).status_code == 200


class TestAsk:
    def ask(self, api, **body):
        return api.client.post("/ask", json={"question": "transformer attention", **body})

    def test_grounded_answer(self, api):
        r = self.ask(api)
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "answered" and body["answer"] == GOOD
        assert body["sources"][0]["n"] == 1 and body["sources"][0]["title"] == "Attention"
        assert body["sources"][0]["topic_id"] == 0 and body["sources"][0]["sentiment_label"] == "positive"
        assert body["check"]["verdict"] == "pass" and body["check"]["citations"] == [1]
        assert body["llm"]["model"] == "fake-model" and body["llm"]["calls"] == 1
        assert body["llm"]["prompt_tokens"] == 10 and body["llm"]["completion_tokens"] == 5
        assert body["top_score"] > 0.5 and body["note"] is None
        assert body["timing"]["total_ms"] >= body["timing"]["retrieval_ms"] > 0

    def test_request_id_in_body_matches_the_header(self, api):
        r = self.ask(api)
        assert r.json()["request_id"] == r.headers["x-request-id"]

    def test_filters_reach_the_retriever_and_top_k_overrides_the_default(self, api, monkeypatch):
        seen = {}
        original = api.pipe.ask

        def spy(question, where=None, top_k=None):
            seen.update(question=question, where=where, top_k=top_k)
            return original(question, where=where, top_k=top_k)

        monkeypatch.setattr(api.pipe, "ask", spy)
        r = self.ask(api, top_k=3, filters={"topic_id": 0, "sentiment": "positive"})
        assert r.status_code == 200 and r.json()["filters"] == {"topic_id": 0, "sentiment": "positive"}
        assert seen["where"] == {"$and": [{"topic_id": 0}, {"sentiment_label": "positive"}]}
        assert seen["top_k"] == 3

    def test_no_filter_means_no_where_clause(self, api, monkeypatch):
        seen = {}
        original = api.pipe.ask
        monkeypatch.setattr(api.pipe, "ask", lambda q, where=None, top_k=None:
                            seen.update(where=where, top_k=top_k) or original(q, where=where, top_k=top_k))
        self.ask(api)
        assert seen == {"where": None, "top_k": None}

    def test_filtered_ask_only_uses_chunks_from_that_topic(self, tmp_path):
        api = make_api(tmp_path, replies=["Convolutional networks classify images [1]."])
        r = api.client.post("/ask", json={"question": "convolutional networks image classification",
                                          "filters": {"topic_id": 1}})
        assert r.status_code == 200 and r.json()["status"] == "answered"
        assert all(s["topic_id"] == 1 for s in r.json()["sources"])

    def test_filter_that_leaves_only_unrelated_chunks_is_a_normal_refusal(self, api):
        r = self.ask(api, filters={"topic_id": 1})        # attention question, vision/RL papers only
        assert r.status_code == 200 and r.json()["status"] == "out_of_domain"
        assert api.llm.calls == []

    def test_off_topic_question_is_a_200_refusal_without_an_llm_call(self, api):
        r = api.client.post("/ask", json={"question": "best pizza recipe with fresh basil"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "out_of_domain" and body["llm"] is None
        assert body["timing"]["generation_ms"] == 0 and body["note"]
        assert api.llm.calls == []

    def test_model_refusal_is_a_200_no_answer(self, tmp_path):
        api = make_api(tmp_path, replies=[REFUSAL_SENTENCE])
        r = api.client.post("/ask", json={"question": "transformer attention"})
        assert r.status_code == 200 and r.json()["status"] == "no_answer"

    def test_unverifiable_answer_is_rejected_but_still_a_200(self, tmp_path):
        bad = "It reached 99.9 BLEU on translation [1]."
        api = make_api(tmp_path, replies=[bad, bad])
        r = api.client.post("/ask", json={"question": "transformer attention"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "rejected_ungrounded" and "99.9" not in body["answer"]
        assert body["llm"]["repairs"] == 1 and body["check"]["verdict"] == "fail"
        assert "draft_answer" not in body                  # the rejected text never leaves the server

    def test_llm_outage_is_a_503_with_the_sources_that_were_found(self, tmp_path):
        api = make_api(tmp_path, replies=[LLMUnavailableError(["model-a: HTTP 503", "model-b: HTTP 429"])])
        r = api.client.post("/ask", json={"question": "transformer attention"})
        assert r.status_code == 503
        assert r.headers["retry-after"] == "30"
        body = r.json()
        assert body["status"] == "llm_unavailable" and body["sources"]
        assert "HTTP 503" in body["note"]

    @pytest.mark.parametrize("question", ["", "   ", "\n\n"])
    def test_blank_question_is_a_400(self, api, question):
        r = api.client.post("/ask", json={"question": question})
        assert r.status_code == 400 and r.json()["error"]["code"] == "blank_text"
        assert api.llm.calls == []

    def test_invalid_json_is_a_400(self, api):
        for payload in ("{not json", '{"question": "hi"', "{'question': 'single quotes'}", "[1, 2,"):
            r = api.client.post("/ask", content=payload, headers={"content-type": "application/json"})
            assert r.status_code == 400, payload
            assert r.json()["error"]["code"] == "invalid_json"

    def test_an_empty_body_is_a_missing_body_not_invalid_json(self, api):
        r = api.client.post("/ask", content="", headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["error"]["details"][0]["type"] == "missing"

    def test_wrong_content_type_is_rejected(self, api):
        r = api.client.post("/ask", content="question=hello", headers={"content-type": "text/plain"})
        assert r.status_code in (400, 422)
        assert "error" in r.json()

    @pytest.mark.parametrize("body", [
        {}, {"question": None}, {"question": 42}, {"question": ["a"]}, {"question": "x" * 1001},
        {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 11}, {"question": "ok", "top_k": "3"},
        {"question": "ok", "top_k": True}, {"question": "ok", "filters": {"topic_id": "1"}},
        {"question": "ok", "filters": {"topic_id": False}},
        {"question": "ok", "filters": {"topic_id": 1.5}}, {"question": "ok", "debug": True},
        {"questions": "typo in the field name"},
    ])
    def test_schema_violations_are_422(self, api, body):
        r = api.client.post("/ask", json=body)
        assert r.status_code == 422, body
        assert r.json()["error"]["code"] == "validation_error"
        assert api.llm.calls == []

    def test_unknown_topic_is_a_400(self, api):
        r = self.ask(api, filters={"topic_id": 42})
        assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_topic"

    def test_question_is_stripped_before_use(self, api):
        r = api.client.post("/ask", json={"question": "   transformer attention  "})
        assert r.json()["question"] == "transformer attention"

    def test_unicode_question(self, api):
        r = api.client.post("/ask", json={"question": "Wie funktioniert die Aufmerksamkeit? 注意力机制 🤖"})
        assert r.status_code == 200


class TestUnexpectedErrors:
    def crash(self, api, monkeypatch, error=RuntimeError("secret internal detail: /home/user/key"), path="/ask"):
        def boom(*args, **kwargs):
            raise error

        monkeypatch.setattr(api.pipe, "ask", boom)
        body = {"question": "transformer attention"} if path == "/ask" else {"query": "x"}
        return api.client.post(path, json=body)

    def test_a_bug_becomes_a_clean_500(self, api, monkeypatch):
        r = self.crash(api, monkeypatch)
        assert r.status_code == 500
        error = r.json()["error"]
        assert error["code"] == "internal_error"
        assert error["request_id"] == r.headers["x-request-id"]

    def test_internals_are_not_leaked_to_the_client(self, api, monkeypatch):
        text = self.crash(api, monkeypatch).text
        assert "secret internal detail" not in text and "Traceback" not in text and "/home/user" not in text

    def test_the_crash_is_logged_with_the_request_id_and_status(self, api, monkeypatch, caplog):
        with caplog.at_level("ERROR", logger="rag.api"):
            r = self.crash(api, monkeypatch)
        assert any(r.headers["x-request-id"] in rec.getMessage() for rec in caplog.records)
        assert any("secret internal detail" in str(rec.exc_info) for rec in caplog.records if rec.exc_info)
        line = log_lines(api)[-1]
        assert line["status_code"] == 500 and line["error_code"] == "internal_error"

    def test_the_service_keeps_working_after_a_crash(self, api, monkeypatch):
        self.crash(api, monkeypatch)
        monkeypatch.undo()
        assert api.client.get("/health").status_code == 200
        assert api.client.post("/search", json={"query": "transformer attention"}).status_code == 200

    def test_a_bug_in_search_is_also_a_500(self, api, monkeypatch):
        monkeypatch.setattr(api_module, "semantic_search", lambda *a, **k: 1 / 0)
        r = api.client.post("/search", json={"query": "transformer"})
        assert r.status_code == 500 and r.json()["error"]["code"] == "internal_error"


class TestErrorShape:
    def test_every_error_uses_the_same_envelope(self, api):
        responses = [
            api.client.post("/ask", json={"question": " "}),                       # 400
            api.client.post("/ask", json={}),                                      # 422
            api.client.post("/ask", content="{", headers={"content-type": "application/json"}),   # 400
            api.client.get("/missing"),                                            # 404
            api.client.get("/ask"),                                                # 405
            api.client.post("/search", json={"query": "x", "filters": {"topic_id": 77}}),   # 400
        ]
        assert [r.status_code for r in responses] == [400, 422, 400, 404, 405, 400]
        for r in responses:
            error = r.json()["error"]
            assert set(error) == {"code", "message", "details", "request_id"}
            assert error["request_id"] == r.headers["x-request-id"]
            assert isinstance(error["code"], str) and isinstance(error["message"], str)

    def test_404_and_405_codes(self, api):
        assert api.client.get("/nope").json()["error"]["code"] == "not_found"
        assert api.client.delete("/ask").json()["error"]["code"] == "method_not_allowed"

    def test_validation_details_never_echo_the_submitted_input(self, api):
        secret = "my-private-question-text-" + "x" * 2000
        r = api.client.post("/ask", json={"question": secret})
        assert r.status_code == 422 and "my-private-question-text" not in r.text


class TestRequestIds:
    def test_a_client_supplied_id_is_echoed(self, api):
        r = api.client.get("/health", headers={"X-Request-ID": "client-trace-12345"})
        assert r.headers["x-request-id"] == "client-trace-12345"
        assert log_lines(api)[-1]["request_id"] == "client-trace-12345"

    @pytest.mark.parametrize("bad", ["short", "has spaces in it", "x" * 65, "semi;colon;injection", "new\\nline"])
    def test_an_unsuitable_id_is_replaced(self, api, bad):
        r = api.client.get("/health", headers={"X-Request-ID": bad})
        assert r.headers["x-request-id"] != bad and len(r.headers["x-request-id"]) == 32

    def test_every_request_gets_a_distinct_id(self, api):
        ids = {api.client.get("/health").headers["x-request-id"] for _ in range(5)}
        assert len(ids) == 5

    def test_response_time_header(self, api):
        assert float(api.client.get("/health").headers["x-response-time-ms"]) >= 0


class TestRequestLogging:
    def test_one_json_line_per_request(self, api):
        api.client.get("/health")
        api.client.post("/ask", json={"question": "transformer attention"})
        api.client.post("/search", json={"query": "attention"})
        api.client.get("/nope")
        lines = log_lines(api)
        assert [(l["method"], l["path"], l["status_code"]) for l in lines] == [
            ("GET", "/health", 200), ("POST", "/ask", 200), ("POST", "/search", 200), ("GET", "/nope", 404)]
        for line in lines:
            assert {"time", "request_id", "duration_ms", "client"} <= set(line)
            assert line["duration_ms"] >= 0 and len(line["request_id"]) >= 8

    def test_ask_lines_carry_pipeline_details(self, api):
        api.client.post("/ask", json={"question": "transformer attention", "filters": {"topic_id": 0}})
        line = log_lines(api)[-1]
        assert line["rag_status"] == "answered" and line["filters"] == {"topic_id": 0}
        assert line["model"] == "fake-model" and line["llm_calls"] == 1
        assert (line["prompt_tokens"], line["completion_tokens"]) == (10, 5)
        assert line["check_verdict"] == "pass" and line["top_score"] > 0.5
        for key in ("embed_ms", "search_ms", "retrieval_ms", "generation_ms", "total_ms"):
            assert key in line
        assert line["question_chars"] == len("transformer attention")

    def test_search_lines_carry_retrieval_details(self, api):
        api.client.post("/search", json={"query": "transformer attention", "top_k": 2, "filters": {"sentiment": "positive"}})
        line = log_lines(api)[-1]
        assert line["n_results"] == 1 and line["top_k"] == 2 and line["filters"] == {"sentiment": "positive"}
        assert line["query_preview"] == "transformer attention" and line["top_score"] > 0

    def test_error_lines_carry_the_error_code(self, api):
        api.client.post("/ask", json={"question": "   "})
        api.client.post("/ask", json={})
        api.client.post("/ask", content="{", headers={"content-type": "application/json"})
        codes = [(l["status_code"], l["error_code"]) for l in log_lines(api)]
        assert codes == [(400, "blank_text"), (422, "validation_error"), (400, "invalid_json")]

    def test_only_a_short_preview_of_the_question_is_logged(self, api):
        long_question = "transformer attention " * 40
        api.client.post("/ask", json={"question": long_question})
        line = log_lines(api)[-1]
        assert len(line["question_preview"]) <= 80 and line["question_preview"].endswith("...")
        assert line["question_chars"] == len(long_question.strip())
        assert long_question.strip() not in api.log_path.read_text()

    def test_llm_outage_is_logged_with_its_status(self, tmp_path):
        api = make_api(tmp_path, replies=[LLMUnavailableError(["x: HTTP 503"])])
        api.client.post("/ask", json={"question": "transformer attention"})
        line = log_lines(api)[-1]
        assert line["status_code"] == 503 and line["rag_status"] == "llm_unavailable"

    def test_lines_are_printed_to_the_console_when_enabled(self, tmp_path, capsys):
        api = make_api(tmp_path, console_log=True)
        r = api.client.get("/health")
        printed = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
        assert printed[-1]["request_id"] == r.headers["x-request-id"]

    def test_nothing_is_printed_when_the_console_is_off(self, api, capsys):
        api.client.get("/health")
        assert capsys.readouterr().out == ""

    def test_an_unwritable_log_file_does_not_stop_the_service(self, tmp_path):
        blocker = tmp_path / "a_file"
        blocker.write_text("not a directory")
        settings = Settings(api_key="k", api_log_path=None)
        app = create_app(make_api(tmp_path).pipe, settings, log_path=str(blocker / "sub" / "log.jsonl"),
                         console_log=False)
        assert TestClient(app).get("/health").status_code == 200

    def test_creating_several_apps_does_not_duplicate_lines(self, tmp_path):
        first = make_api(tmp_path)
        second = make_api(tmp_path)
        second.client.get("/health")
        assert len(log_lines(second)) == 1
        assert first.log_path == second.log_path


class TestDocumentation:
    def test_openapi_lists_every_endpoint(self, api):
        spec = api.client.get("/openapi.json").json()
        assert spec["info"]["version"] == API_VERSION
        assert {"/ask", "/search", "/topics", "/health", "/ready"} <= set(spec["paths"])
        assert "/" not in spec["paths"]

    def test_error_responses_are_documented(self, api):
        spec = api.client.get("/openapi.json").json()
        for path in ("/ask", "/search"):
            responses = spec["paths"][path]["post"]["responses"]
            assert {"200", "400", "422", "500", "503"} <= set(responses)
            assert responses["422"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")

    def test_request_models_have_examples(self, api):
        schemas = api.client.get("/openapi.json").json()["components"]["schemas"]
        assert schemas["AskRequest"]["examples"][0]["question"]
        assert schemas["SearchRequest"]["examples"][0]["filters"]

    def test_interactive_docs_are_served(self, api):
        assert api.client.get("/docs").status_code == 200
        assert "swagger" in api.client.get("/docs").text.lower()


class TestSettings:
    def test_service_settings_have_defaults(self):
        s = Settings()
        assert (s.db_path, s.collection) == ("chroma_db", "arxiv_chunks")
        assert s.api_log_path == "logs/api_requests.jsonl"

    def test_service_settings_come_from_the_environment(self):
        s = load_settings({"CHROMA_DB_PATH": "/data/db", "CHROMA_COLLECTION": "papers",
                           "EMBEDDING_MODEL": "some/model", "API_LOG_PATH": "/var/log/api.jsonl"})
        assert (s.db_path, s.collection, s.embedding_model, s.api_log_path) == (
            "/data/db", "papers", "some/model", "/var/log/api.jsonl")

    def test_empty_api_log_path_switches_file_logging_off(self):
        assert load_settings({"API_LOG_PATH": ""}).api_log_path is None
        assert load_settings({}).api_log_path == "logs/api_requests.jsonl"
