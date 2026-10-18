"""Tests for src/vector_store.py, src/search.py and the Week 2 scripts.

A hashing embedder stands in for the real model: texts that share words get
similar vectors, so search behaviour can be tested without downloading
anything.
"""

import sys
import uuid
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.chunking import chunk_document  # noqa: E402
from src.search import semantic_search  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

import benchmark_retrieval  # noqa: E402
import build_index  # noqa: E402


class HashEmbedder:
    dim = 256
    model_name = "hash-test"
    device = "cpu"
    batch_size = 64

    def embed(self, texts, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                out[row, zlib.crc32(word.encode()) % self.dim] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def embed_timed(self, texts, show_progress=False):
        vectors = self.embed(texts)
        return vectors, {
            "model": self.model_name, "device": self.device, "batch_size": 64,
            "dimension": self.dim, "num_texts": len(texts), "total_seconds": 0.01,
            "texts_per_second": 1000.0, "ms_per_text": 1.0,
        }

    def warm_up(self):
        pass


DOCS = {
    "d1": ("Attention", "transformer attention mechanism for language translation. " * 4),
    "d2": ("Vision", "convolutional networks for image classification and object detection. " * 4),
    "d3": ("RL", "reinforcement learning agents maximize reward in game environments. " * 4),
}


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def store():
    return VectorStore(path=None, collection_name=f"test_{uuid.uuid4().hex}")


@pytest.fixture
def filled_store(store, embedder):
    chunks = []
    for doc_id, (title, text) in DOCS.items():
        chunks.extend(chunk_document(doc_id, text, {"title": title, "source": "arxiv"}, 300, 50))
    store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
    return store


class TestIngestion:
    def test_adds_chunks_and_counts(self, store, embedder):
        chunks = chunk_document("d1", DOCS["d1"][1], {"title": "x"}, 300, 50)
        written = store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
        assert written == len(chunks)
        assert store.count() == len(chunks)

    def test_reingesting_does_not_duplicate(self, store, embedder):
        chunks = chunk_document("d1", DOCS["d1"][1], {"title": "x"}, 300, 50)
        vectors = embedder.embed([c["text"] for c in chunks])
        store.add_chunks(chunks, vectors)
        store.add_chunks(chunks, vectors)
        assert store.count() == len(chunks)

    def test_duplicate_ids_in_one_batch_are_handled(self, store, embedder):
        chunks = [
            {"chunk_id": "a", "text": "first version", "metadata": {}},
            {"chunk_id": "a", "text": "second version", "metadata": {}},
        ]
        store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
        assert store.count() == 1

    def test_empty_text_chunks_are_skipped(self, store, embedder):
        chunks = [
            {"chunk_id": "a", "text": "real content", "metadata": {}},
            {"chunk_id": "b", "text": "   ", "metadata": {}},
        ]
        written = store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
        assert written == 1

    def test_empty_input_writes_nothing(self, store):
        assert store.add_chunks([], np.empty((0, 256))) == 0
        assert store.count() == 0

    def test_mismatched_lengths_raise(self, store, embedder):
        chunks = [{"chunk_id": "a", "text": "hello", "metadata": {}}]
        with pytest.raises(ValueError):
            store.add_chunks(chunks, embedder.embed(["hello", "world"]))

    def test_nan_embeddings_raise(self, store):
        chunks = [{"chunk_id": "a", "text": "hello", "metadata": {}}]
        with pytest.raises(ValueError):
            store.add_chunks(chunks, np.array([[np.nan] * 256]))

    def test_none_and_odd_metadata_values_are_accepted(self, store, embedder):
        chunks = [{"chunk_id": "a", "text": "hello world",
                   "metadata": {"title": None, "tags": ["x", "y"], "n": 3}}]
        store.add_chunks(chunks, embedder.embed(["hello world"]))
        hit = store.query(embedder.embed(["hello world"])[0], k=1)[0]
        assert hit["metadata"]["n"] == 3
        assert "title" not in hit["metadata"]

    def test_large_batch_is_split(self, store, embedder):
        chunks = [{"chunk_id": f"c{i}", "text": f"chunk number {i}", "metadata": {}} for i in range(250)]
        written = store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]), batch_size=100)
        assert written == 250
        assert store.count() == 250

    def test_reset_empties_collection(self, filled_store):
        assert filled_store.count() > 0
        filled_store.reset()
        assert filled_store.count() == 0


class TestPersistence:
    def test_data_survives_reopening_the_database(self, tmp_path, embedder):
        db = str(tmp_path / "db")
        first = VectorStore(path=db, collection_name="persist_test")
        chunks = chunk_document("d1", DOCS["d1"][1], {"title": "x"}, 300, 50)
        first.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
        expected = first.count()
        del first

        reopened = VectorStore(path=db, collection_name="persist_test")
        assert reopened.count() == expected
        out = semantic_search("transformer attention", embedder, reopened, k=1)
        assert out["results"][0]["metadata"]["doc_id"] == "d1"


class TestSearch:
    def test_finds_the_relevant_document(self, filled_store, embedder):
        out = semantic_search("transformer attention", embedder, filled_store, k=3)
        assert out["results"][0]["metadata"]["doc_id"] == "d1"

    def test_results_are_sorted_best_first(self, filled_store, embedder):
        scores = [r["score"] for r in semantic_search("image classification", embedder, filled_store, k=5)["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_filter(self, filled_store, embedder):
        out = semantic_search("attention", embedder, filled_store, k=5, where={"doc_id": "d3"})
        assert out["results"]
        assert all(r["metadata"]["doc_id"] == "d3" for r in out["results"])

    def test_filter_with_no_matches_returns_empty(self, filled_store, embedder):
        out = semantic_search("attention", embedder, filled_store, k=5, where={"doc_id": "nope"})
        assert out["results"] == []

    def test_k_larger_than_collection_is_clipped(self, filled_store, embedder):
        out = semantic_search("attention", embedder, filled_store, k=10_000)
        assert len(out["results"]) == filled_store.count()

    def test_empty_collection_returns_empty_list(self, store, embedder):
        assert semantic_search("anything", embedder, store)["results"] == []

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_query_raises(self, bad, filled_store, embedder):
        with pytest.raises(ValueError):
            semantic_search(bad, embedder, filled_store)

    def test_invalid_k_raises(self, filled_store, embedder):
        with pytest.raises(ValueError):
            semantic_search("attention", embedder, filled_store, k=0)

    def test_timing_is_reported(self, filled_store, embedder):
        timing = semantic_search("attention", embedder, filled_store)["timing"]
        assert set(timing) == {"embed_ms", "search_ms", "total_ms"}
        assert timing["total_ms"] >= timing["search_ms"] >= 0


class TestScripts:
    @pytest.fixture
    def corpus(self):
        return pd.DataFrame({
            "doc_id": list(DOCS),
            "source": "arxiv",
            "title": [t for t, _ in DOCS.values()],
            "text": [x for _, x in DOCS.values()],
            "categories": "cs.AI",
            "published": "2026-10-01",
        })

    def test_build_index_end_to_end(self, corpus, store, embedder):
        report = build_index.build_index(corpus, embedder, store, chunk_size=300, chunk_overlap=50)
        assert report["chunking"]["num_documents"] == 3
        assert report["ingestion"]["collection_count"] == report["chunking"]["num_chunks"]
        assert "| Chunks |" in build_index.markdown_summary(report)

    def test_build_index_rejects_empty_corpus(self, corpus, store, embedder):
        with pytest.raises(ValueError):
            build_index.build_index(corpus.iloc[0:0], embedder, store, 300, 50)

    def test_benchmark_runs_and_summarises(self, filled_store, embedder):
        queries = [("topic", "transformer attention"), ("nonsense", "qzxv blorf")]
        report = benchmark_retrieval.run_benchmark(embedder, filled_store, queries, repeats=3, k=3)
        assert report["overall"]["searches"] == 6
        assert report["overall"]["total_ms_p95"] >= report["overall"]["total_ms_p50"]
        assert len(report["per_query"]) == 2
        assert "| Type | Query |" in benchmark_retrieval.markdown_report(report)
