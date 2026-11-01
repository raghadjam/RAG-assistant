import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nlp_metadata import (  # noqa: E402
    enrich_corpus,
    evaluate_sentiment,
    load_manual_labels,
    score_sentiment,
    sentiment_label,
)
from src.config import Settings  # noqa: E402
from src.rag import RAGPipeline, _source  # noqa: E402
from src.search import build_metadata_filter, semantic_search  # noqa: E402
from scripts.build_index import build_index  # noqa: E402


class TestSentiment:
    def test_vader_thresholds(self):
        assert sentiment_label(0.05) == "positive"
        assert sentiment_label(-0.05) == "negative"
        assert sentiment_label(0.0) == "neutral"

    def test_manual_sentiment_benchmark_is_evaluated(self):
        texts, expected = load_manual_labels(ROOT / "tests" / "fixtures" / "sentiment_labels.csv")
        result = evaluate_sentiment(texts, expected)
        assert result["examples"] == 18
        assert result["accuracy"] == 0.944
        assert result["confusion_matrix"]["negative"]["neutral"] == 1
        assert sum(result["per_class"][label]["support"] for label in result["per_class"]) == 18

    def test_invalid_manual_label_is_reported(self):
        with pytest.raises(ValueError, match="unsupported sentiment"):
            evaluate_sentiment(["example"], ["mixed"])

    def test_empty_and_short_texts_get_a_sentiment(self):
        scored = score_sentiment(["", "LLM"])
        assert len(scored) == 2
        assert all(item["sentiment_label"] in {"negative", "neutral", "positive"} for item in scored)


class TestTopics:
    def test_lda_assigns_topics_and_leaves_short_documents_unassigned(self):
        corpus = pd.DataFrame({
            "doc_id": ["ml1", "ml2", "vision1", "short"],
            "title": ["Neural networks", "Deep learning", "Image processing", "Short"],
            "text": [
                "neural networks learn representations with gradient descent and training data " * 3,
                "deep learning optimizes neural models using backpropagation and training data " * 3,
                "image processing detects visual objects using convolutional features and pixels " * 3,
                "LLM",
            ],
        })
        enriched, report = enrich_corpus(corpus, n_topics=2, min_df=1, min_topic_chars=50, max_iter=3)
        assert report["status"] == "fit"
        assert report["fitted_topics"] == 2
        assert enriched.loc[3, "topic_id"] == -1
        assert enriched.loc[3, "topic_name"] == "unassigned"
        assert len(report["topics"]) == 2
        assert all(topic["review_examples"] for topic in report["topics"])
        assert enriched["sentiment_label"].notna().all()
        assert enriched["sentiment_score"].between(-1, 1).all()

    def test_tiny_corpus_is_reported_without_fitting_topics(self):
        enriched, report = enrich_corpus(
            pd.DataFrame({"text": ["a short document"]}),
            n_topics=2,
            min_topic_chars=50,
        )
        assert report["status"] == "not_fit"
        assert enriched.loc[0, "topic_id"] == -1
        assert report["documents_unassigned"] == 1

    def test_missing_text_column_is_rejected(self):
        with pytest.raises(ValueError, match="text"):
            enrich_corpus(pd.DataFrame({"body": ["text"]}))


class TestMetadataFilters:
    def test_topic_and_sentiment_are_combined(self):
        assert build_metadata_filter(2, "positive") == {
            "$and": [{"topic_id": 2}, {"sentiment_label": "positive"}],
        }

    def test_each_filter_can_be_used_independently(self):
        assert build_metadata_filter(topic_id=-1) == {"topic_id": -1}
        assert build_metadata_filter(sentiment="neutral") == {"sentiment_label": "neutral"}
        assert build_metadata_filter() is None

    def test_invalid_sentiment_is_rejected(self):
        with pytest.raises(ValueError, match="sentiment"):
            build_metadata_filter(sentiment="mixed")

    def test_metadata_filter_is_passed_to_vector_search(self):
        class Embedder:
            def embed(self, texts):
                return [[0.1, 0.2]]

        class Store:
            where = None

            def query(self, query_embedding, k, where):
                self.where = where
                return []

        store = Store()
        where = build_metadata_filter(topic_id=3, sentiment="neutral")
        semantic_search("paper query", Embedder(), store, where=where)
        assert store.where == where

    def test_rag_pipeline_passes_metadata_filter_to_retrieval(self):
        class Embedder:
            def embed(self, texts):
                return [[0.1, 0.2]]

        class Store:
            where = None

            def count(self):
                return 1

            def query(self, query_embedding, k, where):
                self.where = where
                return []

        store = Store()
        where = build_metadata_filter(topic_id=1)
        settings = Settings(min_score=0.1, min_chunk_score=0.05, log_path=None)
        result = RAGPipeline(Embedder(), store, llm=None, settings=settings).ask(
            "a topic-specific query",
            where=where,
        )
        assert result.status == "out_of_domain"
        assert store.where == where

    def test_rag_sources_expose_nlp_metadata(self):
        source = _source(1, {
            "metadata": {
                "doc_id": "paper-1",
                "topic_id": 3,
                "topic_name": "retrieval, ranking",
                "sentiment_label": "neutral",
            },
            "score": 0.8,
        })
        assert source["topic_id"] == 3
        assert source["topic_name"] == "retrieval, ranking"
        assert source["sentiment_label"] == "neutral"


def test_index_builder_copies_nlp_metadata_to_chunks():
    class Embedder:
        def warm_up(self):
            pass

        def embed_timed(self, texts, show_progress=False):
            return np.ones((len(texts), 2), dtype=np.float32), {
                "model": "test", "dimension": 2, "device": "cpu",
                "total_seconds": 0.0, "texts_per_second": 0, "ms_per_text": 0,
            }

    class Store:
        def add_chunks(self, chunks, vectors):
            self.chunks = chunks
            return len(chunks)

        def count(self):
            return len(self.chunks)

    corpus = pd.DataFrame([{
        "doc_id": "paper-1",
        "title": "Transformer methods",
        "text": "Transformer architecture and attention methods are evaluated. " * 4,
        "source": "arxiv",
        "categories": "cs.AI",
        "published": "2026-01-01",
        "topic_id": 2,
        "topic_name": "transformer, attention",
        "sentiment_label": "neutral",
        "sentiment_score": 0.01,
    }])
    store = Store()
    build_index(corpus, Embedder(), store, chunk_size=300, chunk_overlap=50)
    metadata = store.chunks[0]["metadata"]
    assert metadata["topic_id"] == 2
    assert metadata["topic_name"] == "transformer, attention"
    assert metadata["sentiment_label"] == "neutral"
    assert metadata["sentiment_score"] == 0.01
