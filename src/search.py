"""Semantic search: embed the query, look it up in Chroma, report latency."""

from __future__ import annotations

import time

from src.nlp_metadata import SENTIMENT_LABELS


def build_metadata_filter(topic_id: int | None = None, sentiment: str | None = None) -> dict | None:
    """Build a Chroma metadata filter from the supported NLP fields."""
    clauses = []
    if topic_id is not None:
        clauses.append({"topic_id": topic_id})
    if sentiment is not None:
        if sentiment not in SENTIMENT_LABELS:
            raise ValueError(f"sentiment must be one of {SENTIMENT_LABELS}")
        clauses.append({"sentiment_label": sentiment})
    if len(clauses) > 1:
        return {"$and": clauses}
    return clauses[0] if clauses else None


def semantic_search(query: str, embedder, store, k: int = 5, where: dict | None = None) -> dict:
    """Return {"results": [...], "timing": {...}} for a text query.

    Raises ValueError for an empty query or k <= 0. Queries longer than the
    model's limit (256 tokens for MiniLM) are silently truncated by the model.
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be positive")

    t0 = time.perf_counter()
    query_vector = embedder.embed([query.strip()])[0]
    t1 = time.perf_counter()
    results = store.query(query_vector, k=k, where=where)
    t2 = time.perf_counter()

    return {
        "results": results,
        "timing": {
            "embed_ms": (t1 - t0) * 1000,
            "search_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
        },
    }
