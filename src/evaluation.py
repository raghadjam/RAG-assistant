"""Evaluation helpers for retrieval and generation.

Ranking metrics are computed on documents: a retrieved chunk counts as relevant when the
paper it comes from is in the query's relevant set.

Definitions used throughout (K is the number of chunks the search returns):

    Precision@K  relevant chunks among the top K, divided by K
    Recall@K     distinct relevant papers found in the top K, divided by min(number of
                 relevant papers, K). The cap makes 1.0 reachable when a query has
                 hundreds of relevant papers but only K slots. With a single relevant
                 paper this is simply "was it found in the top K" (the hit rate).
    MRR          mean of 1 / rank of the first relevant chunk (0 if none was retrieved)
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from src.search import build_metadata_filter, semantic_search
from src.text_utils import strip_latex

DEFAULT_KS = (1, 3, 5, 10)
KNOWN_ITEM_TYPES = ("title", "sentence")

# Topics that recent AI / ML / statistics abstracts talk about. Edit freely: a phrase is kept only
# if enough papers in the corpus mention it (see min_relevant).
DEFAULT_CONCEPTS = [
    "large language model", "reinforcement learning", "diffusion model", "graph neural network",
    "federated learning", "retrieval-augmented generation", "chain-of-thought", "knowledge distillation",
    "contrastive learning", "fine-tuning", "vision-language model", "hallucination", "alignment",
    "bayesian optimization", "causal inference", "time series forecasting", "recommender system",
    "reward model", "mixture of experts", "quantization", "prompt engineering", "adversarial robustness",
    "differential privacy", "multi-agent", "tool use", "code generation", "speech recognition",
    "anomaly detection", "uncertainty estimation", "benchmark dataset",
]

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their this to was were "
    "which with we can not these those they them than then also such into using used use based".split())


# --------------------------------------------------------------------------- #
# Ranking metrics
# --------------------------------------------------------------------------- #
def precision_at_k(ranked_docs: list, relevant: set, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(d in relevant for d in ranked_docs[:k]) / k


def recall_at_k(ranked_docs: list, relevant: set, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("a query needs at least one relevant document")
    found = {d for d in ranked_docs[:k] if d in relevant}
    return len(found) / min(len(relevant), k)


def reciprocal_rank(ranked_docs: list, relevant: set) -> float:
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_ranking(ranked_docs: list, relevant: set, ks=DEFAULT_KS) -> dict:
    """All metrics for one query. ranked_docs holds one doc id per retrieved chunk, best first."""
    return {
        "precision": {k: precision_at_k(ranked_docs, relevant, k) for k in ks},
        "recall": {k: recall_at_k(ranked_docs, relevant, k) for k in ks},
        "rr": reciprocal_rank(ranked_docs, relevant),
    }


def latency_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {"mean": round(float(np.mean(values)), 1), "p50": round(float(np.percentile(values, 50)), 1),
            "p95": round(float(np.percentile(values, 95)), 1)}


def _mean(values) -> float:
    values = list(values)
    return round(float(np.mean(values)), 3) if values else 0.0


def summarize(records: list[dict], ks, get=lambda r: r["metrics"]) -> dict:
    """Average the per-query metrics. get picks the metrics dict out of a record."""
    return {
        "n": len(records),
        "precision": {k: _mean(get(r)["precision"][k] for r in records) for k in ks},
        "recall": {k: _mean(get(r)["recall"][k] for r in records) for k in ks},
        "mrr": _mean(get(r)["rr"] for r in records),
    }


# --------------------------------------------------------------------------- #
# Retrieval test set
# --------------------------------------------------------------------------- #
def phrase_regex(phrase: str) -> re.Pattern:
    """Match a phrase however it is written: plural endings, hyphen or space between words."""
    words = [re.escape(w) for w in re.split(r"[\s-]+", phrase.strip()) if w]
    return re.compile(r"\b" + r"[\s-]+".join(words) + r"s?\b", re.IGNORECASE)


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join((text or "").split()))
    return [s.strip() for s in sentences if s.strip()]


def abstract_of(title: str, text: str) -> str:
    """The corpus text is the title followed by the abstract."""
    text = (text or "").strip()
    if title and text.startswith(title):
        text = text[len(title):]
    return text.strip()


def _stratified_order(df: pd.DataFrame, rng: np.random.Generator, topic_col: str) -> list[int]:
    """Row positions in a shuffled order that cycles through the topics, so all are represented."""
    if topic_col not in df.columns:
        return list(rng.permutation(len(df)))
    groups = [list(rng.permutation(np.flatnonzero((df[topic_col] == t).to_numpy())))
              for t in sorted(df[topic_col].dropna().unique())]
    order = []
    while any(groups):
        for group in groups:
            if group:
                order.append(group.pop())
    return order


def _clean_query(text: str) -> str:
    return " ".join(strip_latex(text).split())


def build_retrieval_testset(
    corpus: pd.DataFrame,
    concepts: list[str] | None = None,
    n_title: int = 40,
    n_sentence: int = 40,
    min_relevant: int = 10,
    seed: int = 42,
    topic_col: str = "topic_id",
) -> list[dict]:
    """Build labelled queries from the corpus itself, with no manual labelling.

    title     the paper's title as the query; relevant = that paper (easy)
    sentence  one sentence from the middle of its abstract as the query; relevant = that paper (harder)
    concept   a topic phrase such as "federated learning"; relevant = every paper that mentions it

    Known-item queries (title, sentence) are spread evenly over the topics.
    """
    for column in ("doc_id", "title", "text"):
        if column not in corpus.columns:
            raise ValueError(f"corpus needs a '{column}' column")
    rng = np.random.default_rng(seed)
    corpus = corpus.reset_index(drop=True)

    usable = corpus[[len(_clean_query(t)) >= 15 and len(abstract_of(t, x)) >= 200
                     for t, x in zip(corpus["title"], corpus["text"])]]
    order = _stratified_order(usable.reset_index(drop=True), rng, topic_col)
    usable = usable.reset_index(drop=True)

    def topic_of(row):
        if topic_col not in usable.columns or pd.isna(row[topic_col]):
            return None
        return int(row[topic_col])

    items: list[dict] = []
    used: set[int] = set()
    n_made = 0
    for pos in order:                                            # title queries first
        if n_made >= n_title:
            break
        row = usable.iloc[pos]
        items.append({"id": f"title-{n_made:03d}", "type": "title", "query": _clean_query(row["title"]),
                      "relevant_doc_ids": [row["doc_id"]], "topic_id": topic_of(row)})
        used.add(pos)
        n_made += 1

    n_made = 0
    for pos in order:                                            # sentence queries from other papers
        if n_made >= n_sentence:
            break
        if pos in used:
            continue
        row = usable.iloc[pos]
        sentences = split_sentences(abstract_of(row["title"], row["text"]))
        cleaned = [_clean_query(s) for s in sentences[1:]]       # skip the opening sentence
        candidates = [s for s in cleaned if 60 <= len(s) <= 300]
        if not candidates:
            continue
        items.append({"id": f"sentence-{n_made:03d}", "type": "sentence",
                      "query": candidates[int(rng.integers(len(candidates)))],
                      "relevant_doc_ids": [row["doc_id"]], "topic_id": topic_of(row)})
        n_made += 1

    n_made = 0
    for phrase in (DEFAULT_CONCEPTS if concepts is None else concepts):
        rx = phrase_regex(phrase)
        relevant = sorted(corpus.loc[[bool(rx.search(t or "")) for t in corpus["text"]], "doc_id"])
        if len(relevant) >= min_relevant:
            items.append({"id": f"concept-{n_made:03d}", "type": "concept", "query": phrase,
                          "relevant_doc_ids": relevant, "topic_id": None})
            n_made += 1
    return items


def save_testset(items: list[dict], path) -> None:
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")


def load_testset(path) -> list[dict]:
    import json
    from pathlib import Path

    items = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    for item in items:
        if not item.get("query") or not item.get("relevant_doc_ids"):
            raise ValueError(f"test set item {item.get('id')!r} needs a query and relevant_doc_ids")
    return items


# --------------------------------------------------------------------------- #
# Running retrieval on a test set
# --------------------------------------------------------------------------- #
def _ranked_docs(hits: list[dict]) -> list:
    return [(h.get("metadata") or {}).get("doc_id") for h in hits]


def run_retrieval_evaluation(testset: list[dict], embedder, store, ks=DEFAULT_KS, filtered: bool = True) -> dict:
    """Search every query and score the ranking.

    With filtered=True each known-item query is also searched with a filter on its own paper's
    topic. That shows what the metadata filter costs or gains when the topic is known.
    """
    ks = tuple(sorted(ks))
    embedder.embed(["warm up"])
    records = []
    for item in testset:
        relevant = set(item["relevant_doc_ids"])
        out = semantic_search(item["query"], embedder, store, k=max(ks))
        record = {"id": item["id"], "type": item["type"], "query": item["query"],
                  "relevant_count": len(relevant), "latency_ms": out["timing"]["total_ms"],
                  "metrics": evaluate_ranking(_ranked_docs(out["results"]), relevant, ks)}
        if filtered and item["type"] in KNOWN_ITEM_TYPES and item.get("topic_id") is not None:
            where = build_metadata_filter(topic_id=item["topic_id"])
            out_f = semantic_search(item["query"], embedder, store, k=max(ks), where=where)
            record["filtered"] = {"latency_ms": out_f["timing"]["total_ms"],
                                  "metrics": evaluate_ranking(_ranked_docs(out_f["results"]), relevant, ks)}
        records.append(record)

    types = [t for t in ("title", "sentence", "concept") if any(r["type"] == t for r in records)]
    by_type = {}
    for t in types:
        group = [r for r in records if r["type"] == t]
        entry = {**summarize(group, ks), "latency_ms": latency_stats([r["latency_ms"] for r in group]),
                 "relevant_papers_mean": round(float(np.mean([r["relevant_count"] for r in group])), 1)}
        with_filter = [r for r in group if "filtered" in r]
        if with_filter:
            entry["same_queries_unfiltered"] = summarize(with_filter, ks)
            entry["with_topic_filter"] = {**summarize(with_filter, ks, get=lambda r: r["filtered"]["metrics"]),
                                          "latency_ms": latency_stats(
                                              [r["filtered"]["latency_ms"] for r in with_filter])}
            entry["same_queries_unfiltered"]["latency_ms"] = latency_stats([r["latency_ms"] for r in with_filter])
        by_type[t] = entry

    max_k = max(ks)
    misses = [{"id": r["id"], "type": r["type"], "query": r["query"], "relevant_papers": r["relevant_count"]}
              for r in records if r["metrics"]["recall"][max_k] == 0]
    return {
        "ks": list(ks),
        "n_queries": len(records),
        "index_chunks": store.count(),
        "overall": {**summarize(records, ks), "latency_ms": latency_stats([r["latency_ms"] for r in records])},
        "by_type": by_type,
        "not_found_in_top_k": {"k": max_k, "count": len(misses), "examples": misses[:15]},
        "per_query": [{"id": r["id"], "type": r["type"], "query": r["query"][:120],
                       "relevant_papers": r["relevant_count"], "reciprocal_rank": round(r["metrics"]["rr"], 3),
                       "latency_ms": round(r["latency_ms"], 1)} for r in records],
    }


# --------------------------------------------------------------------------- #
# Answer quality helpers
# --------------------------------------------------------------------------- #
CITATION_RE = re.compile(r"\[\d+\]")


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in STOPWORDS and len(w) > 1]


def rouge1_f1(candidate: str, reference: str) -> float:
    """Unigram overlap F1 between an answer and a reference text (citations ignored)."""
    cand = Counter(tokens(CITATION_RE.sub(" ", candidate)))
    ref = Counter(tokens(reference))
    overlap = sum((cand & ref).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(cand.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else 0.0


def citation_coverage(answer: str) -> float:
    """Share of the answer's sentences that carry a [n] citation."""
    sentences = split_sentences(answer)
    return sum(bool(CITATION_RE.search(s)) for s in sentences) / len(sentences) if sentences else 0.0


def first_sentences(text: str, n: int = 2) -> str:
    return " ".join(split_sentences(_clean_query(text))[:n])


def build_doc_questions(corpus: pd.DataFrame, n: int, seed: int = 42, topic_col: str = "topic_id") -> list[dict]:
    """Questions about specific papers, each with a reference answer taken from the abstract.

    The reference is the abstract's first two sentences, which is how the paper describes itself.
    """
    rng = np.random.default_rng(seed)
    corpus = corpus.reset_index(drop=True)
    keep = [i for i, (t, x) in enumerate(zip(corpus["title"], corpus["text"]))
            if 15 <= len(_clean_query(t)) <= 200 and len(abstract_of(t, x)) >= 300]
    pool = corpus.iloc[keep].reset_index(drop=True)
    questions = []
    for pos in _stratified_order(pool, rng, topic_col)[:n]:
        row = pool.iloc[pos]
        questions.append({
            "id": f"doc-{len(questions):02d}", "category": "doc",
            "question": f'What does the paper titled "{_clean_query(row["title"])}" propose or find?',
            "reference": first_sentences(abstract_of(row["title"], row["text"])),
            "doc_id": row["doc_id"],
        })
    return questions
