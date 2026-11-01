"""Corpus-level topic and sentiment metadata for the arXiv RAG pipeline."""

from __future__ import annotations

import pandas as pd

TOPIC_MIN_CHARS = 80
TOPIC_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z0-9_+-]{2,}\b"
SENTIMENT_LABELS = ("negative", "neutral", "positive")
TOPIC_DOMAIN_STOP_WORDS = frozenset({
    "approach", "approaches", "based", "demonstrate", "existing", "framework",
    "frameworks", "learning", "method", "methods", "model", "models", "new",
    "paper", "papers", "performance", "proposed", "results", "show", "study",
    "system", "systems", "task", "tasks", "use", "used", "using", "work",
})


def sentiment_label(compound: float) -> str:
    """Map VADER's compound score to its conventional polarity thresholds."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def score_sentiment(texts: list[str]) -> list[dict]:
    """Return VADER labels and compound scores, including for short texts."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scored = []
    for text in texts:
        compound = float(analyzer.polarity_scores(text or "")["compound"])
        scored.append({"sentiment_label": sentiment_label(compound), "sentiment_score": compound})
    return scored


def evaluate_sentiment(texts: list[str], expected: list[str]) -> dict:
    """Evaluate polarity on explicit human labels; return accuracy and per-class metrics."""
    if len(texts) != len(expected):
        raise ValueError("texts and expected labels must have the same length")
    if not expected:
        raise ValueError("at least one manually labeled example is required")
    invalid = sorted(set(expected) - set(SENTIMENT_LABELS))
    if invalid:
        raise ValueError(f"unsupported sentiment labels: {invalid}")

    predictions = [item["sentiment_label"] for item in score_sentiment(texts)]
    confusion = {label: {predicted: 0 for predicted in SENTIMENT_LABELS} for label in SENTIMENT_LABELS}
    for actual, predicted in zip(expected, predictions):
        confusion[actual][predicted] += 1
    per_class = {}
    for label in SENTIMENT_LABELS:
        true_positive = confusion[label][label]
        actual_count = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in SENTIMENT_LABELS)
        per_class[label] = {
            "precision": round(true_positive / predicted_count, 3) if predicted_count else 0.0,
            "recall": round(true_positive / actual_count, 3) if actual_count else 0.0,
            "support": actual_count,
        }
    return {
        "examples": len(expected),
        "accuracy": round(sum(a == p for a, p in zip(expected, predictions)) / len(expected), 3),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "predictions": predictions,
    }


def enrich_corpus(
    corpus: pd.DataFrame,
    *,
    n_topics: int = 8,
    min_topic_chars: int = TOPIC_MIN_CHARS,
    min_df: int = 2,
    random_state: int = 42,
    max_iter: int = 10,
    representative_count: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Add LDA topic and VADER sentiment fields to each corpus document.

    Short documents remain in the output with topic_id=-1 and topic_name
    ``unassigned``. Technical terms and common hyphenated forms are preserved
    by the vectorizer token pattern rather than normalized away.
    """
    if (n_topics < 1 or min_topic_chars < 1 or min_df < 1 or max_iter < 1
            or representative_count < 0):
        raise ValueError(
            "n_topics, min_topic_chars, min_df and max_iter must be positive; "
            "representative_count cannot be negative"
        )
    if "text" not in corpus.columns:
        raise ValueError("corpus must contain a 'text' column")
    if corpus.empty:
        raise ValueError("corpus is empty")

    enriched = corpus.copy().reset_index(drop=True)
    texts = enriched["text"].fillna("").astype(str).tolist()
    sentiment = score_sentiment(texts)
    enriched["sentiment_label"] = [s["sentiment_label"] for s in sentiment]
    enriched["sentiment_score"] = [s["sentiment_score"] for s in sentiment]

    eligible = [i for i, text in enumerate(texts) if len(text.strip()) >= min_topic_chars]
    enriched["topic_id"] = -1
    enriched["topic_name"] = "unassigned"
    topic_report: dict = {
        "method": "Latent Dirichlet Allocation (scikit-learn)",
        "requested_topics": n_topics,
        "min_topic_chars": min_topic_chars,
        "min_document_frequency": min_df,
        "domain_stop_words": sorted(TOPIC_DOMAIN_STOP_WORDS),
        "documents_total": len(texts),
        "documents_eligible": len(eligible),
        "documents_unassigned": len(texts),
        "topics": [],
    }
    if len(eligible) < 2:
        topic_report["status"] = "not_fit"
        topic_report["reason"] = "LDA requires at least two documents meeting min_topic_chars."
        return enriched, _with_distributions(enriched, topic_report)

    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

    vectorizer = CountVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | TOPIC_DOMAIN_STOP_WORDS),
        token_pattern=TOPIC_TOKEN_PATTERN,
        min_df=min_df,
        max_df=0.95,
    )
    try:
        matrix = vectorizer.fit_transform([texts[i] for i in eligible])
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower() and "no terms remain" not in str(exc).lower():
            raise
        topic_report["status"] = "not_fit"
        topic_report["reason"] = (
            "No usable topic vocabulary remained after stop-word and document-frequency filtering. "
            "Inspect jargon or lower min_df."
        )
        return enriched, _with_distributions(enriched, topic_report)

    active_positions = [position for position in range(matrix.shape[0]) if matrix[position].nnz]
    if len(active_positions) < 2:
        topic_report["status"] = "not_fit"
        topic_report["reason"] = (
            "Fewer than two documents contain usable topic terms. Inspect jargon or lower min_df."
        )
        return enriched, _with_distributions(enriched, topic_report)

    active_documents = [eligible[position] for position in active_positions]
    topic_report["documents_topic_eligible"] = len(active_documents)
    topic_count = min(n_topics, len(active_documents), matrix.shape[1])
    if topic_count < 1:
        topic_report["status"] = "not_fit"
        topic_report["reason"] = "No terms are available to fit an LDA topic."
        return enriched, _with_distributions(enriched, topic_report)

    model = LatentDirichletAllocation(
        n_components=topic_count,
        max_iter=max_iter,
        learning_method="batch",
        random_state=random_state,
    )
    distributions = model.fit_transform(matrix[active_positions])
    feature_names = vectorizer.get_feature_names_out()
    top_terms = []
    for topic_id, weights in enumerate(model.components_):
        terms = [feature_names[i] for i in weights.argsort()[-8:][::-1]]
        topic_name = ", ".join(terms[:4])
        top_terms.append((topic_id, topic_name))
        topic_report["topics"].append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "top_terms": terms,
            "document_count": 0,
            "review_examples": [],
        })

    for row_index, distribution in zip(active_documents, distributions):
        topic_id = int(distribution.argmax())
        enriched.at[row_index, "topic_id"] = topic_id
        enriched.at[row_index, "topic_name"] = top_terms[topic_id][1]

    for topic in topic_report["topics"]:
        topic_rows = [
            (float(distributions[position, topic["topic_id"]]), row_index)
            for position, row_index in enumerate(active_documents)
            if int(distributions[position].argmax()) == topic["topic_id"]
        ]
        topic["document_count"] = len(topic_rows)
        topic_rows.sort(reverse=True)
        for _, row_index in topic_rows[:representative_count]:
            row = enriched.iloc[row_index]
            topic["review_examples"].append({
                "doc_id": str(row.get("doc_id", "")),
                "title": str(row.get("title", "")),
                "text": texts[row_index][:400],
            })

    topic_report["status"] = "fit"
    topic_report["fitted_topics"] = topic_count
    topic_report["manual_review_note"] = (
        "Review each topic's terms and examples. LDA topic labels are descriptive term lists, "
        "not verified semantic categories."
    )
    return enriched, _with_distributions(enriched, topic_report)


def _with_distributions(enriched: pd.DataFrame, report: dict) -> dict:
    report["documents_unassigned"] = int((enriched["topic_id"] == -1).sum())
    report["sentiment_distribution"] = {
        str(label): int(count)
        for label, count in enriched["sentiment_label"].value_counts().sort_index().items()
    }
    report["topic_distribution"] = {
        str(int(topic_id)): int(count)
        for topic_id, count in enriched["topic_id"].value_counts().sort_index().items()
    }
    return report


def load_manual_labels(path) -> tuple[list[str], list[str]]:
    """Read the small, human-annotated CSV used for sentiment evaluation."""
    labels = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(labels.columns):
        raise ValueError(f"manual labels must contain columns {sorted(required)}")
    if (
        labels.empty
        or labels["text"].isna().any()
        or labels["label"].isna().any()
        or labels["text"].astype(str).str.strip().eq("").any()
        or labels["label"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("manual labels must contain non-empty text and label values")
    return labels["text"].astype(str).tolist(), labels["label"].astype(str).str.strip().tolist()
