"""Add LDA topics and VADER sentiment to the cleaned corpus.

    python scripts/enrich_corpus.py
    python scripts/enrich_corpus.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nlp_metadata import enrich_corpus, evaluate_sentiment, load_manual_labels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich corpus documents with topics and sentiment.")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/clean_corpus.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/enriched_corpus.parquet"))
    parser.add_argument("--report", type=Path, default=Path("reports/nlp_metadata.json"))
    parser.add_argument("--labels", type=Path, default=Path("tests/fixtures/sentiment_labels.csv"))
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--min-topic-chars", type=int, default=80)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only enrich the first N documents")
    args = parser.parse_args()

    if not args.corpus.exists():
        parser.error(f"{args.corpus} not found. Run scripts/build_corpus.py first.")
    if not args.labels.exists():
        parser.error(f"manual sentiment labels not found: {args.labels}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    corpus = pd.read_parquet(args.corpus)
    if args.limit is not None:
        corpus = corpus.head(args.limit).copy()
    enriched, report = enrich_corpus(
        corpus,
        n_topics=args.topics,
        min_topic_chars=args.min_topic_chars,
        min_df=args.min_df,
    )
    labeled_texts, expected = load_manual_labels(args.labels)
    report["sentiment_evaluation"] = {
        "dataset": str(args.labels),
        "scope": "small manually labeled examples; not a representative corpus-wide accuracy estimate",
        **evaluate_sentiment(labeled_texts, expected),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(args.output, index=False)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "documents": len(enriched),
        "topic_status": report["status"],
        "topics": report.get("fitted_topics", 0),
        "sentiment_evaluation_examples": report["sentiment_evaluation"]["examples"],
        "sentiment_evaluation_accuracy": report["sentiment_evaluation"]["accuracy"],
        "output": str(args.output),
        "report": str(args.report),
    }, indent=2))
    print("Inspect topic terms and review_examples in the report before treating topics as categories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
