"""Build the labelled queries used by the retrieval evaluation.

    python scripts/build_retrieval_testset.py

Needs no manual labelling. The queries come from the corpus itself, so the correct answers are known:

    title     a paper's title; the relevant paper is that paper
    sentence  a sentence from the middle of its abstract; the relevant paper is that paper
    concept   a topic phrase such as "federated learning"; every paper that mentions it is relevant

Writes data/eval/retrieval_testset.jsonl (kept in git, so the evaluation can be rerun on exactly
the same queries). Use --force to rebuild it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import DEFAULT_CONCEPTS, build_retrieval_testset, save_testset  # noqa: E402


def read_concepts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the retrieval test set.")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/enriched_corpus.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/retrieval_testset.jsonl"))
    parser.add_argument("--n-title", type=int, default=40)
    parser.add_argument("--n-sentence", type=int, default=40)
    parser.add_argument("--min-relevant", type=int, default=10,
                        help="a concept phrase is kept only if at least this many papers mention it")
    parser.add_argument("--concepts-file", type=Path, help="text file with one concept phrase per line")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="overwrite an existing test set")
    args = parser.parse_args()

    if not args.corpus.exists():
        print(f"ERROR: {args.corpus} not found. Run scripts/enrich_corpus.py first.")
        return 1
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists, so it was not overwritten (results stay comparable). Use --force to rebuild.")
        return 1

    concepts = read_concepts(args.concepts_file) if args.concepts_file else list(DEFAULT_CONCEPTS)
    corpus = pd.read_parquet(args.corpus)
    items = build_retrieval_testset(corpus, concepts, args.n_title, args.n_sentence, args.min_relevant, args.seed)
    save_testset(items, args.out)

    counts = Counter(i["type"] for i in items)
    print(f"Wrote {len(items)} queries to {args.out}: {dict(counts)}")
    kept = {i["query"]: len(i["relevant_doc_ids"]) for i in items if i["type"] == "concept"}
    print("\nConcept queries (papers mentioning the phrase):")
    for phrase, n in kept.items():
        print(f"  {phrase:<32} {n}")
    dropped = [c for c in concepts if c not in kept]
    if dropped:
        print(f"\nDropped (fewer than {args.min_relevant} papers mention them): {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
