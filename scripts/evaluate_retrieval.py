"""Evaluate retrieval quality: Precision@K, Recall@K and MRR.

    python scripts/build_retrieval_testset.py     # once
    python scripts/evaluate_retrieval.py

Searches every query in data/eval/retrieval_testset.jsonl against the index and scores the ranking
against the known relevant papers. Retrieval only: no LLM calls, so it uses no API quota.
Writes reports/retrieval_evaluation.json and .md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import DEFAULT_KS, load_testset, run_retrieval_evaluation  # noqa: E402

TYPE_LABELS = {"title": "title (known paper)", "sentence": "sentence (known paper)", "concept": "concept (many papers)"}

DEFINITIONS = """\
K is the number of chunks returned. A chunk is relevant when its paper is in the query's relevant set.

- **Precision@K**: relevant chunks among the top K, divided by K.
- **Recall@K**: distinct relevant papers found in the top K, divided by min(number of relevant papers, K). For a query with one relevant paper this is the hit rate (was the paper found in the top K).
- **MRR**: mean of 1 / (rank of the first relevant chunk); 0 when none was retrieved."""


def _row(label: str, entry: dict, ks: list[int], latency: dict) -> str:
    cells = [f"{entry['precision'][k]:.2f}" for k in ks] + [f"{entry['recall'][k]:.2f}" for k in ks]
    return (f"| {label} | {entry['n']} | " + " | ".join(cells) +
            f" | {entry['mrr']:.2f} | {latency['mean']} / {latency['p95']} |")


def _header(first: str, ks: list[int]) -> list[str]:
    cols = [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks]
    return [f"| {first} | Queries | " + " | ".join(cols) + " | MRR | Latency mean / p95 (ms) |",
            "|---|---|" + "---|" * (len(cols) + 2)]


def markdown_report(report: dict) -> str:
    ks = report["ks"]
    lines = [f"{report['n_queries']} queries against an index of {report['index_chunks']:,} chunks.", "",
             DEFINITIONS, "", "### Results by query type", ""]
    lines += _header("Query type", ks)
    for t, entry in report["by_type"].items():
        lines.append(_row(TYPE_LABELS.get(t, t), entry, ks, entry["latency_ms"]))
    lines.append(_row("**all queries**", report["overall"], ks, report["overall"]["latency_ms"]))

    filtered = {t: e for t, e in report["by_type"].items() if "with_topic_filter" in e}
    if filtered:
        lines += ["", "### Searching with the paper's own topic as a filter", "",
                  "Each known-paper query is searched twice: without a filter, and restricted to the topic "
                  "of the paper it should find.", ""]
        lines += _header("Query type and setting", ks)
        for t, entry in filtered.items():
            lines.append(_row(f"{TYPE_LABELS.get(t, t)}: no filter", entry["same_queries_unfiltered"], ks,
                              entry["same_queries_unfiltered"]["latency_ms"]))
            lines.append(_row(f"{TYPE_LABELS.get(t, t)}: topic filter", entry["with_topic_filter"], ks,
                              entry["with_topic_filter"]["latency_ms"]))

    missed = report["not_found_in_top_k"]
    lines += ["", f"### Queries whose paper was not found in the top {missed['k']}", "",
              f"{missed['count']} of {report['n_queries']} queries."]
    if missed["examples"]:
        lines += ["", "| Type | Query | Relevant papers |", "|---|---|---|"]
        for m in missed["examples"]:
            query = m["query"] if len(m["query"]) <= 90 else m["query"][:87] + "..."
            lines.append(f"| {m['type']} | {query} | {m['relevant_papers']} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--testset", type=Path, default=Path("data/eval/retrieval_testset.jsonl"))
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument("--collection", default="arxiv_chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--no-filtered", action="store_true", help="skip the topic-filter comparison")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if not args.testset.exists():
        print(f"ERROR: {args.testset} not found. Create it with: python scripts/build_retrieval_testset.py")
        return 1
    if min(args.ks) < 1:
        print("ERROR: every K must be at least 1.")
        return 1

    from src.embeddings import Embedder
    from src.vector_store import VectorStore

    store = VectorStore(args.db_path, args.collection)
    if store.count() == 0:
        print("The index is empty. Run: python scripts/build_index.py --reset")
        return 1

    report = run_retrieval_evaluation(load_testset(args.testset), Embedder(args.model), store,
                                      ks=args.ks, filtered=not args.no_filtered)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "retrieval_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    markdown = markdown_report(report)
    (args.out_dir / "retrieval_evaluation.md").write_text(markdown + "\n")
    print(markdown)
    print(f"\nSaved to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
