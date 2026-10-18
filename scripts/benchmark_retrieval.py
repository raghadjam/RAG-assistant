"""Benchmark semantic search: run varied queries several times and log latency.

    python scripts/benchmark_retrieval.py
    python scripts/benchmark_retrieval.py --repeats 10 --k 5

Writes reports/retrieval_benchmark.json and reports/retrieval_benchmark.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.search import semantic_search  # noqa: E402

logger = logging.getLogger("benchmark")

# (label, query). Mixed on purpose: short, long, on-topic, and off-topic.
QUERIES = [
    ("topic", "transformer attention mechanism"),
    ("topic", "retrieval augmented generation for question answering"),
    ("topic", "reinforcement learning from human feedback"),
    ("topic", "graph neural networks for node classification"),
    ("topic", "diffusion models for image generation"),
    ("topic", "federated learning with differential privacy"),
    ("question", "How can large language models be made to hallucinate less?"),
    ("question", "What methods exist for evaluating the reasoning ability of language models?"),
    ("long", "We study how fine-tuning a pretrained language model on a small labeled dataset "
             "affects its robustness to distribution shift and what regularization helps most"),
    ("short", "LLM"),
    ("off-topic", "best pizza recipe with fresh basil"),
    ("nonsense", "qzxv blorf wibble"),
]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q))


def run_benchmark(embedder, store, queries=QUERIES, repeats: int = 5, k: int = 5) -> dict:
    # warm-up so one-time costs (CUDA init, index load) do not pollute the numbers
    semantic_search("warm up query", embedder, store, k=k)

    per_query = []
    all_totals = []
    for label, query in queries:
        embed, search, total = [], [], []
        top = None
        for _ in range(repeats):
            out = semantic_search(query, embedder, store, k=k)
            embed.append(out["timing"]["embed_ms"])
            search.append(out["timing"]["search_ms"])
            total.append(out["timing"]["total_ms"])
            top = out["results"][0] if out["results"] else None
        all_totals.extend(total)
        row = {
            "type": label,
            "query": query,
            "top1_score": round(top["score"], 3) if top else None,
            "top1_title": top["metadata"].get("title") if top else None,
            "embed_ms_mean": round(float(np.mean(embed)), 2),
            "search_ms_mean": round(float(np.mean(search)), 2),
            "total_ms_mean": round(float(np.mean(total)), 2),
            "total_ms_p95": round(percentile(total, 95), 2),
        }
        per_query.append(row)
        logger.info("%-9s %-60.60s top1=%s total=%.1fms", label, query, row["top1_score"], row["total_ms_mean"])

    return {
        "k": k,
        "repeats": repeats,
        "collection_count": store.count(),
        "overall": {
            "queries": len(queries),
            "searches": len(all_totals),
            "total_ms_mean": round(float(np.mean(all_totals)), 2),
            "total_ms_p50": round(percentile(all_totals, 50), 2),
            "total_ms_p95": round(percentile(all_totals, 95), 2),
            "total_ms_max": round(float(np.max(all_totals)), 2),
        },
        "per_query": per_query,
    }


def markdown_report(report: dict) -> str:
    o = report["overall"]
    lines = [
        f"Index size: {report['collection_count']:,} chunks, top-{report['k']} search, "
        f"{report['repeats']} repeats per query ({o['searches']} searches).",
        "",
        f"Overall latency: mean {o['total_ms_mean']} ms, p50 {o['total_ms_p50']} ms, "
        f"p95 {o['total_ms_p95']} ms, max {o['total_ms_max']} ms.",
        "",
        "| Type | Query | Top-1 score | Embed (ms) | Search (ms) | Total (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for r in report["per_query"]:
        query = r["query"] if len(r["query"]) <= 55 else r["query"][:52] + "..."
        lines.append(
            f"| {r['type']} | {query} | {r['top1_score']} | {r['embed_ms_mean']} | "
            f"{r['search_ms_mean']} | {r['total_ms_mean']} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency.")
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument("--collection", default="arxiv_chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/benchmark.log")],
    )

    from src.embeddings import Embedder
    from src.vector_store import VectorStore

    store = VectorStore(args.db_path, args.collection)
    if store.count() == 0:
        print("ERROR: the collection is empty. Run scripts/build_index.py first.")
        return 1

    embedder = Embedder(args.model)
    report = run_benchmark(embedder, store, repeats=args.repeats, k=args.k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "retrieval_benchmark.json").write_text(json.dumps(report, indent=2))
    markdown = markdown_report(report)
    (args.out_dir / "retrieval_benchmark.md").write_text(markdown + "\n")
    print("\n" + markdown)
    print(f"\nSaved reports to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
