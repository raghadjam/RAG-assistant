"""Ask the RAG system a question.

    python scripts/ask.py "How can hallucination in language models be reduced?"
    python scripts/ask.py                      # interactive, type questions, empty line to quit
    python scripts/ask.py "..." --json         # full result as JSON

Needs the index from Week 2 (scripts/build_index.py) and OPENROUTER_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_settings  # noqa: E402


def show(result) -> None:
    t = result.timing
    print(f"\nStatus : {result.status}" + (f"   (model: {result.llm['model']})" if result.llm else ""))
    print(f"\n{result.answer}\n")
    if result.sources:
        print("Sources:")
        for s in result.sources:
            print(f"  [{s['n']}] {s['title']}  ({s['doc_id']})  score {s['score']}")
    if result.check and result.check["soft_failures"]:
        print("\nWarning: " + "; ".join(result.check["soft_failures"]))
    if result.error:
        print(f"\nNote: {result.error}")
    print(f"\nLatency: retrieval {t['retrieval_ms']:.0f} ms | generation {t['generation_ms']:.0f} ms "
          f"| total {t['total_ms']:.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask the RAG system a question.")
    parser.add_argument("question", nargs="?", help="omit for interactive mode")
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument("--collection", default="arxiv_chunks")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--k", type=int, help="chunks to retrieve")
    parser.add_argument("--min-score", type=float, help="off-topic cutoff")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    if args.k:
        settings = replace(settings, top_k=args.k)
    if args.min_score is not None:
        settings = replace(settings, min_score=args.min_score)

    from src.embeddings import Embedder
    from src.llm_client import LLMClient
    from src.rag import RAGPipeline
    from src.vector_store import VectorStore

    store = VectorStore(args.db_path, args.collection)
    if store.count() == 0:
        print("The index is empty. Run: python scripts/build_index.py --reset")
        return 1

    embedder = Embedder(args.model)
    embedder.warm_up()
    pipeline = RAGPipeline(embedder, store, LLMClient(settings), settings)

    def run(question: str) -> None:
        result = pipeline.ask(question)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) if args.json else "", end="")
        if not args.json:
            show(result)

    if args.question:
        run(args.question)
        return 0

    print(f"Index: {store.count():,} chunks. Type a question (empty line to quit).")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        run(question)
    return 0


if __name__ == "__main__":
    sys.exit(main())
