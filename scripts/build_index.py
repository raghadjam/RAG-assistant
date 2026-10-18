"""Build the vector index: chunk the clean corpus, embed it, store it in ChromaDB.

    python scripts/build_index.py
    python scripts/build_index.py --limit 200      # quick trial run
    python scripts/build_index.py --reset          # wipe the collection first

Writes reports/embedding_timing.json with chunking, embedding and ingestion times.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import chunk_document  # noqa: E402

logger = logging.getLogger("build_index")


def build_index(df: pd.DataFrame, embedder, store, chunk_size: int, chunk_overlap: int) -> dict:
    """Chunk every document, embed all chunks, write them to the store."""
    t0 = time.perf_counter()
    chunks: list[dict] = []
    for row in df.itertuples(index=False):
        meta = {
            "title": row.title,
            "source": row.source,
            "categories": row.categories,
            "published": row.published,
        }
        chunks.extend(chunk_document(row.doc_id, row.text, meta, chunk_size, chunk_overlap))
    chunk_seconds = time.perf_counter() - t0

    if not chunks:
        raise ValueError("no chunks were produced; is the corpus empty?")

    lengths = np.array([len(c["text"]) for c in chunks])
    chunking_stats = {
        "num_documents": int(len(df)),
        "num_chunks": len(chunks),
        "chunks_per_document": round(len(chunks) / len(df), 2),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_chars_mean": round(float(lengths.mean()), 1),
        "chunk_chars_min": int(lengths.min()),
        "chunk_chars_max": int(lengths.max()),
        "chunking_seconds": round(chunk_seconds, 3),
    }
    logger.info("Chunking: %s", chunking_stats)

    embedder.warm_up()
    vectors, embedding_stats = embedder.embed_timed([c["text"] for c in chunks], show_progress=True)

    t0 = time.perf_counter()
    written = store.add_chunks(chunks, vectors)
    ingest_seconds = time.perf_counter() - t0
    ingestion_stats = {
        "chunks_written": written,
        "collection_count": store.count(),
        "ingestion_seconds": round(ingest_seconds, 3),
    }
    logger.info("Ingestion: %s", ingestion_stats)

    return {"chunking": chunking_stats, "embedding": embedding_stats, "ingestion": ingestion_stats}


def markdown_summary(report: dict) -> str:
    c, e, i = report["chunking"], report["embedding"], report["ingestion"]
    return "\n".join([
        "| Metric | Value |",
        "|---|---|",
        f"| Documents | {c['num_documents']:,} |",
        f"| Chunks | {c['num_chunks']:,} ({c['chunks_per_document']} per document) |",
        f"| Chunk size / overlap | {c['chunk_size']} / {c['chunk_overlap']} characters |",
        f"| Chunk length (mean, min, max) | {c['chunk_chars_mean']}, {c['chunk_chars_min']}, {c['chunk_chars_max']} |",
        f"| Embedding model | {e['model']} ({e['dimension']} dimensions) |",
        f"| Device | {e['device']} |",
        f"| Embedding time | {e['total_seconds']} s ({e['texts_per_second']} chunks/s, {e['ms_per_text']} ms/chunk) |",
        f"| Ingestion time | {i['ingestion_seconds']} s |",
        f"| Chunks in ChromaDB | {i['collection_count']:,} |",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk, embed and index the clean corpus.")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/clean_corpus.parquet"))
    parser.add_argument("--db-path", default="chroma_db")
    parser.add_argument("--collection", default="arxiv_chunks")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None, help="only index the first N documents")
    parser.add_argument("--reset", action="store_true", help="delete the collection before indexing")
    parser.add_argument("--report", type=Path, default=Path("reports/embedding_timing.json"))
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/build_index.log")],
    )

    if not args.corpus.exists():
        print(f"ERROR: {args.corpus} not found. Run scripts/build_corpus.py first.")
        return 1

    from src.embeddings import Embedder
    from src.vector_store import VectorStore

    df = pd.read_parquet(args.corpus)
    if args.limit:
        df = df.head(args.limit)
    logger.info("Loaded %d documents from %s", len(df), args.corpus)

    embedder = Embedder(args.model, batch_size=args.batch_size)
    store = VectorStore(args.db_path, args.collection)
    if args.reset:
        store.reset()

    report = build_index(df, embedder, store, args.chunk_size, args.chunk_overlap)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print("\n" + markdown_summary(report))
    print(f"\nSaved timing report to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
