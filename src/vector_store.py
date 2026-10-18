"""ChromaDB wrapper: ingest chunks with embeddings and query them.

Embeddings are computed outside Chroma and passed in, so the same model is
used for indexing and querying and Chroma never downloads its own model.
"""

from __future__ import annotations

import logging

import chromadb
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, path: str | None = "chroma_db", collection_name: str = "arxiv_chunks"):
        # path=None gives an in-memory client (used by the tests)
        self.client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection (used for a clean rebuild)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[dict], embeddings: np.ndarray, batch_size: int = 1000) -> int:
        """Insert chunks (dicts from chunk_document) with their embeddings.

        Uses upsert, so running it twice does not create duplicates. Returns
        the number of chunks written.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if len(chunks) == 0:
            return 0
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings contain NaN or infinite values")

        # Chroma rejects duplicate ids inside one batch, so keep the last of each
        unique: dict[str, int] = {}
        for i, chunk in enumerate(chunks):
            if chunk["text"].strip():  # empty text would embed to meaningless noise
                unique[chunk["chunk_id"]] = i
        indices = list(unique.values())
        skipped = len(chunks) - len(indices)
        if skipped:
            logger.warning("Skipped %d empty or duplicate chunks", skipped)

        # Chroma has a maximum batch size that depends on the SQLite build
        limit = getattr(self.client, "get_max_batch_size", lambda: batch_size)()
        step = max(1, min(batch_size, limit))

        written = 0
        for start in range(0, len(indices), step):
            batch = indices[start:start + step]
            metadatas = [_clean_metadata(chunks[i].get("metadata")) for i in batch]
            self.collection.upsert(
                ids=[chunks[i]["chunk_id"] for i in batch],
                documents=[chunks[i]["text"] for i in batch],
                embeddings=embeddings[batch].tolist(),
                metadatas=metadatas if any(m is not None for m in metadatas) else None,
            )
            written += len(batch)
        return written

    def query(self, query_embedding: np.ndarray, k: int = 5, where: dict | None = None) -> list[dict]:
        """Return up to k nearest chunks, best first. Empty list if nothing matches."""
        if k <= 0:
            raise ValueError("k must be positive")
        total = self.count()
        if total == 0:
            return []
        k = min(k, total)  # asking for more results than exist can raise in Chroma

        try:
            raw = self.collection.query(
                query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            if where:
                raise ValueError(f"invalid metadata filter {where!r}: {exc}") from exc
            raise

        hits = []
        for chunk_id, text, meta, dist in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            hits.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": meta,
                "distance": float(dist),
                "score": 1.0 - float(dist),  # cosine similarity
            })
        return hits


def _clean_metadata(metadata: dict | None) -> dict | None:
    """Chroma only accepts str/int/float/bool values, and rejects None values
    and empty dicts. Returns None when nothing is left to store."""
    cleaned = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned or None
