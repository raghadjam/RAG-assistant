"""Embedding generation with sentence-transformers, with timing stats."""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Turns text into normalized vectors.

    Vectors are L2-normalized, so cosine similarity and dot product agree.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 64):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )

    def embed_timed(self, texts: list[str], show_progress: bool = False) -> tuple[np.ndarray, dict]:
        """Embed and return (vectors, stats). Call warm_up() first for fair numbers."""
        start = time.perf_counter()
        vectors = self.embed(texts, show_progress=show_progress)
        seconds = time.perf_counter() - start
        stats = {
            "model": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "dimension": int(self.dim),
            "num_texts": len(texts),
            "total_seconds": round(seconds, 3),
            "texts_per_second": round(len(texts) / seconds, 1) if seconds > 0 else None,
            "ms_per_text": round(1000 * seconds / len(texts), 3) if texts else None,
        }
        logger.info("Embedded %d texts in %.2fs (%s texts/s)", len(texts), seconds, stats["texts_per_second"])
        return vectors, stats

    def warm_up(self) -> None:
        """The first call is slow (CUDA init, kernel setup), so run one throwaway call."""
        self.embed(["warm up"])
