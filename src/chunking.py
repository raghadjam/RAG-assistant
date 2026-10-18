"""Recursive text chunking.

The splitter tries the biggest natural boundary first (paragraph), and only
falls back to smaller ones (line, sentence, word, character) for pieces that
are still too long. This keeps sentences together whenever possible.
"""

from __future__ import annotations

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_split(
    text: str,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into chunks of at most chunk_size characters.

    chunk_overlap is the most text (in characters) that is repeated at the
    start of the next chunk. Overlap is made of whole pieces, so a chunk gets
    less overlap (or none) if the last piece before the cut is longer than
    chunk_overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    if not text or not text.strip():
        return []

    chunks = _split(text.strip(), list(separators or DEFAULT_SEPARATORS), chunk_size, chunk_overlap)
    return [c.strip() for c in chunks if c.strip()]


def _split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    # first separator that actually occurs in the text ("" means split by character)
    sep = ""
    remaining: list[str] = []
    for i, candidate in enumerate(separators):
        if candidate == "" or candidate in text:
            sep = candidate
            remaining = separators[i + 1:]
            break

    if sep == "":
        pieces = list(text)
    else:
        parts = text.split(sep)
        # keep the separator at the end of each piece so no punctuation is lost
        pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    pieces = [p for p in pieces if p]

    chunks: list[str] = []
    buffer: list[str] = []
    for piece in pieces:
        if len(piece) > chunk_size:
            # too big even on its own: flush what we have, then split it further
            chunks.extend(_merge(buffer, chunk_size, overlap))
            buffer = []
            chunks.extend(_split(piece, remaining, chunk_size, overlap))
        else:
            buffer.append(piece)
    chunks.extend(_merge(buffer, chunk_size, overlap))
    return chunks


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Pack small pieces into chunks, carrying a short tail into the next chunk."""
    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len("".join(current)) + len(piece) > chunk_size:
            chunks.append("".join(current))
            # drop pieces from the front until the leftover is a valid overlap
            while current and (
                len("".join(current)) > overlap
                or len("".join(current)) + len(piece) > chunk_size
            ):
                current.pop(0)
        current.append(piece)
    if current:
        chunks.append("".join(current))
    return chunks


def chunk_document(
    doc_id: str,
    text: str,
    metadata: dict | None = None,
    chunk_size: int = 600,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Chunk one document into records ready for the vector store."""
    metadata = metadata or {}
    pieces = recursive_split(text, chunk_size, chunk_overlap)
    return [
        {
            "chunk_id": f"{doc_id}::{i}",
            "text": piece,
            "metadata": {**metadata, "doc_id": doc_id, "chunk_index": i},
        }
        for i, piece in enumerate(pieces)
    ]
