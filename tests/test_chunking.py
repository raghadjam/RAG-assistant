"""Unit tests for src/chunking.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document, recursive_split  # noqa: E402

SENTENCES = [f"This is sentence number {i} about neural networks." for i in range(40)]
LONG_TEXT = " ".join(SENTENCES)


class TestRecursiveSplit:
    def test_short_text_is_one_chunk(self):
        assert recursive_split("A short abstract.", chunk_size=600) == ["A short abstract."]

    def test_empty_and_whitespace_return_nothing(self):
        assert recursive_split("") == []
        assert recursive_split("   \n\n  ") == []

    def test_no_chunk_exceeds_chunk_size(self):
        chunks = recursive_split(LONG_TEXT, chunk_size=200, chunk_overlap=40)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_no_empty_chunks(self):
        chunks = recursive_split(LONG_TEXT, chunk_size=150, chunk_overlap=30)
        assert all(c.strip() for c in chunks)

    def test_all_content_is_kept(self):
        chunks = recursive_split(LONG_TEXT, chunk_size=200, chunk_overlap=40)
        original_words = set(LONG_TEXT.split())
        chunk_words = set(" ".join(chunks).split())
        assert original_words <= chunk_words

    def test_overlap_repeats_text_between_neighbours(self):
        chunks = recursive_split(LONG_TEXT, chunk_size=200, chunk_overlap=80)
        shared = 0
        for a, b in zip(chunks, chunks[1:]):
            if a.split()[-1] in b.split():
                shared += 1
        assert shared > 0

    def test_zero_overlap_does_not_repeat_sentences(self):
        chunks = recursive_split(LONG_TEXT, chunk_size=200, chunk_overlap=0)
        joined = " ".join(chunks)
        for sentence in SENTENCES:
            assert joined.count(sentence) == 1

    def test_prefers_sentence_boundaries(self):
        chunks = recursive_split(LONG_TEXT, chunk_size=200, chunk_overlap=0)
        # every chunk except possibly the last should end on a full sentence
        assert all(c.endswith(".") for c in chunks)

    def test_paragraphs_are_split_first(self):
        text = ("First paragraph sentence. " * 8).strip() + "\n\n" + ("Second paragraph sentence. " * 8).strip()
        chunks = recursive_split(text, chunk_size=230, chunk_overlap=0)
        assert chunks[0].startswith("First")
        assert chunks[-1].startswith("Second")

    def test_text_with_no_separators_splits_by_character(self):
        chunks = recursive_split("x" * 1000, chunk_size=300, chunk_overlap=0)
        assert all(len(c) <= 300 for c in chunks)
        assert sum(len(c) for c in chunks) == 1000

    def test_is_deterministic(self):
        assert recursive_split(LONG_TEXT, 200, 40) == recursive_split(LONG_TEXT, 200, 40)

    def test_unicode_text_is_safe(self):
        text = "Résumé des résultats sur les réseaux de neurones. " * 30
        chunks = recursive_split(text, chunk_size=200, chunk_overlap=40)
        assert all("é" in c for c in chunks)

    @pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (100, -1), (100, 100), (100, 150)])
    def test_invalid_parameters_raise(self, size, overlap):
        with pytest.raises(ValueError):
            recursive_split("some text", chunk_size=size, chunk_overlap=overlap)


class TestChunkDocument:
    def test_ids_are_unique_and_ordered(self):
        chunks = chunk_document("doc1", LONG_TEXT, chunk_size=200, chunk_overlap=40)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
        assert ids[0] == "doc1::0"
        assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))

    def test_metadata_is_carried_over(self):
        chunks = chunk_document("doc1", LONG_TEXT, {"title": "T", "source": "arxiv"}, 200, 40)
        assert all(c["metadata"]["title"] == "T" for c in chunks)
        assert all(c["metadata"]["doc_id"] == "doc1" for c in chunks)

    def test_empty_document_gives_no_chunks(self):
        assert chunk_document("doc1", "") == []
