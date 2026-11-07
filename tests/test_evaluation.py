"""Tests for src/evaluation.py and src/text_utils.py."""

import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document  # noqa: E402
from src.evaluation import (  # noqa: E402
    abstract_of,
    build_doc_questions,
    build_retrieval_testset,
    citation_coverage,
    cosine_similarity,
    evaluate_ranking,
    first_sentences,
    latency_stats,
    load_testset,
    phrase_regex,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge1_f1,
    run_retrieval_evaluation,
    save_testset,
    split_sentences,
    summarize,
    tokens,
)
from src.text_utils import strip_latex  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402
from tests.fakes import HashEmbedder  # noqa: E402


class TestRankingMetrics:
    RANKED = ["a", "b", "a", "c", "d"]
    RELEVANT = {"a", "d"}

    def test_precision_counts_relevant_chunks_over_k(self):
        assert precision_at_k(self.RANKED, self.RELEVANT, 1) == 1.0
        assert precision_at_k(self.RANKED, self.RELEVANT, 3) == pytest.approx(2 / 3)     # a, b, a
        assert precision_at_k(self.RANKED, self.RELEVANT, 5) == pytest.approx(3 / 5)     # a, a, d

    def test_recall_counts_distinct_relevant_papers_over_min_of_relevant_and_k(self):
        assert recall_at_k(self.RANKED, self.RELEVANT, 1) == 1.0         # 1 found / min(2, 1)
        assert recall_at_k(self.RANKED, self.RELEVANT, 3) == 0.5         # {a} / min(2, 3)
        assert recall_at_k(self.RANKED, self.RELEVANT, 5) == 1.0         # {a, d} / 2

    def test_repeated_chunks_from_one_paper_help_precision_but_not_recall(self):
        ranked = ["a"] * 5
        assert precision_at_k(ranked, {"a", "b"}, 5) == 1.0
        assert recall_at_k(ranked, {"a", "b"}, 5) == 0.5

    def test_single_relevant_paper_recall_is_a_hit_rate(self):
        ranked = ["x", "y", "z", "a", "w"]
        assert [recall_at_k(ranked, {"a"}, k) for k in (1, 3, 4, 5)] == [0, 0, 1, 1]
        assert precision_at_k(ranked, {"a"}, 5) == pytest.approx(1 / 5)

    def test_many_relevant_papers_recall_is_capped_by_k(self):
        relevant = {f"d{i}" for i in range(100)}
        assert recall_at_k([f"d{i}" for i in range(5)], relevant, 5) == 1.0

    def test_fewer_results_than_k_lowers_precision(self):
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)
        assert recall_at_k(["a"], {"a"}, 5) == 1.0

    def test_nothing_relevant(self):
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0
        assert recall_at_k(["x", "y"], {"a"}, 2) == 0.0
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_ranking(self):
        assert precision_at_k([], {"a"}, 3) == 0.0 and recall_at_k([], {"a"}, 3) == 0.0

    def test_reciprocal_rank(self):
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
        assert reciprocal_rank(["a"], {"a"}) == 1.0
        assert reciprocal_rank(["x", "y", "z", "a"], {"a", "q"}) == 0.25

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_must_be_positive(self, k):
        with pytest.raises(ValueError):
            precision_at_k(["a"], {"a"}, k)
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, k)

    def test_recall_needs_a_relevant_paper(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], set(), 3)

    def test_evaluate_ranking_bundles_everything(self):
        result = evaluate_ranking(self.RANKED, self.RELEVANT, ks=(1, 5))
        assert result["precision"] == {1: 1.0, 5: pytest.approx(0.6)}
        assert result["recall"] == {1: 1.0, 5: 1.0}
        assert result["rr"] == 1.0

    def test_summarize_averages_queries(self):
        a = {"metrics": evaluate_ranking(["a"], {"a"}, (1,))}
        b = {"metrics": evaluate_ranking(["x"], {"a"}, (1,))}
        summary = summarize([a, b], (1,))
        assert summary == {"n": 2, "precision": {1: 0.5}, "recall": {1: 0.5}, "mrr": 0.5}

    def test_latency_stats(self):
        assert latency_stats([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        stats = latency_stats(list(range(1, 101)))
        assert stats["mean"] == 50.5 and stats["p50"] == 50.5 and stats["p95"] == pytest.approx(95.05, abs=0.06)


class TestTextHelpers:
    def test_strip_latex(self):
        raw = r"METR's 50\% time horizon \emph{converts} human time, a $10 \times$ jump."
        clean = strip_latex(raw)
        assert "converts" in clean and "50%" in clean and "\\" not in clean and "$" not in clean
        assert strip_latex(r"\textbf{\emph{bold}} text") == "bold text"
        assert strip_latex("") == "" and strip_latex(None) == ""

    @pytest.mark.parametrize("text,phrase", [
        ("We study Federated Learning here", "federated learning"),
        ("federated-learning systems", "federated learning"),
        ("many graph neural networks exist", "graph neural network"),
        ("a multi agent setup", "multi-agent"),
        ("retrieval-augmented generation", "retrieval augmented generation"),
    ])
    def test_phrase_regex_matches_variants(self, text, phrase):
        assert phrase_regex(phrase).search(text)

    @pytest.mark.parametrize("text,phrase", [
        ("reward modeling", "reward model"),
        ("unfederated learning", "federated learning"),
        ("federated and learning", "federated learning"),
        ("alignmentx", "alignment"),
    ])
    def test_phrase_regex_does_not_overmatch(self, text, phrase):
        assert not phrase_regex(phrase).search(text)

    def test_split_sentences(self):
        assert split_sentences("One idea. Two ideas!  Three? Done") == ["One idea.", "Two ideas!", "Three?", "Done"]
        assert split_sentences("") == []

    def test_abstract_of_removes_the_title(self):
        assert abstract_of("My Title", "My Title\n\nThe abstract begins.") == "The abstract begins."
        assert abstract_of("Other", "No title here.") == "No title here."

    def test_first_sentences_cleans_latex(self):
        assert first_sentences(r"A \emph{great} idea. Another one. A third.") == "A great idea. Another one."


class TestAnswerMetrics:
    def test_tokens_drop_stopwords_and_punctuation(self):
        assert tokens("The Cats, and the DOGS!") == ["cats", "dogs"]

    def test_rouge1_f1(self):
        assert rouge1_f1("cats chase mice", "cats chase mice") == 1.0
        assert rouge1_f1("alpha beta", "gamma delta") == 0.0
        assert rouge1_f1("cats chase mice", "cats chase dogs") == pytest.approx(2 / 3)   # 2 shared of 3 and 3
        assert rouge1_f1("", "cats") == 0.0

    def test_rouge_ignores_citations(self):
        assert rouge1_f1("cats chase mice [1][2]", "cats chase mice") == 1.0

    def test_cosine_similarity(self):
        assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)
        assert cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_citation_coverage(self):
        assert citation_coverage("First claim [1]. Second claim [2]. Uncited claim.") == pytest.approx(2 / 3)
        assert citation_coverage("Only one sentence [1].") == 1.0
        assert citation_coverage("No citations at all.") == 0.0
        assert citation_coverage("") == 0.0


def synthetic_corpus(n=120, topics=4, seed=0):
    rng = np.random.default_rng(seed)
    vocab = {t: [f"topicword{t}x{j}" for j in range(8)] for t in range(topics)}
    rows = []
    for i in range(n):
        topic = i % topics
        title = f"Study {i} of {vocab[topic][0]} and {vocab[topic][1]} methods"
        sentences = [f"We study zq{i}x using {vocab[topic][2]} and {vocab[topic][3]} approaches in detail here."]
        for s in range(3):
            words = " ".join(rng.choice(vocab[topic], size=8))
            sentences.append(f"In part {s} we analyse {words} for zq{i}x across many realistic settings.")
        if i % 3 == 0:
            sentences.append("This also covers Federated Learning in the wild, with enough words to be a sentence.")
        if i % 7 == 0:
            sentences.append("We use reinforcement learning as well, which gives a long enough closing sentence.")
        rows.append({"doc_id": f"doc{i}", "title": title, "text": f"{title}\n\n" + " ".join(sentences),
                     "topic_id": topic})
    return pd.DataFrame(rows)


CONCEPTS = ["federated learning", "reinforcement learning", "never mentioned phrase"]


class TestBuildTestset:
    def build(self, **kwargs):
        options = {"n_title": 12, "n_sentence": 12, "min_relevant": 10, "concepts": CONCEPTS, **kwargs}
        return build_retrieval_testset(synthetic_corpus(), **options)

    def test_counts_and_types(self):
        items = self.build()
        types = [i["type"] for i in items]
        assert types.count("title") == 12 and types.count("sentence") == 12
        assert types.count("concept") == 2            # "never mentioned phrase" and too-rare ones are dropped

    def test_known_item_queries_have_one_relevant_paper_and_a_topic(self):
        for item in self.build():
            if item["type"] in ("title", "sentence"):
                assert len(item["relevant_doc_ids"]) == 1 and item["topic_id"] in range(4)

    def test_title_query_is_the_papers_title(self):
        corpus = synthetic_corpus()
        titles = dict(zip(corpus["doc_id"], corpus["title"]))
        for item in self.build():
            if item["type"] == "title":
                assert item["query"] == titles[item["relevant_doc_ids"][0]]

    def test_sentence_query_comes_from_the_abstract_but_is_not_the_opening_sentence(self):
        corpus = synthetic_corpus()
        texts = dict(zip(corpus["doc_id"], corpus["text"]))
        for item in self.build():
            if item["type"] == "sentence":
                doc = item["relevant_doc_ids"][0]
                assert item["query"] in " ".join(texts[doc].split())
                assert not item["query"].startswith("We study zq")
                assert 60 <= len(item["query"]) <= 300

    def test_title_and_sentence_queries_use_different_papers(self):
        items = self.build()
        title_docs = {i["relevant_doc_ids"][0] for i in items if i["type"] == "title"}
        sentence_docs = {i["relevant_doc_ids"][0] for i in items if i["type"] == "sentence"}
        assert not title_docs & sentence_docs
        assert len(title_docs) == 12 and len(sentence_docs) == 12

    def test_topics_are_covered_evenly(self):
        topics = [i["topic_id"] for i in self.build() if i["type"] == "title"]
        counts = np.bincount(topics)
        assert counts.max() - counts.min() <= 1 and len(counts) == 4

    def test_concept_relevance_is_exactly_the_papers_mentioning_the_phrase(self):
        corpus = synthetic_corpus()
        concept = next(i for i in self.build() if i["query"] == "federated learning")
        expected = [d for d, t in zip(corpus["doc_id"], corpus["text"]) if "federated learning" in t.lower()]
        assert concept["relevant_doc_ids"] == sorted(expected) and concept["topic_id"] is None

    def test_min_relevant_drops_rare_concepts(self):
        queries = [i["query"] for i in self.build(min_relevant=30) if i["type"] == "concept"]
        assert queries == ["federated learning"]                       # 40 papers; reinforcement has 18

    def test_same_seed_same_testset_different_seed_different(self):
        assert self.build(seed=1) == self.build(seed=1)
        assert self.build(seed=1) != self.build(seed=2)

    def test_latex_is_removed_from_queries(self):
        corpus = synthetic_corpus()
        corpus.loc[0, "title"] = r"Fast \emph{training} of $O(n)$ models with extra words"
        corpus.loc[0, "text"] = corpus.loc[0, "title"] + "\n\n" + "A long enough abstract sentence goes here. " * 12
        for item in build_retrieval_testset(corpus, n_title=200, n_sentence=0, concepts=[]):
            assert "\\" not in item["query"] and "$" not in item["query"]

    def test_short_abstracts_and_titles_are_skipped(self):
        corpus = synthetic_corpus(n=8)
        corpus.loc[0, "text"] = corpus.loc[0, "title"] + "\n\nToo short."
        corpus.loc[1, "title"] = "Tiny"
        items = build_retrieval_testset(corpus, n_title=50, n_sentence=0, concepts=[])
        docs = {i["relevant_doc_ids"][0] for i in items}
        assert "doc0" not in docs and "doc1" not in docs and len(docs) == 6

    def test_works_without_a_topic_column(self):
        corpus = synthetic_corpus().drop(columns="topic_id")
        items = build_retrieval_testset(corpus, n_title=5, n_sentence=5, concepts=[])
        assert len(items) == 10 and all(i["topic_id"] is None for i in items)

    def test_asking_for_more_than_exist_returns_what_exists(self):
        items = build_retrieval_testset(synthetic_corpus(n=10), n_title=100, n_sentence=100, concepts=[])
        assert len(items) <= 10

    @pytest.mark.parametrize("missing", ["doc_id", "title", "text"])
    def test_required_columns(self, missing):
        with pytest.raises(ValueError, match=missing):
            build_retrieval_testset(synthetic_corpus().drop(columns=missing))

    def test_save_and_load_roundtrip(self, tmp_path):
        items = self.build()
        save_testset(items, tmp_path / "sub" / "set.jsonl")
        assert load_testset(tmp_path / "sub" / "set.jsonl") == items

    def test_load_rejects_incomplete_items(self, tmp_path):
        (tmp_path / "bad.jsonl").write_text('{"id": "x", "query": "q", "relevant_doc_ids": []}\n')
        with pytest.raises(ValueError, match="relevant_doc_ids"):
            load_testset(tmp_path / "bad.jsonl")


@pytest.fixture(scope="module")
def retrieval_setup():
    corpus = synthetic_corpus()
    embedder = HashEmbedder()
    store = VectorStore(path=None, collection_name=f"ev_{uuid.uuid4().hex}")
    chunks = []
    for row in corpus.itertuples(index=False):
        chunks.extend(chunk_document(row.doc_id, row.text, {"title": row.title, "topic_id": int(row.topic_id)},
                                     500, 80))
    store.add_chunks(chunks, embedder.embed([c["text"] for c in chunks]))
    testset = build_retrieval_testset(corpus, n_title=16, n_sentence=16, concepts=CONCEPTS, min_relevant=10)
    return embedder, store, testset


class TestRunRetrievalEvaluation:
    def test_report_structure(self, retrieval_setup):
        embedder, store, testset = retrieval_setup
        report = run_retrieval_evaluation(testset, embedder, store, ks=(1, 3, 5))
        assert report["ks"] == [1, 3, 5] and report["n_queries"] == len(testset) == 34
        assert report["index_chunks"] == store.count()
        assert set(report["by_type"]) == {"title", "sentence", "concept"}
        assert len(report["per_query"]) == 34
        for entry in report["by_type"].values():
            assert set(entry["precision"]) == {1, 3, 5} and 0 <= entry["mrr"] <= 1
            assert {"mean", "p50", "p95"} == set(entry["latency_ms"])

    def test_known_item_queries_are_found(self, retrieval_setup):
        embedder, store, testset = retrieval_setup
        by_type = run_retrieval_evaluation(testset, embedder, store, ks=(1, 5))["by_type"]
        assert by_type["title"]["recall"][5] >= 0.9 and by_type["title"]["mrr"] >= 0.8
        assert by_type["sentence"]["recall"][5] >= 0.9

    def test_concept_queries_have_many_relevant_papers_and_good_precision(self, retrieval_setup):
        embedder, store, testset = retrieval_setup
        concept = run_retrieval_evaluation(testset, embedder, store, ks=(5,))["by_type"]["concept"]
        assert concept["relevant_papers_mean"] >= 10
        assert concept["precision"][5] >= 0.8

    def test_filtered_runs_exist_only_for_known_item_queries(self, retrieval_setup):
        embedder, store, testset = retrieval_setup
        by_type = run_retrieval_evaluation(testset, embedder, store, ks=(1, 5))["by_type"]
        assert "with_topic_filter" in by_type["title"] and "with_topic_filter" in by_type["sentence"]
        assert "with_topic_filter" not in by_type["concept"]

    def test_the_right_topic_filter_does_not_lose_the_paper(self, retrieval_setup):
        embedder, store, testset = retrieval_setup
        entry = run_retrieval_evaluation(testset, embedder, store, ks=(1, 5))["by_type"]["title"]
        assert entry["with_topic_filter"]["recall"][5] >= entry["same_queries_unfiltered"]["recall"][5]
        assert entry["with_topic_filter"]["n"] == entry["same_queries_unfiltered"]["n"] == 16

    def test_filtering_can_be_switched_off(self, retrieval_setup):
        embedder, store, testset = retrieval_setup
        by_type = run_retrieval_evaluation(testset, embedder, store, ks=(5,), filtered=False)["by_type"]
        assert all("with_topic_filter" not in e for e in by_type.values())

    def test_queries_that_are_never_found_are_listed(self, retrieval_setup):
        embedder, store, _ = retrieval_setup
        impossible = [{"id": "x-1", "type": "title", "query": "completely unrelated gibberish words",
                       "relevant_doc_ids": ["no-such-doc"], "topic_id": None}]
        report = run_retrieval_evaluation(impossible, embedder, store, ks=(1, 5))
        assert report["not_found_in_top_k"]["count"] == 1
        assert report["not_found_in_top_k"]["examples"][0]["id"] == "x-1"
        assert report["overall"]["precision"][5] == 0.0 and report["overall"]["mrr"] == 0.0


class TestDocQuestions:
    def test_questions_have_references_and_ids(self):
        questions = build_doc_questions(synthetic_corpus(), n=8)
        assert len(questions) == 8 and len({q["doc_id"] for q in questions}) == 8
        for q in questions:
            assert q["question"].startswith('What does the paper titled "') and q["category"] == "doc"
            assert len(split_sentences(q["reference"])) == 2

    def test_reproducible_and_stratified(self):
        corpus = synthetic_corpus()
        assert build_doc_questions(corpus, 8, seed=3) == build_doc_questions(corpus, 8, seed=3)
        topics = [int(corpus.loc[corpus["doc_id"] == q["doc_id"], "topic_id"].iloc[0])
                  for q in build_doc_questions(corpus, 8)]
        assert sorted(np.bincount(topics)) == [2, 2, 2, 2]

    def test_papers_with_short_abstracts_or_titles_are_not_used(self):
        corpus = synthetic_corpus(n=6)
        corpus.loc[0, "text"] = corpus.loc[0, "title"] + "\n\nShort."
        corpus.loc[1, "title"] = "Tiny"
        assert {q["doc_id"] for q in build_doc_questions(corpus, 10)} == {"doc2", "doc3", "doc4", "doc5"}
