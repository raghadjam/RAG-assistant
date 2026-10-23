"""Guardrails: decide if a query is on-topic, and check an answer against its sources.

None of these checks needs a second LLM call. They are cheap rules, so they
catch obvious problems but not every subtle one (see the README limitations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.prompts import REFUSAL_SENTENCE

CITATION_RE = re.compile(r"\[(\d+)\]")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "that", "this", "with", "from", "have", "been", "were", "their", "which", "these", "those",
    "also", "such", "than", "then", "they", "them", "there", "where", "while", "about", "into",
    "using", "used", "uses", "based", "can", "may", "more", "most", "other", "some", "both",
    "each", "does", "will", "would", "could", "should", "its", "the", "and", "for", "are", "was",
    "documents", "document", "context", "sources", "source", "paper", "papers", "study", "provided",
}


# --------------------------------------------------------------------------- #
# Before generation: is the question about this collection at all?
# --------------------------------------------------------------------------- #
@dataclass
class GateResult:
    in_domain: bool
    top_score: float | None
    reason: str


def gate_query(hits: list[dict], min_score: float) -> GateResult:
    """A question is on-topic if its best-matching chunk is similar enough."""
    if not hits:
        return GateResult(False, None, "no_results")
    top = max(h["score"] for h in hits)
    if top < min_score:
        return GateResult(False, top, f"top_score {top:.3f} < {min_score}")
    return GateResult(True, top, "ok")


# --------------------------------------------------------------------------- #
# After generation: is the answer supported by the sources it was given?
# --------------------------------------------------------------------------- #
@dataclass
class AnswerCheck:
    is_refusal: bool = False
    citations: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    support_ratio: float = 1.0
    hard_failures: list[str] = field(default_factory=list)   # answer must not be shown
    soft_failures: list[str] = field(default_factory=list)   # answer is shown with a warning

    @property
    def verdict(self) -> str:
        if self.is_refusal:
            return "refusal"
        if self.hard_failures:
            return "fail"
        if self.soft_failures:
            return "warn"
        return "pass"


def _normalize(text: str) -> str:
    # LaTeX leftovers in the corpus: "50\%" and "$10 \times$" should still match "50" and "10"
    return text.replace("\\", " ").replace("$", " ").lower()


def _content_prefixes(text: str) -> set[str]:
    """First 5 letters of each meaningful word, a cheap way to match plurals and endings."""
    return {w[:5] for w in WORD_RE.findall(text) if len(w) >= 4 and w not in STOPWORDS}


def is_refusal(answer: str) -> bool:
    text = answer.strip().lower()
    phrase = REFUSAL_SENTENCE.lower().rstrip(".")
    return text.startswith(phrase) or (phrase in text and not CITATION_RE.search(text))


def check_answer(answer: str, used_hits: list[dict], support_threshold: float = 0.6) -> AnswerCheck:
    check = AnswerCheck()
    if is_refusal(answer):
        check.is_refusal = True
        return check

    n_sources = len(used_hits)
    context = _normalize("\n".join(h["text"] for h in used_hits))

    # 1. citations: present, and pointing at sources that exist
    check.citations = [int(c) for c in CITATION_RE.findall(answer)]
    check.invalid_citations = sorted({c for c in check.citations if c < 1 or c > n_sources})
    if not check.citations:
        check.hard_failures.append("the answer has no citations")
    if check.invalid_citations:
        check.hard_failures.append(f"it cites sources that do not exist: {check.invalid_citations}")

    # 2. numbers: every figure in the answer must appear in the sources
    body = CITATION_RE.sub(" ", answer)
    context_numbers = set(NUMBER_RE.findall(context))
    for number in NUMBER_RE.findall(_normalize(body)):
        significant = len(number.replace(".", "")) >= 2 or "." in number
        if significant and number not in context_numbers and number not in check.unsupported_numbers:
            check.unsupported_numbers.append(number)
    if check.unsupported_numbers:
        check.hard_failures.append(f"it contains numbers not found in the sources: {check.unsupported_numbers}")

    # 3. leaked instructions
    if "use only the information inside" in answer.lower():
        check.hard_failures.append("it repeats the system instructions")

    # 4. wording: most sentences should share their key words with the sources
    context_prefixes = _content_prefixes(context)
    judged = supported = 0
    for sentence in SENTENCE_SPLIT_RE.split(body.strip()):
        words = {w[:5] for w in WORD_RE.findall(sentence.lower()) if len(w) >= 4 and w not in STOPWORDS}
        if len(words) < 3:
            continue
        judged += 1
        if len(words & context_prefixes) / len(words) >= 0.5:
            supported += 1
        else:
            check.unsupported_sentences.append(sentence.strip())
    check.support_ratio = supported / judged if judged else 1.0
    if check.support_ratio < support_threshold:
        check.soft_failures.append(
            f"only {check.support_ratio:.0%} of sentences overlap with the sources"
        )
    return check
