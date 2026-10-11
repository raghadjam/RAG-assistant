"""
src/preprocessing.py

Modular, unit-testable text-cleaning functions for the RAG ingestion pipeline.

Each transformation is a small, pure function so it can be tested in
isolation and composed into a pipeline (see `clean_text` / `clean_document`).

Pipeline stages:
    1. strip_html            — remove markup, keep visible text
    2. normalize_unicode      — fix mojibake, normalize to NFC, drop control chars
    3. normalize_whitespace   — collapse runs of whitespace, trim
    4. detect_language        — identify language (ISO 639-1 code)
    5. is_valid_document      — length / language / emptiness gatekeeping
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import ftfy
from bs4 import BeautifulSoup
from langdetect import DetectorFactory, LangDetectException, detect

# Make langdetect deterministic across runs (it's otherwise seeded from
# wall-clock time, which makes tests flaky).
DetectorFactory.seed = 0


# --------------------------------------------------------------------------- #
# Stage 1: HTML stripping
# --------------------------------------------------------------------------- #
def strip_html(text: str) -> str:
    """Remove HTML/XML markup and return visible text only.

    Uses BeautifulSoup so malformed markup degrades gracefully instead of
    raising. Script/style contents are dropped entirely (not just tags).
    """
    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    # separator=" " avoids gluing "</p><p>" into one run-on word
    return soup.get_text(separator=" ")


# --------------------------------------------------------------------------- #
# Stage 2: Unicode normalization
# --------------------------------------------------------------------------- #
_CONTROL_CHAR_RE = re.compile(
    "[" + "".join(chr(c) for c in range(0, 32) if c not in (9, 10, 13)) + "]"
)


def normalize_unicode(text: str) -> str:
    """Fix encoding mistakes (mojibake), normalize to NFC form, and strip
    non-printable control characters (keeping tab/newline/carriage return).
    """
    if not text:
        return ""

    # ftfy repairs common mangled-encoding artifacts, e.g. "â€™" -> "'"
    fixed = ftfy.fix_text(text)
    normalized = unicodedata.normalize("NFC", fixed)
    return _CONTROL_CHAR_RE.sub("", normalized)


# --------------------------------------------------------------------------- #
# Stage 3: Whitespace cleanup
# --------------------------------------------------------------------------- #
_MULTI_SPACE_RE = re.compile(r"[^\S\n]+")       # runs of spaces/tabs (not newlines)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")       # 3+ newlines -> 2
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")    # trailing spaces before a newline


def normalize_whitespace(text: str) -> str:
    """Collapse redundant whitespace while preserving paragraph breaks."""
    if not text:
        return ""

    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Stage 4: Language detection / filtering
# --------------------------------------------------------------------------- #
def detect_language(text: str) -> Optional[str]:
    """Return an ISO 639-1 language code (e.g. 'en'), or None if detection
    is not possible (empty/too-short/ambiguous text).
    """
    if not text or len(text.strip()) < 20:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


# --------------------------------------------------------------------------- #
# Composition: full pipeline
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Run the full cleaning pipeline (HTML -> unicode -> whitespace) on a
    raw string and return the cleaned string. Does not filter by language;
    use `is_valid_document` for that after cleaning.
    """
    text = strip_html(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


@dataclass
class ValidationResult:
    is_valid: bool
    reason: Optional[str] = None
    language: Optional[str] = None


def is_valid_document(
    text: str,
    *,
    min_chars: int = 200,
    allowed_languages: tuple[str, ...] = ("en",),
) -> ValidationResult:
    """Gatekeep a cleaned document: reject empty/too-short text and text
    not in an allowed language.
    """
    if not text or not text.strip():
        return ValidationResult(is_valid=False, reason="empty_after_cleaning")

    if len(text) < min_chars:
        return ValidationResult(is_valid=False, reason=f"too_short(<{min_chars} chars)")

    language = detect_language(text)
    if language is None:
        return ValidationResult(is_valid=False, reason="language_undetectable")

    if language not in allowed_languages:
        return ValidationResult(is_valid=False, reason=f"language_filtered({language})", language=language)

    return ValidationResult(is_valid=True, language=language)


def clean_document(raw_text: str, *, min_chars: int = 200,
                    allowed_languages: tuple[str, ...] = ("en",)) -> tuple[str, ValidationResult]:
    """Convenience wrapper: clean `raw_text` and validate the result in one
    call. Returns (cleaned_text, validation_result).
    """
    cleaned = clean_text(raw_text)
    validation = is_valid_document(cleaned, min_chars=min_chars, allowed_languages=allowed_languages)
    return cleaned, validation
