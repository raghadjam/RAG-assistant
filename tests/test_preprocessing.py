"""
tests/test_preprocessing.py

Unit tests for src/preprocessing.py.

Run with:
    pytest tests/test_preprocessing.py -v
    pytest --cov=src --cov-report=term-missing   # with coverage
"""

import sys
from pathlib import Path

import pytest

# Allow `import src.preprocessing` when running pytest from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    ValidationResult,
    clean_document,
    clean_text,
    detect_language,
    is_valid_document,
    normalize_unicode,
    normalize_whitespace,
    strip_html,
)


# --------------------------------------------------------------------------- #
# strip_html
# --------------------------------------------------------------------------- #
class TestStripHtml:
    def test_removes_simple_tags(self):
        # strip_html alone may leave extra whitespace at tag boundaries
        # (that's normalize_whitespace's job later in the pipeline) —
        # here we only assert the tags themselves are gone.
        result = strip_html("<p>Hello <b>world</b></p>")
        assert "<" not in result and ">" not in result
        assert result.split() == ["Hello", "world"]

    def test_removes_script_and_style_content(self):
        html = "<div>Visible<script>alert('x')</script><style>.a{color:red}</style>Text</div>"
        result = strip_html(html)
        assert "alert" not in result
        assert "color:red" not in result
        assert "Visible" in result and "Text" in result

    def test_handles_malformed_markup_gracefully(self):
        # Unclosed tags shouldn't raise
        result = strip_html("<div><p>Broken<span>markup</div>")
        assert "Broken" in result and "markup" in result

    def test_empty_string_returns_empty(self):
        assert strip_html("") == ""

    def test_plain_text_passthrough(self):
        assert strip_html("just plain text") == "just plain text"

    def test_preserves_separation_between_block_elements(self):
        # Without a separator, "</p><p>" style breaks would glue words together
        result = strip_html("<p>First</p><p>Second</p>")
        assert "First" in result and "Second" in result
        assert "FirstSecond" not in result


# --------------------------------------------------------------------------- #
# normalize_unicode
# --------------------------------------------------------------------------- #
class TestNormalizeUnicode:
    def test_fixes_mojibake(self):
        # "â€™" is a classic mis-decoded right single quote
        mangled = "Itâ€™s a test"
        result = normalize_unicode(mangled)
        assert "â€™" not in result
        assert "It" in result

    def test_strips_control_characters(self):
        text = "Hello\x00\x01World"
        result = normalize_unicode(text)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_preserves_tabs_and_newlines(self):
        text = "Line1\nLine2\tTabbed"
        result = normalize_unicode(text)
        assert "\n" in result
        assert "\t" in result

    def test_nfc_normalization_composes_characters(self):
        # 'e' + combining acute accent (NFD) should normalize to 'é' (NFC)
        decomposed = "e\u0301cole"
        result = normalize_unicode(decomposed)
        assert result == "\u00e9cole"

    def test_empty_string_returns_empty(self):
        assert normalize_unicode("") == ""


# --------------------------------------------------------------------------- #
# normalize_whitespace
# --------------------------------------------------------------------------- #
class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("Hello    world") == "Hello world"

    def test_collapses_tabs(self):
        assert normalize_whitespace("Hello\t\tworld") == "Hello world"

    def test_collapses_excess_newlines_but_keeps_paragraphs(self):
        text = "Para one.\n\n\n\n\nPara two."
        result = normalize_whitespace(text)
        assert result == "Para one.\n\nPara two."

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_whitespace("   Hello world   ") == "Hello world"

    def test_removes_trailing_spaces_before_newline(self):
        text = "Line one.   \nLine two."
        result = normalize_whitespace(text)
        assert "   \n" not in result

    def test_empty_string_returns_empty(self):
        assert normalize_whitespace("") == ""

    def test_preserves_single_newlines(self):
        text = "Line one.\nLine two."
        assert normalize_whitespace(text) == "Line one.\nLine two."


# --------------------------------------------------------------------------- #
# detect_language
# --------------------------------------------------------------------------- #
class TestDetectLanguage:
    def test_detects_english(self):
        text = "This is a reasonably long sentence written in English for detection."
        assert detect_language(text) == "en"

    def test_detects_non_english(self):
        text = "Esto es una oracion bastante larga escrita en espanol para deteccion de idioma."
        assert detect_language(text) == "es"

    def test_returns_none_for_empty_text(self):
        assert detect_language("") is None

    def test_returns_none_for_very_short_text(self):
        assert detect_language("Hi") is None

    def test_returns_none_for_whitespace_only(self):
        assert detect_language("     ") is None


# --------------------------------------------------------------------------- #
# clean_text (full pipeline composition)
# --------------------------------------------------------------------------- #
class TestCleanText:
    def test_full_pipeline_on_messy_html(self):
        raw = "<div>  Hello   <b>Itâ€™s</b>   a\n\n\n\ntest  </div>"
        result = clean_text(raw)
        assert "<div>" not in result
        assert "â€™" not in result
        assert "  " not in result  # no double spaces
        assert result == result.strip()

    def test_idempotent_on_already_clean_text(self):
        clean = "This is already clean text."
        assert clean_text(clean) == clean


# --------------------------------------------------------------------------- #
# is_valid_document / clean_document
# --------------------------------------------------------------------------- #
class TestIsValidDocument:
    def test_rejects_empty_text(self):
        result = is_valid_document("")
        assert result.is_valid is False
        assert result.reason == "empty_after_cleaning"

    def test_rejects_too_short_text(self):
        result = is_valid_document("Too short.", min_chars=200)
        assert result.is_valid is False
        assert "too_short" in result.reason

    def test_accepts_valid_long_english_text(self):
        text = ("This is a long enough English paragraph to pass the minimum "
                 "character threshold for document validity checks. " * 3)
        result = is_valid_document(text, min_chars=200)
        assert result.is_valid is True
        assert result.language == "en"

    def test_rejects_disallowed_language(self):
        text = ("Esto es un texto suficientemente largo en espanol para superar "
                 "el umbral minimo de caracteres requerido. " * 3)
        result = is_valid_document(text, min_chars=200, allowed_languages=("en",))
        assert result.is_valid is False
        assert "language_filtered" in result.reason

    def test_returns_validation_result_dataclass(self):
        result = is_valid_document("short")
        assert isinstance(result, ValidationResult)


class TestCleanDocument:
    def test_returns_tuple_of_text_and_validation(self):
        raw = "<p>" + ("This is valid English content for the pipeline test. " * 5) + "</p>"
        cleaned, validation = clean_document(raw, min_chars=100)
        assert isinstance(cleaned, str)
        assert isinstance(validation, ValidationResult)
        assert "<p>" not in cleaned
        assert validation.is_valid is True

    def test_end_to_end_rejects_junk_input(self):
        cleaned, validation = clean_document("<div></div>")
        assert validation.is_valid is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
