"""OCR engines and Persian text helpers."""

from __future__ import annotations

from nidocr.ocr.text import (
    clean_ocr_text,
    collapse_spaces,
    digit_ratio,
    has_persian_letters,
    normalize_for_compare,
    normalize_persian,
    only_digits,
    persian_ratio,
    to_ascii_digits,
    to_persian_digits,
)
from nidocr.ocr.tesseract_engine import (
    OCRLine,
    OCRResult,
    TesseractEngine,
    TextToken,
    get_engine,
)

__all__ = [
    "OCRLine",
    "OCRResult",
    "TesseractEngine",
    "TextToken",
    "clean_ocr_text",
    "collapse_spaces",
    "digit_ratio",
    "get_engine",
    "has_persian_letters",
    "normalize_for_compare",
    "normalize_persian",
    "only_digits",
    "persian_ratio",
    "to_ascii_digits",
    "to_persian_digits",
]
