"""Persian normalisation helpers."""

from __future__ import annotations

from nidocr.ocr.text import (
    collapse_spaces,
    digit_ratio,
    has_persian_letters,
    normalize_for_compare,
    normalize_persian,
    only_digits,
    persian_ratio,
    sort_tokens_ltr,
    sort_tokens_rtl,
    to_ascii_digits,
    to_persian_digits,
)


def test_arabic_letters_fold_to_persian():
    """Arabic yeh/kaf are folded to their Persian forms."""
    assert normalize_persian("علي") == "علی"
    assert normalize_persian("كتاب") == "کتاب"


def test_diacritics_and_bidi_marks_are_removed():
    """Diacritics and invisible bidi marks are stripped."""
    assert normalize_persian("مُحَمَّد") == "محمد"
    assert normalize_persian("\u200fعلی\u200e") == "علی"


def test_digit_conversion_roundtrip():
    """Persian/Arabic/Latin digits convert in both directions."""
    assert to_ascii_digits("۱۲۳۴۵") == "12345"
    assert to_ascii_digits("١٢٣") == "123"
    assert to_persian_digits("0123456789") == "۰۱۲۳۴۵۶۷۸۹"
    assert to_ascii_digits(to_persian_digits("1360/01/01")) == "1360/01/01"


def test_normalize_for_compare_ignores_presentation():
    """ZWNJ and punctuation do not change the comparison form."""
    assert normalize_for_compare("نام خانوادگی") == normalize_for_compare("نام\u200cخانوادگی")
    assert normalize_for_compare("  محمدی،  ") == "محمدی"


def test_token_sorting_follows_writing_direction():
    """Token helpers order words right-to-left or left-to-right by their x position."""
    tokens = [("الف", 100), ("ب", 200)]
    assert sort_tokens_rtl(tokens) == "ب الف"
    assert sort_tokens_ltr(tokens) == "الف ب"


def test_character_class_helpers():
    """The digit/Persian ratio helpers classify mixed strings correctly."""
    assert only_digits("۱۲ab۳") == "123"
    assert has_persian_letters("کارت ۱۲۳")
    assert not has_persian_letters("12345")
    assert persian_ratio("سلام ۱۲") > 0.5
    assert digit_ratio("سلام ۱۲") > 0.2
    assert collapse_spaces("  یک   دو ") == "یک دو"
