"""Numeric fields: glyph segmentation, date separators and the ROI preparation.

These cover the bug that made a real card's dates unreadable: the card's ``/`` is a
thin diagonal that the classifier calls a ``۸``, so a date row arrived with two glyphs
too many and the calendar validator never saw a date at all.
"""

from __future__ import annotations

import cv2
import numpy as np

from nidocr.fields.definitions import get_field_spec
from nidocr.ocr.digit_reader import Glyph, _date_limits, _separator_alternatives
from nidocr.pipeline import GLYPH_TALL_FACTOR, GLYPH_UPSCALE
from nidocr.vision.enhance import LINE_TARGET_HEIGHT


def glyphs_from(text: str, confidences: list[float] | None = None) -> list[Glyph]:
    """Build glyphs from a string, optionally with per-glyph confidences."""
    confidences = confidences or [95.0] * len(text)
    return [
        Glyph(box=(index * 10, 0, index * 10 + 8, 20), value=char, confidence=confidences[index])
        for index, char in enumerate(text)
    ]


def test_date_limits_follow_the_iso_structure():
    """A date is grouped 4/2/2; anything else falls back to a single group."""
    assert _date_limits(8) == (4, 2, 2)
    assert _date_limits(10) == (10, 0, 0)


def test_separator_alternatives_relabel_the_doubtful_glyphs():
    """The two least confident glyphs become separators when that yields 4/2/2."""
    # 14,06,03,14 (eight digits) plus the two separators read as «۸»
    text = "۱۴۰۶۸۰۳۸۱۴"
    confidences = [95, 95, 95, 95, 40, 95, 95, 38, 95, 95]
    alternatives = _separator_alternatives(glyphs_from(text, confidences), limits=(4, 2, 2))
    assert "۱۴۰۶/۰۳/۱۴" in alternatives
    # every alternative keeps all digits and only re-labels two glyphs
    assert all(alt.count("/") == 2 for alt in alternatives)


def test_separator_alternatives_need_the_right_glyph_count():
    """A row with the wrong number of glyphs is left to the other heuristics."""
    assert _separator_alternatives(glyphs_from("۱۴۰۶۸۰۳۸"), limits=(4, 2, 2)) == []


def test_separator_alternatives_survive_a_calendar_check():
    """The repaired string is a calendar-valid date, which is what the vote needs."""
    spec = get_field_spec("birth_date")
    alternatives = _separator_alternatives(
        glyphs_from("۱۳۷۲۸۰۹۸۱۲", [95, 95, 95, 95, 44, 95, 95, 39, 95, 95]),
        limits=_date_limits(8),
    )
    outcomes = [spec.finalize(alt) for alt in alternatives]
    assert any(outcome.validated and outcome.value == "1372/09/12" for outcome in outcomes)


def test_glyph_line_flattens_and_normalises_the_height(ocr):
    """A tall patterned ROI becomes a flat line of the canonical height."""
    rng = np.random.default_rng(3)
    roi = np.full((120, 420), 150, np.uint8)
    roi += rng.integers(-18, 18, roi.shape, dtype=np.int16).astype(np.uint8)
    for index in range(6):
        x = 40 + index * 55
        cv2.putText(roi, "5", (x, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 30, 5)
    line = ocr._glyph_line(roi)
    assert line.shape[0] == LINE_TARGET_HEIGHT
    assert line.mean() > 150  # background flattened towards white
    tall = ocr._glyph_line(roi, height_factor=GLYPH_TALL_FACTOR)
    assert tall.shape[0] == int(round(LINE_TARGET_HEIGHT * GLYPH_TALL_FACTOR))
    assert GLYPH_UPSCALE >= 1.0


def test_date_reading_of_a_rendered_card_is_exact(ocr, sample_card):
    """The glyph reader returns the rendered birth date, separators included."""
    image, truth = sample_card
    birth = ocr.extract(image).fields["birth_date"]
    assert birth.found
    assert birth.value.count("/") == 2
    assert birth.value == truth["birth_date"]
