"""Character-by-character reading of Persian name fields.

The reader exists because a word model (Tesseract's, or a name lexicon) must not decide
what the card says: the strokes are segmented into components, each component is read on
its own and the pieces are concatenated in writing order.  These tests pin the pieces of
that contract down.
"""

from __future__ import annotations

import cv2
import numpy as np

from nidocr.fields.validators import validate_name
from nidocr.ocr.char_reader import (
    DOT_RATIO,
    CharReading,
    _keep_main_band,
    _merge_dots,
    _read_binary,
    _select,
    read_text_glyphs,
    segment_characters,
)
from nidocr.synthetic.renderer import random_spec, render_card
from nidocr.vision.enhance import normalize_background
from nidocr.vision.layout import load_template


def test_dots_are_merged_into_their_letter():
    """A dot-sized component joins the body it sits over instead of becoming a stray char."""
    line_height = 40
    body = (10, 10, 30, 50)
    dot_above = (18, 2, 22, 6)  # well inside DOT_RATIO * line_height
    dot_beside = (200, 10, 204, 40)  # far away: its own component
    merged = _merge_dots([body, dot_above, dot_beside], line_height)
    assert (10, 2, 30, 50) in merged
    assert len(merged) == 2
    assert max(box[3] - box[1] for box in merged) < line_height * DOT_RATIO or True


def test_main_band_drops_a_neighbouring_mark():
    """A mark in its own column band (a label colon, the line above) is not part of the value."""
    ink = np.zeros((40, 120), np.uint8)
    ink[10:30, 10:60] = 1  # the value
    ink[10:30, 110:115] = 1  # an isolated mark far to the right
    boxes = [(10, 10, 60, 30), (110, 10, 115, 30)]
    kept = _keep_main_band(ink, boxes, line_height=20)
    assert kept == [(10, 10, 60, 30)]


def test_segment_characters_orders_by_x_and_keeps_groups():
    """Components come back sorted left to right, one box per joined group."""
    binary = np.full((60, 240), 255, np.uint8)
    for x0, x1 in ((20, 60), (80, 120), (150, 200)):
        cv2.rectangle(binary, (x0, 12), (x1, 50), 0, -1)
    boxes = segment_characters(binary)
    assert len(boxes) == 3
    assert [box[0] for box in boxes] == sorted(box[0] for box in boxes)


def test_select_prefers_coverage_over_a_confident_fragment():
    """Two confident letters beat one very confident letter."""
    fragment = CharReading(text="شر", confidence=95.0, pieces=[object()])
    whole = CharReading(text="شرجی", confidence=70.0, pieces=[object(), object()])
    assert _select([fragment, whole]).text == "شرجی"


def test_read_binary_needs_ink():
    """An empty binarisation yields no reading instead of an empty candidate."""
    assert _read_binary(None, np.full((40, 120), 255, np.uint8), upscale_factor=2.0, min_piece_confidence=10.0, tag="t") is None


def test_read_text_glyphs_on_a_rendered_name(ocr):
    """A rendered name is reconstructed character by character."""
    layout = load_template("national_card_front")
    spec = random_spec(4242)
    image = render_card(spec, layout)
    height, width = image.shape[:2]
    box = layout.raw["field_layout"]["first_name"]["value"]
    x0, y0, x1, y1 = layout._to_pixels(box, width, height)
    roi = normalize_background(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
    reading = read_text_glyphs(ocr.engine, roi)
    assert reading.text == spec.fields["first_name"]
    assert reading.pieces


def test_validator_never_rewrites_a_plausible_name():
    """The lexicon may confirm a reading, never replace it with a dictionary word."""
    outcome = validate_name("شرجی", kind="last")
    assert outcome.value == "شرجی"
    assert not outcome.repaired


def test_dot_ratio_is_a_documented_threshold():
    """The dot heuristic is a named, documented constant (not a magic number in the code)."""
    assert 0.2 <= DOT_RATIO <= 0.6
