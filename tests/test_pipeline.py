"""End-to-end extraction on a rendered card."""

from __future__ import annotations

import numpy as np

from nidocr.fields.definitions import FIELD_SPECS
from nidocr.metrics import character_accuracy, exact_match
from nidocr.ocr.text import normalize_for_compare


def test_extract_reads_every_rendered_field(ocr, sample_card):
    """A rendered card is read almost perfectly (at most one field may miss)."""
    image, fields = sample_card
    result = ocr.extract(image, debug=True)
    assert result.success
    assert result.fields.keys() == FIELD_SPECS.keys()
    assert result.processing_ms > 0
    correct = [key for key, value in fields.items() if exact_match(value, result.get(key))]
    # the renderer produces a clean card: the pipeline must be near perfect on it
    assert len(correct) >= len(fields) - 1, result.flat()


def test_result_carries_geometry_and_confidence(ocr, sample_card):
    """Every found field carries a confidence in 0..100 and a normalised box inside the card."""
    image, _ = sample_card
    result = ocr.extract(image)
    for field in result.ordered_fields():
        assert 0 <= field.confidence <= 100
        if field.found:
            assert field.bbox is not None
            assert 0 <= field.bbox.x <= 1 and 0 <= field.bbox.y <= 1
            assert field.bbox.width > 0 and field.bbox.height > 0


def test_persian_digit_mirror(ocr, sample_card):
    """``value`` stays ASCII while ``value_fa`` is the Persian-digit mirror."""
    image, _ = sample_card
    field = ocr.extract(image).fields["national_code"]
    assert field.value.isascii()
    assert all("۰" <= char <= "۹" or char == "/" for char in field.value_fa)


def test_extract_single_field(ocr, sample_card):
    """Reading one field returns the right value and its ROI image."""
    image, fields = sample_card
    result = ocr.extract_field(image, "national_code", debug=True)
    assert result.field.key == "national_code"
    assert exact_match(fields["national_code"], result.field.value)
    assert result.field.roi_png


def test_extract_accepts_bytes_and_paths(ocr, sample_card_file):
    """Bytes and file paths go through exactly the same pipeline."""
    path, fields = sample_card_file
    from_bytes = ocr.extract(path.read_bytes())
    from_path = ocr.extract(path)
    assert from_bytes.flat() == from_path.flat()


def test_extract_batch(ocr, sample_card_file):
    """The batch helper returns one identical result per input image."""
    path, _ = sample_card_file
    results = ocr.extract_batch([path, path])
    assert len(results) == 2
    assert results[0].flat() == results[1].flat()


def test_describe_lists_the_public_contract(ocr):
    """``describe()`` covers every field with its key, path and label."""
    described = ocr.describe()
    assert {item["key"] for item in described} == set(FIELD_SPECS)
    for item in described:
        assert item["path"] and item["label_fa"] and item["kind"]


def test_blank_image_is_a_clean_failure(ocr):
    """A blank image fails gracefully: ``success=False`` and no fake values."""
    result = ocr.extract(np.full((400, 640, 3), 255, np.uint8))
    assert result.success is False
    assert all(not field.found for field in result.fields.values())


def test_metrics_match_after_normalisation():
    """The metric helpers normalise before comparing and report CER as accuracy."""
    assert exact_match("۱۳۶۰/۰۱/۰۱", "1360/01/01")
    assert normalize_for_compare("نام\u200cخانوادگی") == normalize_for_compare("نام خانوادگی")
    assert character_accuracy("محمدی", "محمدی") == 1.0
    assert character_accuracy("محمدی", "محمد") < 1.0
