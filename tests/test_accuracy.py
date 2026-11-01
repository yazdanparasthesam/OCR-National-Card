"""Tests for :mod:`nidocr.accuracy` — measured and estimated accuracy."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from nidocr.accuracy import (
    AccuracyReport,
    FieldAccuracy,
    compare_card,
    estimate_card,
    estimate_field,
    evaluate,
    load_dataset,
    run_synthetic,
)


@dataclass
class FakeField:
    """Minimal stand-in for ``FieldResult`` (the estimator only reads six attributes)."""

    key: str
    label_fa: str = ""
    value: str = "123"
    found: bool = True
    validated: bool = True
    repaired: bool = False
    confidence: float = 95.0


def test_evaluate_reports_per_field_and_overall(sample_card_file, ocr):
    """``evaluate`` returns one row per measured field plus the card totals."""
    report = evaluate([sample_card_file], ocr=ocr, difficulty="clean")
    assert report.cards == 1
    assert {row.key for row in report.fields} == set(sample_card_file[1])
    assert 0.0 <= report.exact <= 1.0
    assert 0.0 <= report.characters <= 1.0
    assert report.characters >= report.exact  # characters is the softer metric
    assert report.mean_latency_ms > 0


def test_evaluate_accepts_plain_field_mappings(sample_card_file, ocr):
    """A bare ``{field: value}`` payload (no ``fields`` wrapper) is understood too."""
    path, fields = sample_card_file
    report = evaluate([(path, dict(fields))], ocr=ocr)
    assert report.cards == 1
    assert report.fields


def test_progress_callback_is_called_per_card(sample_card_file, ocr):
    """The progress hook receives (done, total, label) for every card."""
    calls: list[tuple[int, int, str]] = []
    evaluate([sample_card_file], ocr=ocr, progress=lambda *args: calls.append(args))
    assert calls and calls[0][:2] == (1, 1)
    assert isinstance(calls[0][2], str)


def test_report_serialisation_has_percentages(sample_card_file, ocr):
    """``as_dict``/``to_markdown`` expose the same numbers as percentages."""
    report = evaluate([sample_card_file], ocr=ocr)
    payload = report.as_dict()
    assert payload["exact"] == pytest.approx(report.exact * 100, abs=0.01)
    assert payload["fields"][0]["samples"] >= 1
    markdown = report.to_markdown(title="T")
    assert markdown.startswith("# T")
    assert "field exact match" in markdown


def test_estimate_field_penalises_missing_and_repaired():
    """A missing field estimates 0 %, a repaired one less than a clean read."""
    clean = estimate_field(FakeField("national_code"))
    repaired = estimate_field(FakeField("national_code", repaired=True))
    missing = estimate_field(FakeField("national_code", found=False))
    assert missing.exact == 0.0
    assert 0.0 < repaired.exact < clean.exact <= 0.99
    assert clean.characters >= clean.exact


def test_estimate_field_respects_difficulty():
    """Photo estimates never exceed clean estimates for the same reading."""
    field = FakeField("serial_number")
    assert estimate_field(field, difficulty="photo").exact < estimate_field(field, difficulty="clean").exact
    assert estimate_field(field, difficulty="hard").exact < estimate_field(field, difficulty="photo").exact


def test_estimate_card_is_the_mean_of_its_fields():
    """The card total is the mean of the per-field estimates (same rule as measured runs)."""
    fields = [FakeField("national_code"), FakeField("last_name"), FakeField("gender")]
    estimate = estimate_card(fields, difficulty="clean")
    mean = sum(row.exact for row in estimate["fields"]) / len(estimate["fields"])
    assert estimate["exact"] == pytest.approx(mean)
    assert [row.key for row in estimate["fields"]] == ["national_code", "last_name", "gender"]


def test_estimate_handles_an_empty_card():
    """No fields is not a division by zero."""
    estimate = estimate_card([], difficulty="clean")
    assert estimate["exact"] == 0.0 and estimate["characters"] == 0.0


def test_compare_card_counts_exact_fields(sample_card_file, ocr):
    """A card compared with its own truth scores 100 % and reports every field."""
    path, fields = sample_card_file
    result = ocr.extract(path)
    score = compare_card(result, fields)
    assert score["measured"] == len(fields)
    assert score["correct"] == len(fields)
    assert score["exact_rate"] == 1.0

    wrong = compare_card(result, {**fields, "national_code": "0000000000"})
    assert wrong["correct"] == len(fields) - 1
    assert 0.0 < wrong["characters"] < 1.0


def test_compare_card_ignores_fields_left_blank(sample_card_file, ocr):
    """Blank inputs are skipped instead of counted as misses."""
    path, fields = sample_card_file
    result = ocr.extract(path)
    score = compare_card(result, {"national_code": "", "gender": fields["gender"]})
    assert score["measured"] == 1


def test_load_dataset_reads_image_json_pairs(sample_card_file):
    """``load_dataset`` resolves the image name written inside each JSON."""
    path, fields = sample_card_file
    records = load_dataset(path.parent)
    assert records and records[0][0].name == path.name
    assert records[0][1]["fields"] == fields


def test_load_dataset_ignores_orphan_json(tmp_path):
    """A JSON without its image is skipped rather than half-loaded."""
    (tmp_path / "only_truth.json").write_text(json.dumps({"fields": {"national_code": "1"}}), encoding="utf-8")
    assert load_dataset(tmp_path) == []


def test_run_synthetic_measures_rendered_cards(ocr):
    """The self-contained run renders, reads and scores its own cards."""
    report = run_synthetic(count=2, difficulty="clean", seed=7, ocr=ocr)
    assert report.cards == 2
    assert len(report.fields) == 7
    assert report.exact > 0.5  # clean renders must not collapse
    assert isinstance(report, AccuracyReport)


def test_field_accuracy_row_is_rounded():
    """``FieldAccuracy.as_row`` rounds percentages to one decimal."""
    row = FieldAccuracy(key="k", label_fa="ل", samples=3, exact=0.9967, characters=0.9812, confidence=88.44)
    payload = row.as_row()
    assert payload == {
        "key": "k",
        "label_fa": "ل",
        "samples": 3,
        "exact": 99.7,
        "characters": 98.1,
        "confidence": 88.4,
        "validated": 0.0,
    }
