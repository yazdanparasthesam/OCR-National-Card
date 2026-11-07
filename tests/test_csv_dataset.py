"""The CSV → dataset bridge used for real, labelled card sets."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from nidocr.accuracy import load_dataset
from scripts.csv_to_dataset import convert, map_columns


@pytest.fixture()
def labelled_folder(tmp_path):
    """Two card-size images plus a label CSV that also lists a card with no image."""
    images = tmp_path / "cards"
    images.mkdir()
    for name in ("1", "2"):
        cv2.imwrite(str(images / f"{name}.jpg"), np.full((90, 130, 3), 235, np.uint8))
    labels = tmp_path / "labels.csv"
    labels.write_text(
        "id,national_id,first_name,last_name,date_of_birth,father_name,expiry_date\n"
        "1,5116168397,مبینا,اسدی,1380/04/01,نریمان,1409/10/19\n"
        "2,5182864095,سمانه,خزایی,1385/08/04,یزدان,1414/11/23\n"
        "9,9999999999,غایب,نیست,1380/01/01,کسی,1410/01/01\n",
        encoding="utf-8",
    )
    return labels, images, tmp_path / "out"


def test_columns_are_matched_however_they_are_spelled():
    """Underscores, spaces and case must not matter — nor Persian headers."""
    mapping = map_columns(["National ID", "first name", "شماره ملی", "نام خانوادگی", "unknown"])
    assert mapping["National ID"] == "national_code"
    assert mapping["first name"] == "first_name"
    assert mapping["شماره ملی"] == "national_code"
    assert mapping["نام خانوادگی"] == "last_name"
    assert "unknown" not in mapping


def test_csv_becomes_a_loadable_dataset(labelled_folder):
    """Every labelled row with an image becomes one JSON record the benchmark can read."""
    labels, images, out = labelled_folder
    written, skipped = convert(labels, images, out)

    assert written == 2
    assert skipped == ["9"]  # the row without an image is reported, not silently dropped
    payload = json.loads((out / "1.json").read_text(encoding="utf-8"))
    assert payload["image"] == "1.jpg"
    assert payload["fields"] == {
        "national_code": "5116168397",
        "first_name": "مبینا",
        "last_name": "اسدی",
        "birth_date": "1380/04/01",
        "father_name": "نریمان",
        "expiry_date": "1409/10/19",
    }
    records = load_dataset(out)
    assert len(records) == 2
    assert records[0][0].name in {"1.jpg", "2.jpg"}
