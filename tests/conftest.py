"""Shared fixtures: a synthetic card, an OCR pipeline and a cached card photo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nidocr.config import Settings
from nidocr.synthetic.renderer import random_spec, render_card
from nidocr.vision.layout import load_template


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(cache_ocr=False)


@pytest.fixture(scope="session")
def card_template():
    return load_template("national_card_front")


@pytest.fixture(scope="session")
def sample_card(card_template):
    """A rendered card + the ground truth it was rendered from."""
    spec = random_spec(2024)
    image = render_card(spec, card_template)
    return image, spec.fields


@pytest.fixture(scope="session")
def sample_card_file(tmp_path_factory, sample_card) -> tuple[Path, dict]:
    image, fields = sample_card
    directory = tmp_path_factory.mktemp("cards")
    path = directory / "card.png"
    import cv2

    cv2.imwrite(str(path), image)
    (directory / "card.json").write_text(json.dumps({"image": path.name, "fields": fields}), encoding="utf-8")
    return path, fields


@pytest.fixture(scope="session")
def ocr(settings):
    from nidocr.pipeline import NationalCardOCR

    return NationalCardOCR(settings)
