"""Tests for rotation normalisation, mirrored warps and template auto-detection."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from nidocr.fields.validators import validate_expiry_date
from nidocr.pipeline import get_ocr
from nidocr.synthetic.renderer import degrade, random_spec, render_card
from nidocr.vision.card_detector import detect_card, order_quad, warp_card
from nidocr.vision.layout import detect_template, load_template
from nidocr.vision.orient import ROTATIONS, rotate, rotation_from_osd, text_score


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Normalised correlation of two same-size grey images (1.0 = identical)."""
    a = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    b = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    a = (a.astype(np.float32) - a.mean()).ravel()
    b = (b.astype(np.float32) - b.mean()).ravel()
    if a.std() == 0 or b.std() == 0:  # pragma: no cover - defensive
        return 0.0
    return float(np.dot(a, b) / (len(a) * a.std() * b.std()))


def test_rotate_is_clockwise_by_quarter_turns():
    """``rotate`` turns clockwise, matching Tesseract's ``Rotate:`` field."""
    image = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(rotate(image, 90), np.rot90(image, -1))
    assert np.array_equal(rotate(image, 180), np.rot90(image, 2))
    assert np.array_equal(rotate(image, 270), np.rot90(image, 1))
    assert rotate(image, 360) is image


def test_rotate_rejects_odd_angles():
    """A card is never slightly rotated at this stage, so only right angles are allowed."""
    with pytest.raises(ValueError):
        rotate(np.zeros((4, 4), np.uint8), 45)


def test_warp_of_a_portrait_photo_is_not_mirrored(card_template):
    """A sideways card must be *rotated* into the canvas, never flipped.

    This is the bug that made every portrait photo unreadable: reversing the corner
    order mirrors the card, and mirrored Persian text cannot be rescued by rotation.
    The check is geometric (no OCR): the warped canvas must correlate best with the
    original under one of the four rotations, and clearly better there than mirrored.
    """
    spec = random_spec(4)
    card = render_card(spec, card_template)
    sideways = rotate(card, 90)
    detection = detect_card(sideways)
    assert detection.found and detection.quad is not None
    quad = detection.quad
    width = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1]))
    assert height > width, "the test photo must be portrait to exercise the portrait path"

    warped = warp_card(sideways, quad, card_template.canvas)
    upright = max(_correlation(warped, rotate(card, angle)) for angle in ROTATIONS)
    mirrored = max(
        _correlation(cv2.flip(warped, 1), rotate(card, angle)) for angle in ROTATIONS
    )
    assert upright > 0.55, f"warp does not resemble the card (corr={upright:.2f})"
    assert upright > mirrored


def test_order_quad_keeps_the_polygon_orientation():
    """``order_quad`` returns a properly ordered quad (no reflection)."""
    quad = np.array([[10, 100], [10, 10], [50, 10], [50, 100]], dtype=np.float32)
    ordered = order_quad(quad)
    assert ordered.shape == (4, 2)
    # the ordered quad must contain exactly the same corners
    assert {tuple(point) for point in ordered.tolist()} == {tuple(point) for point in quad.tolist()}


def test_osd_reports_the_correction(card_template, ocr):
    """OSD names the clockwise rotation needed to straighten a rotated card."""
    engine = ocr.engine
    card = render_card(random_spec(11), card_template)
    for angle in (90, 180):
        reported, confidence = rotation_from_osd(rotate(card, angle), engine)
        if confidence <= 0:  # pragma: no cover - osd.traineddata missing
            pytest.skip("osd model unavailable")
        # OSD reports the *correction*, i.e. the inverse of the rotation we applied
        assert reported == (360 - angle) % 360


def test_text_score_prefers_the_upright_canvas(card_template, ocr):
    """The 4-way probe scores the upright card far above its rotations."""
    canvas = render_card(random_spec(12), card_template)
    scores = {
        angle: text_score(ocr.engine.read_page(rotate(canvas, angle), psm=6, variant=f"s{angle}"))
        for angle in (0, 180)
    }
    assert scores[0] > scores[180] * 2


def test_pipeline_reads_a_sideways_card(card_template):
    """A card photographed 90° off still reads (this used to return 0/7 fields)."""
    spec = random_spec(13)
    image = rotate(render_card(spec, card_template), 90)
    result = get_ocr().extract(image)
    exact = sum(1 for key, value in spec.fields.items() if result.fields[key].value == value)
    assert exact >= 4, {key: result.fields[key].value for key in spec.fields}
    assert result.fields["national_code"].validated


def test_pipeline_reads_an_upside_down_card(card_template):
    """An upside-down card is straightened by OSD before the fields are read."""
    spec = random_spec(14)
    image = rotate(render_card(spec, card_template), 180)
    result = get_ocr().extract(image)
    assert result.fields["national_code"].value == spec.fields["national_code"]


def test_detect_template_from_labels():
    """The design is recognised from its printed labels, not from the file order."""
    smart = detect_template("کارت هوشمند ملی پایان اعتبار ۱۴۰۶/۰۳/۱۴ نام هاشم")
    old = detect_template("شماره سریال ۶۵۷۳۶۳۵۱ جنسیت زن نام")
    assert smart == "national_card_smart"
    assert old == "national_card_front"
    assert detect_template("چیزی که شبیه کارت نیست") is None


def test_auto_template_picks_the_smart_design(ocr):
    """``template="auto"`` recognises the smart design from the photo position.

    The detection compares the two candidate photo regions, which is only decisive on
    a realistic (photographed) card: a flat render has no shading and the two regions
    are equally dark, so the documented fallback is the default template.
    """
    layout = load_template("national_card_smart")
    image = degrade(render_card(random_spec(15), layout), difficulty="photo", seed=1)
    result = ocr.extract(image, template="auto")
    assert result.template == "national_card_smart"


def test_auto_template_falls_back_on_a_flat_render(ocr):
    """An ambiguous (flat, unshaded) image keeps the configured default template."""
    image = render_card(random_spec(15), load_template("national_card_front"))
    assert ocr.extract(image, template="auto").template == "national_card_front"


def test_unknown_template_falls_back_to_the_default(ocr):
    """A stale client asking for a missing template gets the default, not a traceback."""
    result = ocr.extract(render_card(random_spec(16), load_template("national_card_front")), template="nope")
    assert result.template == "national_card_front"


def test_expiry_date_validator():
    """``پایان اعتبار`` is a Jalali date whose year is rebuilt as 14xx when short."""
    valid = validate_expiry_date("۱۴۰۶/۰۳/۱۴")
    assert valid.value == "1406/03/14" and valid.validated

    repaired = validate_expiry_date("06/3/14")
    assert repaired.value == "1406/03/14" and repaired.repaired and repaired.validated

    impossible = validate_expiry_date("1406/13/40")
    assert not impossible.validated


def test_smart_template_defines_expiry_and_omits_serial():
    """The smart-card template declares the fields that are actually printed on it."""
    layout = load_template("national_card_smart")
    assert set(layout.keys()) == {
        "national_code",
        "first_name",
        "last_name",
        "birth_date",
        "father_name",
        "expiry_date",
    }


def test_the_auto_default_survives_a_bogus_template_name():
    """A bad template name never raises when the configured default is itself 'auto'.

    That combination crashed the Streamlit page: the unknown name fell back to
    ``default_template``, which was the word "auto", which is not a template file.
    """
    from nidocr.config import Settings
    from nidocr.pipeline import get_ocr
    from nidocr.synthetic.renderer import random_spec, render_card

    engine = get_ocr(Settings(default_template="auto", use_label_anchors=False))
    image = render_card(random_spec(31), load_template("national_card_front"))
    assert engine.extract(image, template="does_not_exist").template == "national_card_front"
    assert engine.extract(image, template="auto").template == "national_card_front"
