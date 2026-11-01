"""Geometry: card detection, warping, ROI refinement and overlays."""

from __future__ import annotations

import cv2
import numpy as np

from nidocr.vision.annotate import data_uri, draw_overlay, encode_jpeg_base64, encode_png_base64, field_color
from nidocr.vision.card_detector import detect_card, order_quad, warp_card
from nidocr.vision.enhance import (
    crop_with_padding,
    enhance_gray,
    field_variants,
    normalize_background,
    to_gray,
)
from nidocr.vision.layout import load_template, refine_rect, value_rect_from_label


def _card_on_desk(scale: float = 0.7) -> np.ndarray:
    """Build a synthetic «card lying on a desk» photo with a known card area."""
    desk = np.full((900, 1400, 3), (70, 90, 110), dtype=np.uint8)
    card = np.full((int(640 * scale), int(1012 * scale), 3), (238, 240, 244), dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (card.shape[1] - 1, card.shape[0] - 1), (120, 130, 140), 3)
    y0 = (900 - card.shape[0]) // 2
    x0 = (1400 - card.shape[1]) // 2
    desk[y0 : y0 + card.shape[0], x0 : x0 + card.shape[1]] = card
    return desk


def test_order_quad_is_tl_tr_br_bl():
    """Corners are always ordered top-left, top-right, bottom-right, bottom-left."""
    points = [[100, 100], [10, 10], [100, 10], [10, 100]]
    quad = order_quad(np.array(points, dtype=np.float32))
    assert quad[0].tolist() == [10.0, 10.0]
    assert quad[2].tolist() == [100.0, 100.0]


def test_detects_the_card_inside_a_desk_photo():
    """The detector finds the card (not the frame) and its aspect is card-like."""
    image = _card_on_desk()
    detection = detect_card(image)
    assert detection.found
    assert detection.method == "contour"
    assert 1.3 < detection.aspect < 1.9


def test_warp_produces_the_canonical_canvas():
    """The perspective warp returns exactly the template canvas size."""
    layout = load_template("national_card_front")
    detection = detect_card(_card_on_desk())
    warped = warp_card(_card_on_desk(), detection.quad, size=layout.canvas)
    assert warped.shape[:2] == (layout.canvas[1], layout.canvas[0])


def test_detection_never_raises_on_blank_images():
    """Blank/noisy images still produce a (degenerate) quad instead of an exception."""
    for image in (
        np.zeros((10, 10, 3), np.uint8),
        np.full((600, 400, 3), 255, np.uint8),
        np.random.default_rng(0).integers(0, 255, (640, 480, 3), dtype=np.uint8),
    ):
        detection = detect_card(image)
        assert detection.quad.shape == (4, 2)


def test_refine_rect_tightens_around_the_ink():
    """ROI refinement shrinks the box onto the printed text."""
    canvas = np.full((200, 400), 255, np.uint8)
    cv2.putText(canvas, "test", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    refined = refine_rect(canvas, (20, 40, 380, 180))
    assert refined[2] - refined[0] < 360
    assert refined[3] - refined[1] < 140


def test_refine_rect_keeps_width_for_free_text():
    """With ``allow_columns=False`` the horizontal extent is preserved for names."""
    canvas = np.full((80, 400), 255, np.uint8)
    cv2.putText(canvas, "AB", (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    refined = refine_rect(canvas, (0, 0, 400, 80), allow_columns=False)
    assert refined[2] - refined[0] >= 395


def test_value_rect_from_label_stays_inside_the_canvas():
    """Label-derived value boxes never leave the canvas."""
    for direction in ("below", "left"):
        rect = value_rect_from_label((600, 100, 720, 130), (1012, 638), direction=direction)
        assert 0 <= rect[0] < rect[2] <= 1012
        assert 0 <= rect[1] < rect[3] <= 638


def test_enhance_pipeline_shapes():
    """Background flattening/enhancement keep the shape and dtype of the image."""
    gray = to_gray(_card_on_desk())
    enhanced = enhance_gray(gray)
    assert enhanced.shape == gray.shape
    assert enhanced.dtype == np.uint8
    flattened = normalize_background(gray)
    assert flattened.mean() > 100


def test_field_variants_are_upscaled_and_binary():
    """The three preprocessing variants are upscaled and (for two of them) binarised."""
    roi = np.full((40, 200), 240, np.uint8)
    cv2.putText(roi, "123456", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    variants = field_variants(roi, upscale_factor=3.0, passes=3)
    assert [name for name, _ in variants] == ["enhanced", "sauvola", "otsu"]
    for _, image in variants:
        assert image.shape[0] > roi.shape[0] * 2.5


def test_crop_with_padding_never_exceeds_the_image():
    """Padding outside the image is clipped instead of crashing."""
    image = np.zeros((100, 100), np.uint8)
    crop = crop_with_padding(image, (-20, -20, 130, 130), pad_ratio=0.5)
    assert crop.size > 0
    assert crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_overlay_and_encoders():
    """The overlay painter and the base64/PNG/JPEG encoders produce usable images."""
    image = np.full((120, 200, 3), 245, np.uint8)
    overlay = draw_overlay(image, {"national_code": (10, 10, 100, 40)}, labels={"national_code": "NID"}, colors={"national_code": (0, 120, 0)})
    assert overlay.shape == image.shape
    assert encode_png_base64(image)
    assert encode_jpeg_base64(image)
    assert data_uri(image, "png").startswith("data:image/png;base64,")
    assert field_color(True, 99) != field_color(False, 10)
