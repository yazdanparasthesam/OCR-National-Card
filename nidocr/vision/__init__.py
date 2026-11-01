"""OpenCV building blocks: card detection, enhancement, binarisation, layout."""

from __future__ import annotations

from nidocr.vision.binarize import binarize, despeckle, otsu, sauvola
from nidocr.vision.card_detector import (
    CardDetection,
    detect_card,
    estimate_quality,
    order_quad,
    warp_card,
)
from nidocr.vision.enhance import (
    crop_with_padding,
    enhance_gray,
    field_variants,
    normalize_background,
    resize_to_max_side,
    to_gray,
    upscale,
)

__all__ = [
    "CardDetection",
    "binarize",
    "crop_with_padding",
    "despeckle",
    "detect_card",
    "enhance_gray",
    "estimate_quality",
    "field_variants",
    "normalize_background",
    "order_quad",
    "otsu",
    "resize_to_max_side",
    "sauvola",
    "to_gray",
    "upscale",
    "warp_card",
]
