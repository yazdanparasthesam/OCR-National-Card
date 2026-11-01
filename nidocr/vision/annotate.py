"""Debug/UI rendering: solid field boxes for confidence and ROI previews."""

from __future__ import annotations

import base64

import cv2
import numpy as np

# BGR palettes
COLOR_VALID = (94, 197, 34)  # green
COLOR_REPAIRED = (0, 170, 245)  # amber
COLOR_INVALID = (86, 86, 240)  # red
COLOR_NEUTRAL = (200, 200, 200)


def field_color(validated: bool, confidence: float, repaired: bool = False):
    if validated and confidence >= 60:
        return COLOR_VALID
    if repaired or validated:
        return COLOR_REPAIRED
    return COLOR_INVALID


def draw_overlay(
    card: np.ndarray,
    boxes: dict[str, tuple[int, int, int, int]],
    *,
    labels: dict[str, str] | None = None,
    colors: dict[str, tuple[int, int, int]] | None = None,
    alpha: float = 0.35,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw translucent filled boxes with Persian labels under each box."""
    overlay = card.copy()
    labels = labels or {}
    colors = colors or {}
    for key, (x0, y0, x1, y1) in boxes.items():
        color = colors.get(key, COLOR_NEUTRAL)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, cv2.FILLED)
        cv2.rectangle(card, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, alpha, card, 1 - alpha, 0)
    for key, (x0, y0, x1, y1) in boxes.items():
        text = labels.get(key)
        if not text:
            continue
        color = colors.get(key, COLOR_NEUTRAL)
        cv2.putText(
            blended,
            text,
            (x0 + 2, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return blended


def encode_png_base64(image: np.ndarray, *, quality: int = 88) -> str:
    """Encode a BGR image as a base64 PNG/JPEG data payload (no prefix)."""
    if image is None or image.size == 0:
        return ""
    success, buffer = cv2.imencode(".png", image)
    if not success:  # pragma: no cover - defensive
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def encode_jpeg_base64(image: np.ndarray, quality: int = 82) -> str:
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:  # pragma: no cover
        return ""
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def data_uri(image: np.ndarray, fmt: str = "png") -> str:
    payload = encode_png_base64(image) if fmt == "png" else encode_jpeg_base64(image)
    return f"data:image/{fmt};base64,{payload}" if payload else ""


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
