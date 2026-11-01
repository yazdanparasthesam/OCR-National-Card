"""Locate the national card inside a photo and warp it to a canonical canvas.

The Iranian national card is a standard ID-1 card: 85.6 × 53.98 mm → aspect ratio
``1.5857``.  Detection uses several independent hypotheses (Canny contours,
Otsu mask contours, min-area rectangles) and scores every quadrilateral by how
close it is to that aspect ratio, how much of the frame it fills and how close
its corners are to 90°.  When nothing convincing is found the full frame is used,
which keeps the pipeline usable for pre-cropped scans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from nidocr.config import Settings, get_settings
from nidocr.vision.enhance import resize_to_max_side

CARD_ASPECT = 85.6 / 53.98


@dataclass
class CardDetection:
    """Result of :func:`detect_card`."""

    quad: np.ndarray  # 4×2 float32, ordered TL → TR → BR → BL
    score: float
    method: str
    aspect: float
    area_ratio: float
    found: bool
    image_size: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)

    def as_points(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.quad]


def order_quad(points: np.ndarray | list[list[float]]) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    if _is_duplicate(ordered):
        # degenerate quad (some points coincide) — fall back to angle ordering
        centre = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
        ordered = pts[np.argsort(angles)]
        ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    return ordered


def _is_duplicate(quad: np.ndarray) -> bool:
    uniq = {tuple(np.round(point, 1)) for point in quad}
    return len(uniq) < 4


def _quad_metrics(quad: np.ndarray, frame_area: float) -> tuple[float, float, float]:
    """Return ``(aspect, area_ratio, rectangularity)`` of a quadrilateral."""
    (tl, tr, br, bl) = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if width < 10 or height < 10:
        return 0.0, 0.0, 0.0
    aspect = max(width, height) / max(1e-6, min(width, height))
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = float(min(1.0, area / max(1.0, frame_area)))
    rectangularity = area / max(1e-6, width * height)
    return float(aspect), area_ratio, float(min(1.0, rectangularity))


def _angle_score(quad: np.ndarray) -> float:
    """1.0 when all four corners are ~90°, 0.0 for a degenerate quad."""
    angles = []
    for index in range(4):
        prev_point = quad[(index - 1) % 4]
        point = quad[index]
        next_point = quad[(index + 1) % 4]
        v1 = prev_point - point
        v2 = next_point - point
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angles.append(abs(np.degrees(np.arccos(cosine)) - 90.0))
    deviation = float(np.mean(angles))
    return float(np.exp(-((deviation / 18.0) ** 2)))


def _touches_border(quad: np.ndarray, width: int, height: int, tolerance: float = 0.01) -> bool:
    """True when the quad hugs every image border (i.e. it is the frame itself)."""
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    margin_x, margin_y = tolerance * width, tolerance * height
    return (
        x0 <= margin_x
        and y0 <= margin_y
        and x1 >= width - 1 - margin_x
        and y1 >= height - 1 - margin_y
    )


def _score(quad: np.ndarray, frame_area: float, expected_aspect: float) -> float:
    aspect, area_ratio, rectangularity = _quad_metrics(quad, frame_area)
    if aspect <= 1.0:
        return 0.0
    aspect_score = float(np.exp(-(((aspect - expected_aspect) / 0.30) ** 2)))
    area_score = min(1.0, area_ratio / 0.45)
    angle = _angle_score(quad)
    score = 0.50 * aspect_score + 0.28 * area_score + 0.12 * angle + 0.10 * rectangularity
    if area_ratio < 0.03:
        score *= 0.3
    return float(round(min(1.0, score), 4))


def _contour_candidates(image: np.ndarray) -> list[np.ndarray]:
    """Quadrilateral hypotheses from edge and intensity based contours."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    height, width = gray.shape[:2]
    min_area = 0.04 * height * width
    candidates: list[np.ndarray] = []

    masks: list[np.ndarray] = []
    edges = cv2.Canny(gray, 40, 130)
    masks.append(cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=2))
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)))
    # the card border is often darker than the desk it lies on
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    masks.append(cv2.dilate(gradient, np.ones((9, 9), np.uint8), iterations=2))

    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
            if len(approx) == 4:
                candidates.append(order_quad(approx.reshape(4, 2)))
            else:
                box = cv2.boxPoints(cv2.minAreaRect(hull))
                candidates.append(order_quad(box))
    return candidates


def detect_card(image: np.ndarray, settings: Settings | None = None) -> CardDetection:
    """Find the card quad. Never raises: falls back to the full frame."""
    settings = settings or get_settings()
    if image is None or image.size == 0:
        raise ValueError("empty image")

    height, width = image.shape[:2]
    working = resize_to_max_side(image, 1000)
    scale = width / working.shape[1]
    frame_area = float(working.shape[0] * working.shape[1])

    working_height, working_width = working.shape[:2]
    best_quad, best_score = None, 0.0
    for quad in _contour_candidates(working):
        score = _score(quad, frame_area, settings.card_aspect)
        if _touches_border(quad, working_width, working_height):
            # The frame is always a "perfect" quadrilateral once the edge map is
            # dilated, so it would win every score on a desk photo.  Keep it (a
            # card can genuinely fill the frame — a scan) but demote it so a real
            # card quad inside the photo wins.
            score *= 0.55
        if score > best_score:
            best_quad, best_score = quad, score

    if best_quad is None:
        return CardDetection(
            quad=order_quad([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]),
            score=0.0,
            method="full_frame",
            aspect=width / height,
            area_ratio=1.0,
            found=False,
            image_size=(width, height),
            notes=["کارت در تصویر پیدا نشد؛ کل تصویر پردازش شد"],
        )

    quad = best_quad * scale
    aspect, area_ratio, _ = _quad_metrics(quad, float(width * height))
    found = best_score >= 0.42 and area_ratio >= 0.05
    if not found:
        return CardDetection(
            quad=order_quad([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]),
            score=best_score,
            method="full_frame_fallback",
            aspect=width / height,
            area_ratio=1.0,
            found=False,
            image_size=(width, height),
            notes=["تشخیص کارت با اطمینان کافی انجام نشد؛ کل تصویر پردازش شد"],
        )
    notes = []
    if abs(aspect - settings.card_aspect) > 0.25:
        notes.append(f"نسبت ابعاد تشخیص داده‌شده ({aspect:.2f}) از کارت استاندارد (1.59) فاصله دارد")
    return CardDetection(
        quad=quad.astype(np.float32),
        score=best_score,
        method="contour",
        aspect=aspect,
        area_ratio=area_ratio,
        found=True,
        image_size=(width, height),
        notes=notes,
    )


def warp_card(
    image: np.ndarray,
    quad: np.ndarray | None = None,
    size: tuple[int, int] | None = None,
    settings: Settings | None = None,
) -> np.ndarray:
    """Perspective-warp the card quad to ``size`` (landscape, canonical canvas)."""
    settings = settings or get_settings()
    size = size or (settings.canonical_width, settings.canonical_height)
    if quad is None:
        quad = np.array([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], dtype=np.float32)
    quad = order_quad(quad)
    (tl, tr, br, bl) = quad
    quad_width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    quad_height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if quad_height > quad_width:  # portrait photo of a landscape card
        quad = np.roll(quad, 1, axis=0)[::-1]
    target_width, target_height = size
    destination = np.array(
        [[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    return cv2.warpPerspective(image, matrix, (target_width, target_height), flags=cv2.INTER_CUBIC)


def estimate_quality(gray: np.ndarray, detection: CardDetection | None = None) -> dict:
    """Cheap image-quality metrics used for warnings in the API and the UI."""
    height, width = gray.shape[:2]
    scaled = resize_to_max_side(gray, 900)
    sharpness = float(cv2.Laplacian(scaled, cv2.CV_64F).var())
    brightness = float(np.mean(scaled))
    contrast = float(np.std(scaled))
    glare_ratio = float(np.mean(scaled > 245))
    warnings: list[str] = []
    if sharpness < 35:
        warnings.append("تصویر تار است؛ دقت استخراج کاهش می‌یابد")
    if brightness < 60:
        warnings.append("تصویر تاریک است")
    if brightness > 225:
        warnings.append("تصویر بیش از حد روشن است")
    if glare_ratio > 0.12:
        warnings.append("انعکاس نور روی کارت وجود دارد")
    if contrast < 25:
        warnings.append("کنتراست تصویر پایین است")
    if min(width, height) < 320:
        warnings.append("رزولوشن تصویر پایین است")
    if detection is not None and not detection.found:
        warnings.extend(detection.notes)
    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "glare_ratio": round(glare_ratio, 4),
        "resolution": (width, height),
        "warnings": warnings,
    }
