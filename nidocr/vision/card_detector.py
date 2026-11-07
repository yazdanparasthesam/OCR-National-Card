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
from nidocr.vision.layout import load_template

CARD_ASPECT = 85.6 / 53.98

#: Both designs are tried when a hypothesis is judged by what its warp *contains*.
CONTENT_TEMPLATES = ("national_card_front", "national_card_smart")
CONTENT_PREVIEW = (340, 215)


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
        """The four corners as a plain list of ``(x, y)`` tuples (JSON friendly)."""
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
    """True when two corners are the same point, i.e. the quad is degenerate."""
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
    """Score a quadrilateral hypothesis: closeness to the card aspect ratio, area and
    right angles, in 0..1. Higher is a better card candidate.
    """
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


def _card_mask_quad(image: np.ndarray) -> np.ndarray | None:
    """Quadrilateral of the biggest bright, low-saturation blob — the card itself.

    Used only as a *rescue* hypothesis: when no contour is convincing (a card on a
    cluttered desk, a card in a hand, a card next to a labelled bottle) the picture is
    either the card or the scene around it, and an Otsu split on saturation and on
    brightness separates the pale card from skin, wood and plastic.
    """
    work = resize_to_max_side(image, 900)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    _, low_saturation = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, bright = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(low_saturation, bright)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == biggest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    quad = (
        approx.reshape(4, 2).astype(np.float32)
        if len(approx) == 4
        else cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    )
    return quad * (image.shape[1] / work.shape[1])


def _content_score(canvas: np.ndarray) -> float:
    """How much a warped canvas looks like *a card*, in 0..1.

    A correct warp carries a portrait in the photo box (15–45 % of it below the card's
    own ink level) and a column of six text rows to its right (2–9 %).  Both designs
    are tried and the better score wins, so it does not matter which one the card is.
    """
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    level = float(np.percentile(gray, 25))
    height, width = gray.shape[:2]
    best = 0.0
    for name in CONTENT_TEMPLATES:
        layout = load_template(name)
        photo = layout.raw.get("decorations", {}).get("photo")
        values = [layout.value_rect(key, width, height) for key in layout.keys()]  # noqa: SIM118
        values = [rect for rect in values if rect]
        if not photo or len(values) < 4:
            continue
        px0, py0, px1, py1 = layout._to_pixels(photo, width, height)
        portrait = float((gray[py0:py1, px0:px1] < level).mean())
        vx0, vy0 = min(r[0] for r in values), min(r[1] for r in values)
        vx1, vy1 = max(r[2] for r in values), max(r[3] for r in values)
        ink = float((gray[vy0:vy1, vx0:vx1] < level).mean())
        portrait_term = max(0.0, 1.0 - abs(portrait - 0.30) / 0.28)
        ink_term = max(0.0, 1.0 - abs(ink - 0.050) / 0.050)
        best = max(best, 0.55 * ink_term + 0.45 * portrait_term)
    return float(best)


def _rescue_quad(image: np.ndarray, frame_area: float, settings: Settings) -> np.ndarray | None:
    """The colour-based card quad, when it convincingly beats the whole frame.

    Called only where the geometric detection gave up, so the full frame is what the
    pipeline would otherwise use: the rescue has to earn its place by *looking* like a
    card (a portrait and a column of text), and a quad that does not is dropped.
    """
    candidate = _card_mask_quad(image)
    if candidate is None:
        return None
    aspect, area_ratio, rectangularity = _quad_metrics(candidate, frame_area)
    if aspect <= 1.0 or area_ratio < 0.08 or rectangularity < 0.55:
        return None
    if np.exp(-(((aspect - settings.card_aspect) / 0.45) ** 2)) < 0.55:
        return None
    canvas = warp_card(image, candidate, size=CONTENT_PREVIEW)
    return candidate if _content_score(canvas) >= 0.55 else None


#: HSV bands of the printed card itself.  An Iranian national card is a pale cyan-blue
#: sheet with a pink band; a hand, a wooden desk, a bottle of water and a bin are all
#: outside these bands or much darker, which is what makes the card separable by colour
#: on photos where its own border is too faint for an edge contour.
CARD_COLOUR_BANDS = (
    (85, 118, 12, 135, 120),  # cyan-blue body: hue, hue, min S, max S, min V
    (138, 178, 18, 145, 140),  # the pink band
)
CARD_PALE_MIN_V = 165
CARD_PALE_MAX_S = 28
#: A warped canvas is accepted as "the card" when this much of its border ring is card.
CARD_BORDER_FILL = 0.80
#: The bar a colour quad has to clear when the geometric quad is not a card either —
#: then colour only has to beat a junk quad, not a real card.
CARD_BORDER_FILL_WEAK = 0.55
BORDER_PROBE = (240, 151)


def _card_colour_mask(image: np.ndarray) -> np.ndarray:
    """Binary mask of the pale cyan/pink sheet, cleaned into one solid blob."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
    mask = ((saturation < CARD_PALE_MAX_S) & (value >= CARD_PALE_MIN_V)).astype(np.uint8)
    for low, high, min_s, max_s, min_v in CARD_COLOUR_BANDS:
        band = (hue >= low) & (hue <= high) & (saturation >= min_s) & (saturation <= max_s) & (value >= min_v)
        mask |= band.astype(np.uint8)
    mask *= 255
    kernel = max(9, image.shape[1] // 45) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    return mask


def _largest_blob_quad(mask: np.ndarray) -> np.ndarray | None:
    """Quadrilateral of the biggest connected blob in the mask."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == biggest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _border_fill(mask: np.ndarray, quad: np.ndarray) -> float:
    """Share of a hypothesis' border ring that is card: 1.0 means the card fills it.

    The warp of the *mask* answers the question directly — if the ring around a warped
    canvas is still mask, the quad is on the card; background bleeding into the ring is
    exactly what a wrong quad looks like.
    """
    warped = warp_card(mask, quad, size=BORDER_PROBE)
    height, width = warped.shape[:2]
    ring = max(2, height // 22)
    inside = float((warped > 127).mean())
    centre = float((warped[ring : height - ring, ring : width - ring] > 127).mean())
    return min(inside, max(centre, 0.0))


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

    # Colour is the second, independent hypothesis.  On the real photos the edge
    # contour very often latches onto a bottle, a hand or a shadow: the card's own
    # border is faint while its pale cyan sheet is unmistakable.  A colour quad is used
    # when the card fills the canvas it produces — the one thing an edge quad cannot
    # claim when it has drifted off the card.
    mask = _card_colour_mask(working)
    colour_quad = _largest_blob_quad(mask)
    colour_fill = 0.0
    if colour_quad is not None:
        colour_aspect, colour_area, colour_rect = _quad_metrics(colour_quad, frame_area)
        if 1.2 <= colour_aspect <= 2.1 and colour_area >= 0.06 and colour_rect >= 0.6:
            colour_fill = _border_fill(mask, colour_quad)
        else:
            colour_quad = None
    geometry_fill = _border_fill(mask, best_quad) if best_quad is not None else 0.0
    geometry_area = _quad_metrics(best_quad, frame_area)[1] if best_quad is not None else 0.0
    # Is the geometric quad a *card*?  A card is found when the edge quad is convincing,
    # is shaped like a card (aspect 1.59 ± 0.25 — a rotated card's bounding quad is
    # 1.1–1.3 and squashes the whole canvas, which is how a national code ended up read
    # from the «نام» row) and covers a plausible part of the picture (a scan or a render
    # is 0.9–1.0, a card in a hand 0.5–0.85).  When it is not, the colour hypothesis
    # only has to beat junk, so it gets the weaker bar.
    geometry_aspect = _quad_metrics(best_quad, frame_area)[0] if best_quad is not None else 0.0
    geometry_is_card = (
        best_score >= 0.42
        and abs(geometry_aspect - settings.card_aspect) <= 0.25
        and geometry_area >= 0.65
    )
    needed = CARD_BORDER_FILL if geometry_is_card else CARD_BORDER_FILL_WEAK

    if (
        colour_quad is not None
        and not geometry_is_card
        and colour_fill >= needed
        and colour_fill > geometry_fill + 0.02
    ):
        quad = colour_quad * scale
        aspect, area_ratio, _ = _quad_metrics(quad, float(width * height))
        return CardDetection(
            quad=quad.astype(np.float32),
            score=round(colour_fill, 4),
            method="colour",
            aspect=aspect,
            area_ratio=area_ratio,
            found=True,
            image_size=(width, height),
            notes=[],
        )

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
        rescue = _rescue_quad(working, frame_area, settings)
        if rescue is not None:
            rescued = rescue * scale
            resc_aspect, resc_area, _ = _quad_metrics(rescued, float(width * height))
            return CardDetection(
                quad=rescued.astype(np.float32),
                score=round(best_score, 4),
                method="mask",
                aspect=resc_aspect,
                area_ratio=resc_area,
                found=True,
                image_size=(width, height),
                notes=["کارت از رنگ و روشنایی جدا شد (لبه‌های آن به‌اندازه کافی واضح نبودند)"],
            )
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
    if quad_height > quad_width:
        # portrait photo of a landscape card: shift the corners by one place, which
        # is a *rotation* of the quadrilateral.  (Reversing them as well would mirror
        # the canvas — mirrored text is unreadable no matter how it is turned, and the
        # difference is invisible to the eye in a thumbnail.)
        quad = np.roll(quad, 1, axis=0)
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
