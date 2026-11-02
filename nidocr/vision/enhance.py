"""Grayscale enhancement used before OCR.

Iranian national cards are printed on a guilloche (lathe-work) background with a
colour gradient and a hologram; naive binarisation destroys thin Persian letters.
The chain used here is:

1. optional colour-channel selection (the card ink is the darkest channel),
2. background flattening by morphological closing + division,
3. CLAHE for local contrast,
4. unsharp masking to rebuild stroke edges,
5. upscaling (tesseract likes tall x-height glyphs for Persian).

``field_variants`` then produces the different images that are voted in the OCR
multi-pass.
"""

from __future__ import annotations

import cv2
import numpy as np

from nidocr.vision.binarize import despeckle, otsu, sauvola


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single channel uint8 image."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so that the longer side is at most ``max_side`` (never upscales)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)


def upscale(image: np.ndarray, factor: float) -> np.ndarray:
    """Resize by ``factor`` using cubic interpolation (falls back to identity)."""
    if factor <= 1.0:
        return image
    return cv2.resize(
        image,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_CUBIC if image.shape[0] * factor < 2000 else cv2.INTER_LANCZOS4,
    )


def normalize_background(gray: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """Flatten the illumination and the guilloche pattern (white-paper effect).

    The closing kernel must be large compared to the strokes but small compared to
    the image, otherwise it eats the text of a small ROI.  It therefore adapts to
    the input size.
    """
    if kernel_size is None:
        kernel_size = min(31, max(5, min(gray.shape[:2]) // 6))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), 9)
    normalized = cv2.divide(gray, background, scale=255)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def remove_shadows(gray: np.ndarray) -> np.ndarray:
    """Divide by a heavily blurred copy — removes gradients and shadows."""
    blur = cv2.GaussianBlur(gray, (0, 0), 21)
    return cv2.divide(gray, blur, scale=255)


def unsharp(gray: np.ndarray, amount: float = 1.5, radius: float = 1.2) -> np.ndarray:
    """Unsharp mask: add ``amount × (image − blur)`` back to the image to rebuild stroke edges."""
    blurred = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)


def enhance_roi(gray: np.ndarray, *, clahe_clip: float = 2.0) -> np.ndarray:
    """Contrast + sharpening for an *already background-flattened* ROI."""
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)
    sharpened = unsharp(contrasted, amount=1.0, radius=1.0)
    # 3×3 median removes the remaining guilloche specks without eating strokes
    return cv2.medianBlur(sharpened, 3)


def enhance_gray(gray: np.ndarray, *, clahe_clip: float = 2.5) -> np.ndarray:
    """Full enhancement chain for a whole card (background flattening included)."""
    return enhance_roi(normalize_background(gray), clahe_clip=clahe_clip)


def strip_annotation_overlay(color: np.ndarray, *, min_coverage: float = 0.004) -> np.ndarray:
    """Erase the coloured boxes that an earlier run drew *on top of* the card image.

    A screenshot of the app's own annotated result carries the field rectangles in the
    pixels, and their four perfectly straight sides run across the very value they point
    at, which wrecks segmentation.  Two tests must both pass before a pixel is touched:
    enough of the image is saturated orange/green (a real card photo scores about 0.1 % of
    its pixels there, an annotated screenshot about 2 %), and most of that colour lies on
    long straight lines (0.41 measured on an annotated screenshot, at most 0.12 for the
    warm tint of an ordinary photo, which the colour test alone would wrongly accept).
    """
    if color.ndim != 3:
        return color
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    overlay = ((hue >= 12) & (hue <= 40) & (saturation > 110) & (value > 140)).astype(np.uint8)
    if float(overlay.mean()) < min_coverage:
        return color
    height, width = overlay.shape[:2]
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(11, int(0.25 * width)), 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(11, int(0.25 * height))))
    ruled = cv2.bitwise_or(
        cv2.morphologyEx(overlay, cv2.MORPH_OPEN, horizontal),
        cv2.morphologyEx(overlay, cv2.MORPH_OPEN, vertical),
    )
    if float(ruled.sum()) / max(1, int(overlay.sum())) < 0.25:
        return color
    ruled = cv2.dilate(ruled, np.ones((5, 5), np.uint8), iterations=1)
    return cv2.inpaint(color, ruled * 255, 6, cv2.INPAINT_NS)


def crop_with_padding(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    pad_ratio: float = 0.12,
    bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    """Crop ``rect`` = ``(x0, y0, x1, y1)`` adding proportional padding."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = rect
    pad_x = max(2, int(round((x1 - x0) * pad_ratio)))
    pad_y = max(2, int(round((y1 - y0) * pad_ratio)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return image[0:1, 0:1].copy()
    crop = image[y0:y1, x0:x1]
    if bounds is not None:
        crop = cv2.resize(crop, bounds, interpolation=cv2.INTER_CUBIC)
    return crop


LINE_TARGET_HEIGHT = 44
"""Height in pixels a field ROI is scaled to before OCR.

Tesseract's line finder misbehaves at both ends of the size range: a 20-px line has
too few pixels per stroke and a 80-px line is rejected outright (``psm 7`` returns an
empty string, no error). 44 px sits in the middle of the band that reads reliably on
every card we tested, so the ROI is always normalised to it instead of being
multiplied by a magnification that depends on the canvas size.
"""

LINE_TARGET_HEIGHTS = (44, 60)
"""Line heights the vote is built on.

The smaller height is best on clean scans (the strokes are thin and Tesseract's line
finder is happy), the larger one keeps more detail on blurred phone photos, so both are
produced and the field vote picks the winner. Two scales cost one extra OCR call per
binarisation and measurably reduce the "which card resolution did I get" luck.
"""

LINE_PAD = 0
"""White border kept around the trimmed line.

Zero on purpose: this build of Tesseract drops the whole line ("``psm 7`` returns an
empty string") as soon as a white frame surrounds Persian text, which was measurably
worse than a bare crop. The trim already leaves a few pixels of margin around the ink.
"""


def fit_line(
    roi_gray: np.ndarray,
    *,
    target_height: int = LINE_TARGET_HEIGHT,
    pad_value: int = LINE_PAD,
    min_ink_px: int = 6,
) -> np.ndarray:
    """Trim a field ROI to its ink and scale it to a canonical text-line height.

    The trim removes the empty margins and the guilloche that surrounds the value, and
    the resize normalises the glyph size, which is what makes a reading independent of
    the card resolution (a 470-px photo and a 1012-px scan produce the same input).
    A ROI without enough ink is returned untouched, so callers can still decide it is
    empty.
    """
    if roi_gray.size == 0:
        return roi_gray
    binary = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    columns = np.where((binary > 0).sum(axis=0) > 1)[0]
    rows = np.where((binary > 0).sum(axis=1) > 1)[0]
    if len(columns) < min_ink_px or len(rows) < 3:
        return roi_gray
    x0, x1 = max(int(columns[0]) - 5, 0), min(int(columns[-1]) + 6, roi_gray.shape[1])
    y0, y1 = max(int(rows[0]) - 4, 0), min(int(rows[-1]) + 5, roi_gray.shape[0])
    line = roi_gray[y0:y1, x0:x1]
    if line.size == 0:
        return roi_gray
    factor = target_height / float(line.shape[0])
    if abs(factor - 1.0) > 0.02:
        interpolation = cv2.INTER_AREA if factor < 1.0 else cv2.INTER_CUBIC
        line = cv2.resize(
            line,
            (max(int(round(line.shape[1] * factor)), 1), max(int(round(line.shape[0] * factor)), 1)),
            interpolation=interpolation,
        )
    # a hair of white keeps the glyph edges away from the image border
    return cv2.copyMakeBorder(line, pad_value, pad_value, pad_value, pad_value, cv2.BORDER_CONSTANT, value=255)


def field_variants(
    roi_gray: np.ndarray,
    *,
    passes: int = 3,
) -> list[tuple[str, np.ndarray]]:
    """Build the OCR inputs that are voted for one field.

    ``roi_gray`` is expected to come from the *already enhanced* card
    (:func:`enhance_gray`), so only normalisation and binarisation happen here —
    running the background flattening twice would erase thin Persian strokes.

    Every variant starts from :func:`fit_line`, i.e. a canonical text height, which is
    what makes a reading independent of the card resolution (a phone photo is
    supersampled, a scan is not).  The binarisations are produced at each height in
    :data:`LINE_TARGET_HEIGHTS`; ``passes`` still controls how many of them exist.
    """
    variants: list[tuple[str, np.ndarray]] = []
    for index, target in enumerate(LINE_TARGET_HEIGHTS):
        if index and passes < 3:
            break
        suffix = "" if index == 0 else "_wide"
        padded = fit_line(roi_gray, target_height=target)
        variants.append((f"enhanced{suffix}", padded))
        if passes >= 2:
            binary = sauvola(padded, window=max(15, (padded.shape[1] // 12) | 1), k=0.22)
            variants.append((f"sauvola{suffix}", despeckle(binary, min_area_px=8)))
        if passes >= 3:
            variants.append((f"otsu{suffix}", despeckle(otsu(padded), min_area_px=8)))
    return variants[: max(1, passes * 2)]


def card_variants(color_card: np.ndarray) -> dict[str, np.ndarray]:
    """Full-card variants used for label anchoring and previews."""
    gray = to_gray(color_card)
    enhanced = enhance_gray(gray)
    binary = despeckle(sauvola(enhanced, window=31, k=0.2), min_area_px=4)
    return {"gray": gray, "enhanced": enhanced, "binary": binary}
