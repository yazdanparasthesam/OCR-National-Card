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


def field_variants(
    roi_gray: np.ndarray,
    *,
    upscale_factor: float = 3.0,
    passes: int = 3,
    pad_value: int = 18,
) -> list[tuple[str, np.ndarray]]:
    """Build the OCR inputs that are voted for one field.

    ``roi_gray`` is expected to come from the *already enhanced* card
    (:func:`enhance_gray`), so only upscaling and binarisation happen here — running
    the background flattening twice would erase thin Persian strokes.

    All variants are padded with white borders: tesseract treats the outer frame as
    a page boundary and needs breathing room around a single text line.
    """
    big = upscale(roi_gray, upscale_factor)
    # a small white frame keeps the ROI from being interpreted as a "page"
    padded = cv2.copyMakeBorder(big, pad_value, pad_value, pad_value, pad_value, cv2.BORDER_CONSTANT, value=255)

    variants: list[tuple[str, np.ndarray]] = [("enhanced", padded)]
    if passes >= 2:
        binary = sauvola(padded, window=max(15, (padded.shape[1] // 12) | 1), k=0.22)
        variants.append(("sauvola", despeckle(binary, min_area_px=8)))
    if passes >= 3:
        binary = otsu(padded)
        variants.append(("otsu", despeckle(binary, min_area_px=8)))
    if passes >= 4:
        adaptive = cv2.adaptiveThreshold(
            padded, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )
        variants.append(("adaptive", despeckle(adaptive, min_area_px=8)))
    return variants[: max(1, passes)]


def card_variants(color_card: np.ndarray) -> dict[str, np.ndarray]:
    """Full-card variants used for label anchoring and previews."""
    gray = to_gray(color_card)
    enhanced = enhance_gray(gray)
    binary = despeckle(sauvola(enhanced, window=31, k=0.2), min_area_px=4)
    return {"gray": gray, "enhanced": enhanced, "binary": binary}
