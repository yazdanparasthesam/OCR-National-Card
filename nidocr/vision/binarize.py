"""Binarisation and speckle removal tuned for Persian print on patterned cards."""

from __future__ import annotations

import cv2
import numpy as np

BINARIZE_METHODS = ("sauvola", "otsu", "adaptive")


def sauvola(gray: np.ndarray, window: int = 31, k: float = 0.2, dynamic_range: float = 128.0) -> np.ndarray:
    """Sauvola local threshold (integral-image implementation, O(n))."""
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if window % 2 == 0:
        window += 1
    data = gray.astype(np.float32)
    mean = cv2.boxFilter(data, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)
    sq_mean = cv2.boxFilter(data * data, cv2.CV_32F, (window, window), borderType=cv2.BORDER_REFLECT)
    variance = np.maximum(sq_mean - mean * mean, 0.0)
    std = np.sqrt(variance)
    threshold = mean * (1.0 + k * (std / dynamic_range - 1.0))
    return ((data > threshold) * 255).astype(np.uint8)


def otsu(gray: np.ndarray) -> np.ndarray:
    """Global Otsu threshold."""
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def adaptive(gray: np.ndarray, block_size: int = 31, c: int = 12) -> np.ndarray:
    """Gaussian adaptive threshold — the fourth preprocessing variant of a field ROI."""
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)


def binarize(gray: np.ndarray, method: str = "sauvola", **kwargs) -> np.ndarray:
    """Dispatch to one of the supported binarisation methods."""
    method = (method or "sauvola").lower()
    if method == "sauvola":
        return sauvola(gray, **kwargs)
    if method == "otsu":
        return otsu(gray)
    if method == "adaptive":
        return adaptive(gray, **kwargs)
    raise ValueError(f"unknown binarisation method {method!r}; expected one of {BINARIZE_METHODS}")


def despeckle(binary: np.ndarray, min_area_ratio: float = 0.0, min_area_px: int = 6) -> np.ndarray:
    """Remove tiny connected components (guilloche specks) while keeping letters.

    The threshold is deliberately tiny: Persian letters are *separated* strokes and
    their dots are only a few pixels, so an aggressive filter would destroy the text
    instead of the background pattern.
    """
    if binary.ndim != 2:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)
    # components of the *ink* are dark, so invert, clean, invert back
    ink = 255 - binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count <= 1:
        return binary
    area_limit = max(min_area_px, int(min_area_ratio * ink.shape[0] * ink.shape[1]))
    mask = np.zeros_like(ink)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= area_limit:
            mask[labels == index] = 255
    return 255 - mask


def count_ink(binary: np.ndarray) -> float:
    """Fraction of dark pixels (used by layout refinement)."""
    if binary.size == 0:
        return 0.0
    return float(np.mean(binary < 128))
