"""Rotation normalisation for the warped card canvas.

A photo of a national card is very often not upright: phones store a 90° EXIF
orientation, the user holds the camera sideways, or the card is simply upside down
on the desk.  The perspective warp only removes the *perspective* — the text inside
the canvas can still be rotated by 90/180/270°, which is fatal for OCR (Persian
letters rotated by 90° are read as Latin garbage).

Two independent signals are used, cheapest first:

* :func:`rotation_from_osd` — one Tesseract OSD call (~0.3 s) that asks Tesseract
  which way the text runs.  Cheap, but its confidence is low on small crops.
* :func:`probe_scores` / :func:`best_rotation` — a one-pass OCR probe of each
  candidate rotation, scored by how much Persian text it yields.  Slower
  (≈0.35 s per angle) but reliable, so it is only used to *rescue* a card whose
  first reading came out broken (see :meth:`NationalCardOCR.extract`).

The public entry points are :func:`rotate` (clockwise, the same convention as
Tesseract's ``Rotate:`` field) and :func:`upright` (OSD-driven, returns the angle it
applied).
"""

from __future__ import annotations

import numpy as np

from nidocr.ocr.text import PERSIAN_LETTERS

#: All orientations a card can have relative to the template, clockwise degrees.
ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)

#: Minimum OSD orientation confidence before its answer is trusted on its own.
#: Measured on rendered cards: 3.1–5.1 for the correct answer, so the bar stays low
#: and the caller still verifies the reading afterwards.
OSD_MIN_CONFIDENCE = 0.5


def rotate(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate ``image`` by ``angle`` degrees **clockwise** (0/90/180/270).

    The convention matches Tesseract's ``Rotate:`` field, so an OSD answer can be
    applied directly.  Non-multiples of 90 are rejected: a card is never slightly
    rotated at this point (the warp already straightened it).
    """
    angle %= 360
    if angle not in ROTATIONS:
        raise ValueError(f"angle must be one of {ROTATIONS}, got {angle}")
    if angle == 0:
        return image
    # np.rot90 turns counter-clockwise, so a clockwise turn is a negative quarter turn
    return np.ascontiguousarray(np.rot90(image, k=-(angle // 90)))


def rotation_from_osd(canvas: np.ndarray, engine) -> tuple[int, float]:
    """Ask Tesseract's OSD which way the text runs: ``(clockwise_degrees, confidence)``.

    Returns ``(0, confidence)`` when OSD is unavailable or its answer is not one of
    the four right angles; a confidence below :data:`OSD_MIN_CONFIDENCE` is still
    reported so the caller can decide to verify it.
    """
    info = engine.osd(canvas)
    angle = int(info.get("rotate", 0) or 0)
    confidence = float(info.get("confidence", 0.0) or 0.0)
    if angle not in ROTATIONS:
        return 0, confidence
    return angle, confidence


def text_score(page) -> float:
    """Score one OCR result as evidence that the text is upright.

    Persian letters and digits count in favour (weighted by their confidence);
    Latin letters count against, because sideways or upside-down Persian is what
    Tesseract tends to render as Latin words (``"355"``, ``"jancs"``).  The value is
    unbounded; only comparisons between rotations are meaningful.
    """
    score = 0.0
    for token in getattr(page, "tokens", ()):
        text = getattr(token, "text", "") or ""
        if not text:
            continue
        weight = max(float(getattr(token, "confidence", 0.0) or 0.0), 0.0) / 100.0
        letters = sum(1 for ch in text if ch in PERSIAN_LETTERS)
        digits = sum(1 for ch in text if ch.isdigit())
        latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        score += (letters + 0.7 * digits) * weight - 0.8 * latin
    return score


def probe_scores(
    canvas: np.ndarray,
    engine,
    *,
    angles: tuple[int, ...] = ROTATIONS,
    psm: int = 6,
) -> dict[int, float]:
    """One cheap OCR pass per rotation → ``{angle: text_score}``.

    ``psm 6`` (a uniform block of text) is used because it returns whole lines at
    once, which is ~4× faster than the per-field reads and plenty to tell an
    upright card from a rotated one.  Results are cached inside the engine, so
    repeating the probe for the chosen angle is free.
    """
    scores: dict[int, float] = {}
    for angle in angles:
        page = engine.read_page(rotate(canvas, angle), psm=psm, variant=f"orient-{angle}")
        scores[angle] = text_score(page)
    return scores


def best_rotation(
    canvas: np.ndarray,
    engine,
    *,
    angles: tuple[int, ...] = ROTATIONS,
    psm: int = 6,
    margin: float = 0.05,
) -> tuple[int, dict[int, float]]:
    """Pick the rotation with the highest probe score: ``(angle, all_scores)``.

    ``margin`` is a relative lead the winner must have over the *current* (0°)
    orientation to be worth re-reading the whole card — without it, two equally
    bad readings would trigger a pointless second pass.
    """
    scores = probe_scores(canvas, engine, angles=angles, psm=psm)
    winner = max(scores, key=lambda key: (scores[key], key == 0))
    if winner != 0 and scores[winner] < scores.get(0, 0.0) + margin:
        return 0, scores
    return winner, scores


def upright(canvas: np.ndarray, engine) -> tuple[np.ndarray, int, float]:
    """Return ``(upright_canvas, applied_angle, osd_confidence)``.

    The OSD answer is applied only when it is one of the four right angles *and*
    carries at least :data:`OSD_MIN_CONFIDENCE`; otherwise the canvas is returned
    untouched and the pipeline keeps its rescue path (probe + re-read) available.
    """
    angle, confidence = rotation_from_osd(canvas, engine)
    if angle == 0 or confidence < OSD_MIN_CONFIDENCE:
        return canvas, 0, confidence
    return rotate(canvas, angle), angle, confidence


def looks_rotated(result) -> bool:
    """Heuristic: does this reading look like the card was not upright?

    Used to decide whether the expensive rescue path is worth running.  A card is
    considered suspicious when almost nothing was found, or when nothing at all
    passed a validator *and* the mean confidence is low — the signature of letters
    read sideways (the validators are what normally paper over a hard image).
    """
    found = int(getattr(result, "total_fields", 0) or 0)
    validated = int(getattr(result, "validated_fields", 0) or 0)
    confidence = float(getattr(result, "mean_confidence", 0.0) or 0.0)
    if found <= 2:
        return True
    return validated == 0 and confidence < 72.0


__all__ = [
    "OSD_MIN_CONFIDENCE",
    "ROTATIONS",
    "best_rotation",
    "looks_rotated",
    "probe_scores",
    "rotate",
    "rotation_from_osd",
    "text_score",
    "upright",
]
