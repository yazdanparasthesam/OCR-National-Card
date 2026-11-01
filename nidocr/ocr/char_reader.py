"""Character-by-character reader for Persian *name* fields.

The numeric reader splits a row into single digits and reads them one at a time, which
is why a national code reads glyph by glyph. Names were left to the whole-line decoder
of Tesseract, whose Persian model prefers words it has seen — which is exactly how a
dictionary ends up "correcting" the card into a different surname.

This module applies the same idea to Persian text: the line is split into ink
components (with the dots and the hamza attached to the letter they belong to), each
piece is read on its own with a single-character page-segmentation mode, and the pieces
are simply concatenated right to left — the logical order of Persian. No lexicon is
consulted here: the value is whatever the strokes say, and the caller decides how much
to trust it.

Persian letters are cursive, so one component can hold two or three joined letters. The
reader therefore asks Tesseract for the *word* of the component too, and keeps whichever
of the two readings is more confident; a joined pair comes back as two characters and the
concatenation is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from nidocr.ocr.tesseract_engine import TesseractEngine
from nidocr.ocr.text import PERSIAN_LETTERS, ZWNJ, clean_ocr_text, normalize_persian
from nidocr.vision.binarize import despeckle, otsu, sauvola

Rect = tuple[int, int, int, int]

#: Components smaller than this fraction of the line height are dots, not letters.
DOT_RATIO = 0.42
#: Letters a name may contain. Kept for reference: restricting Tesseract with it measured
#: *worse* on blurred photos (it turns a doubtful glyph into nothing instead of a letter),
#: so the reader runs unrestricted and the shape rule below handles the bare alef.
LETTER_WHITELIST = "".join(sorted(PERSIAN_LETTERS)) + ZWNJ
#: Minimum ink for a component to be considered at all (relative to the line height).
MIN_AREA_RATIO = 0.004


@dataclass
class CharPiece:
    """One segmented piece of a text line with its own reading."""

    box: Rect
    text: str
    confidence: float


@dataclass
class CharReading:
    """Result of the character-by-character reading of one text line."""

    text: str = ""
    confidence: float = 0.0
    pieces: list[CharPiece] = field(default_factory=list)
    alternatives: list[str] = field(default_factory=list)
    variant: str = "chars"

    def __bool__(self) -> bool:
        """A reading is truthy when it produced any text."""
        return bool(self.text)


def _select(readings: list[CharReading]) -> CharReading | None:
    """Pick the reading with the best *coverage*, tolerating a lower mean confidence.

    A variant that saw every component beats one that confidently returned half the
    word, as long as its characters are credible.
    """
    if not readings:
        return None
    best_mean = max(reading.confidence for reading in readings)
    eligible = [reading for reading in readings if reading.confidence >= 0.65 * best_mean]
    return max(eligible, key=lambda reading: (len(reading.pieces), reading.confidence))


def _read_binary(
    engine: TesseractEngine,
    binary: np.ndarray,
    *,
    upscale_factor: float,
    min_piece_confidence: float,
    tag: str,
) -> CharReading | None:
    """Segment one binarised line and read every component right to left."""
    boxes = segment_characters(binary)
    if not boxes:
        return None
    widths = [box[2] - box[0] for box in boxes] or [1]
    median_width = float(np.median(widths)) or 1.0
    pieces: list[CharPiece] = []
    for index, box in enumerate(reversed(boxes)):  # right to left = reading order
        x0, y0, x1, y1 = box
        # black on white, the polarity Tesseract expects: the inverted crop loses the
        # thin joins between letters (measured on a rendered card: «حسین» vs nothing)
        crop = binary[max(y0 - 2, 0) : y1 + 2, max(x0 - 2, 0) : x1 + 2]
        if crop.size == 0:
            continue
        text, confidence = _read_piece(
            engine,
            crop,
            upscale_factor=upscale_factor,
            tag=f"{tag}{index}",
            wide=(box[2] - box[0]) > 1.35 * median_width,
        )
        if not text or confidence < min_piece_confidence:
            continue
        pieces.append(CharPiece(box=box, text=text, confidence=confidence))
    if not pieces:
        return None
    text = normalize_persian("".join(piece.text for piece in pieces))
    confidence = float(np.mean([piece.confidence for piece in pieces]))
    return CharReading(text=text, confidence=confidence, pieces=pieces)


def _keep_main_band(ink: np.ndarray, boxes: list[Rect], line_height: int) -> list[Rect]:
    """Keep the components that belong to the value, not to a neighbouring mark.

    A name ROI often catches the colon of its label, or the end of the line above. Those
    marks live in their own column band, separated by a gap wider than a letter space, so
    only the widest continuous band of ink columns is kept.
    """
    if not boxes:
        return []
    columns = ink.any(axis=0).astype(np.int8)
    bands: list[tuple[int, int]] = []
    start = None
    gap = 0
    # letter spacing inside a word can be wide (ا, د, ر do not join), so a gap only
    # splits the line when it is a good fraction of the full line height
    max_gap = max(4, int(0.45 * line_height))
    for index, flag in enumerate(columns):
        if flag:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                bands.append((start, index - gap))
                start = None
                gap = 0
    if start is not None:
        bands.append((start, len(columns) - 1))
    if not bands:
        return boxes
    # a neighbouring mark (the colon of the label) is a *small* band; a word is spread over
    # bands of comparable ink, so drop the bands that carry almost nothing
    column_ink = ink.sum(axis=0)
    masses = [int(column_ink[start : end + 1].sum()) for start, end in bands]
    largest = max(masses) or 1
    keep = [band for band, mass in zip(bands, masses, strict=True) if mass >= 0.15 * largest]
    kept = [box for box in boxes if any(box[0] >= left - 2 and box[2] <= right + 2 for left, right in keep)]
    return kept or boxes


def _merge_dots(boxes: list[Rect], line_height: int) -> list[Rect]:
    """Attach dot-sized components to the letter body that sits above or below them.

    Persian letters carry one to three dots; reading them separately produces stray
    ``.`` characters, and dropping them loses the difference between ``ب`` and ``پ``.
    A dot is merged into the widest body whose horizontal span overlaps it.
    """
    if not boxes:
        return []
    bodies = [box for box in boxes if (box[3] - box[1]) >= DOT_RATIO * line_height]
    if not bodies:
        return sorted(boxes, key=lambda box: box[0])
    merged: list[Rect] = list(bodies)
    for box in boxes:
        if box in bodies:
            continue
        cx = (box[0] + box[2]) / 2.0
        best_index, best_distance = -1, None
        for index, body in enumerate(merged):
            overlap = min(box[2], body[2]) - max(box[0], body[0])
            distance = min(abs(cx - body[0]), abs(cx - body[2]))
            if overlap > 0:
                distance = 0.0
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        if best_index < 0 or (best_distance or 0.0) > 0.9 * line_height:
            continue
        body = merged[best_index]
        merged[best_index] = (
            min(body[0], box[0]),
            min(body[1], box[1]),
            max(body[2], box[2]),
            max(body[3], box[3]),
        )
    return sorted(merged, key=lambda box: box[0])


def segment_characters(binary: np.ndarray, *, max_pieces: int = 24) -> list[Rect]:
    """Split a binarised Persian line into character components, left to right.

    Returns the boxes in *reading* order is **not** applied here: the boxes come back
    sorted by ``x`` so the caller can walk them right to left, which is how Persian is
    written and how the characters have to be concatenated.
    """
    ink = (binary < 128).astype(np.uint8)
    if ink.sum() < 12:
        return []
    line_height = int(np.count_nonzero(ink.any(axis=1)))
    if line_height < 4:
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes: list[Rect] = []
    min_area = max(4, int(MIN_AREA_RATIO * line_height * line_height))
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + width), int(y + height)))
    pieces = _merge_dots(boxes, line_height)
    pieces = _keep_main_band(ink, pieces, line_height)
    # Persian letters are cursive: a component is a *group* of one to four joined
    # letters, never a single letter, so the group is kept together and read as a
    # group (psm 7) instead of being cut at an arbitrary column.
    return sorted(pieces, key=lambda box: box[0])[:max_pieces]


#: Width/height ratio below which a component is a plain vertical stroke (``ا``).
STROKE_RATIO = 0.28
#: Confidence reported for a letter decided from its shape instead of from OCR.
STROKE_CONFIDENCE = 92.0


def _is_vertical_stroke(crop: np.ndarray) -> bool:
    """True for a bare vertical bar, which is the letter ``ا``.

    A single alef is a few pixels wide; no page-segmentation mode of this Tesseract build
    reads it (it answers ``۱``, or nothing at all), so the shape is the evidence: names
    contain no digits, and nothing else on a card's name line is that thin and that tall.
    """
    height, width = crop.shape[:2]
    if height < 8 or width > STROKE_RATIO * height:
        return False
    ink = (crop < 128)
    if not ink.any():
        return False
    # the stroke must run through most of the height, not be a speck
    rows = np.where(ink.any(axis=1))[0]
    return len(rows) >= 0.7 * height


def _read_piece(
    engine: TesseractEngine,
    crop: np.ndarray,
    *,
    upscale_factor: float,
    tag: str,
    wide: bool = False,
) -> tuple[str, float]:
    """Read one segmented piece with the modes that fit its shape.

    A narrow component is one letter (``psm 10``); a wide one is a joined group of
    letters, and reading it as a line (``psm 7``) is what keeps the group intact instead
    of returning one of its letters.
    """
    # no white frame around the component: with a frame this Tesseract build returns an
    # empty string for Persian text (measured again on a bare crop vs. a framed crop)
    big = cv2.resize(
        crop,
        None,
        fx=upscale_factor,
        fy=upscale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    best_text, best_confidence = "", 0.0
    for psm in ((7, 8) if wide else (10, 8)):
        result = engine.read(big, psm=psm, variant=f"{tag}{psm}")
        text = clean_ocr_text(result.text).replace(" ", "")
        if not text:
            continue
        if result.confidence > best_confidence:
            best_text, best_confidence = text, float(result.confidence)
    if _is_vertical_stroke(crop) and not best_text.replace(" ", "").isalpha():
        # OCR gave nothing useful (or a digit) for a bare vertical bar: the shape is the
        # better evidence, and a name contains no digits
        return "ا", STROKE_CONFIDENCE
    return best_text, best_confidence


def read_text_glyphs(
    engine: TesseractEngine,
    roi_gray: np.ndarray,
    *,
    upscale_factor: float = 2.0,
    min_confidence: float = 20.0,
    min_piece_confidence: float = 12.0,
    heights: tuple[int, ...] = (44, 60),
) -> CharReading:
    """Read a text line character by character and concatenate the pieces.

    Every line height in ``heights`` is tried on two binarisations (Sauvola and Otsu,
    the pair the numeric reader votes on) and the most confident reading wins: the same
    word is legible at one height and not at the other, so asking both is cheaper than
    guessing. Pieces below ``min_piece_confidence`` are dropped instead of guessed.
    """
    if roi_gray is None or roi_gray.size == 0:
        return CharReading()
    from nidocr.vision.enhance import fit_line  # local: enhance imports binarize

    lines: list[tuple[str, np.ndarray]] = []
    if 40 <= roi_gray.shape[0] <= 72:
        lines.append(("raw", roi_gray))
    for height in heights:
        if abs(roi_gray.shape[0] - height) > 3:
            lines.append((f"h{height}", fit_line(roi_gray, target_height=height)))
    if not lines:
        lines.append(("raw", roi_gray))

    readings: list[CharReading] = []
    for line_name, line in lines:
        window = max(15, (line.shape[1] // 10) | 1)
        binaries = (
            (f"sa{line_name}", despeckle(sauvola(line, window=window, k=0.2), min_area_px=4)),
            (f"ot{line_name}", despeckle(otsu(line), min_area_px=4)),
        )
        for binary_name, binary in binaries:
            reading = _read_binary(
                engine,
                binary,
                upscale_factor=upscale_factor,
                min_piece_confidence=min_piece_confidence,
                tag=binary_name,
            )
            if reading is not None:
                readings.append(reading)
    if not readings:
        return CharReading()
    # coverage beats a lucky fragment: a reading that saw every component wins over one
    # that confidently returned half the word, as long as its characters are credible
    best_mean = max(reading.confidence for reading in readings)
    eligible = [reading for reading in readings if reading.confidence >= 0.65 * best_mean]
    best = max(eligible, key=lambda reading: (len(reading.pieces), reading.confidence))
    if best.confidence < min_confidence:
        return CharReading()
    return best
