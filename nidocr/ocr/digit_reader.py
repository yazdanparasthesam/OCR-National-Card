"""Glyph level reader for numeric card fields.

Tesseract's Persian model is excellent for words but surprisingly weak on *runs* of
Persian digits: the line-recognition decoder prefers letter sequences and drops or
mutates digits (measured: 20–50 % character accuracy on a 10-digit national code).

Splitting the numeric field into individual glyphs and running tesseract in
single-character mode (``--psm 10``) with a digit whitelist fixes this completely
(measured: >98 % per glyph on synthetic and photographed cards).  Segmenting a line
of Persian digits is easy because Persian digits never join, so a simple column
projection is enough.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from nidocr.ocr.digit_classifier import PersianDigitClassifier, get_digit_classifier
from nidocr.ocr.tesseract_engine import TesseractEngine
from nidocr.ocr.text import normalize_persian
from nidocr.vision.binarize import despeckle, otsu, sauvola
from nidocr.vision.enhance import enhance_roi, upscale  # noqa: F401

Rect = tuple[int, int, int, int]


@dataclass
class Glyph:
    """One segmented glyph of a numeric field: its box in the upscaled ROI, the chosen
    value, the confidence that produced it and the runner-up ``(digit, share)`` options.
    """
    box: Rect
    value: str
    confidence: float
    options: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class GlyphReading:
    """The reading of a whole numeric field: the string, its confidence, the glyphs that
    built it and the alternative strings that the field validator may prefer.
    """
    text: str = ""
    confidence: float = 0.0
    glyphs: list[Glyph] = field(default_factory=list)
    variant: str = "glyphs"
    alternatives: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """A reading is truthy when it produced any text."""
        return bool(self.text)

    @property
    def digits_only(self) -> str:
        """The reading with separators and any stray characters removed."""
        return "".join(char for char in self.text if char.isdigit())


def _runs(flags: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Group consecutive True flags into ``(start, end)`` runs, tolerating gaps up to ``max_gap``.
    Used on column/row ink profiles to find glyph boundaries.
    """
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, index - gap))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _split_wide(ink: np.ndarray, box: Rect, median_width: int) -> list[Rect]:
    """Split a box that contains two touching glyphs at its emptiest column."""
    x0, y0, x1, y1 = box
    while x1 - x0 > 1.7 * median_width:
        columns = ink[y0:y1, x0:x1].sum(axis=0)
        midpoint = len(columns) // 2
        window = max(2, len(columns) // 8)
        left = max(0, midpoint - window)
        right = min(len(columns), midpoint + window)
        cut = left + int(np.argmin(columns[left:right])) if right > left else midpoint
        if cut <= 1 or cut >= len(columns) - 1:
            break
        split = x0 + cut
        return [(x0, y0, split, y1), (split, y0, x1, y1)]
    return [box]


def segment_digit_glyphs(
    binary: np.ndarray,
    *,
    max_gap: int = 1,
    min_height_ratio: float = 0.32,
    max_glyphs: int = 20,
) -> list[Rect]:
    """Split a binarised line of Persian digits into glyph boxes (left→right)."""
    ink = (binary < 128).astype(np.uint8)
    if ink.sum() < 5:
        return []
    height = ink.shape[0]
    column_ink = ink.sum(axis=0)
    flags = column_ink > max(1, int(0.015 * height))
    boxes: list[Rect] = []
    for start, end in _runs(flags, max_gap=max_gap):
        if end - start < 1:
            continue
        rows = ink[:, start : end + 1].sum(axis=1)
        filled = np.where(rows > 0)[0]
        if len(filled) == 0:
            continue
        top, bottom = int(filled[0]), int(filled[-1]) + 1
        boxes.append((int(start), top, int(end) + 1, bottom))
    if not boxes:
        return []

    heights = np.array([box[3] - box[1] for box in boxes])
    widths = np.array([box[2] - box[0] for box in boxes])
    median_height = float(np.median(heights))
    median_width = float(np.median(widths))

    kept: list[Rect] = []
    for box in boxes:
        box_height = box[3] - box[1]
        box_width = box[2] - box[0]
        if box_height < min_height_ratio * median_height:
            continue  # noise, dot, date separator
        if box_width < 0.16 * median_width and box_height < 0.6 * median_height:
            continue
        if box_width > 1.55 * median_width:
            kept.extend(_split_wide(ink, box, median_width))
        else:
            kept.append(box)
    kept.sort(key=lambda box: box[0])
    kept = align_vertically(kept)
    kept = _keep_main_cluster(kept)
    return kept[:max_glyphs]


def align_vertically(boxes: list[Rect], tolerance: float = 0.45) -> list[Rect]:
    """Keep the glyphs of the number, drop specks that sit elsewhere vertically.

    The *centre* is used, not the baseline: Persian ``۰`` is much shorter than the
    other digits and ``۷``/``۹`` have descenders, so neither height nor baseline alone
    separates a digit from a background speck.
    """
    if len(boxes) < 3:
        return boxes
    centres = np.array([(box[1] + box[3]) / 2.0 for box in boxes], dtype=np.float32)
    heights = np.array([box[3] - box[1] for box in boxes], dtype=np.float32)
    median_centre = float(np.median(centres))
    median_height = float(np.median(heights)) or 1.0
    kept = [box for box in boxes if abs((box[1] + box[3]) / 2.0 - median_centre) <= tolerance * median_height]
    return kept or boxes


def _filter_heights(boxes: list[Rect], low: float = 0.45, high: float = 1.45) -> list[Rect]:
    """Drop components whose height cannot belong to the digit line.

    ``۰`` is short and ``۷``/``۹`` descend, so the accepted band is wide — but a
    fragment (a piece of the label above) or a component that spans two lines is out.
    """
    if len(boxes) < 3:
        return boxes
    heights = np.array([box[3] - box[1] for box in boxes], dtype=np.float32)
    median_height = float(np.median(heights)) or 1.0
    kept = [box for box in boxes if low * median_height <= (box[3] - box[1]) <= high * median_height]
    return kept or boxes


def is_separator_glyph(patch: np.ndarray) -> bool:
    """True when a narrow component is a date separator (``/``, ``-``) not a digit.

    A separator is a thin *diagonal* stroke, so the horizontal centroid of its ink
    travels as you scan down the glyph; the Persian ``۱`` is a straight vertical bar.
    """
    if patch is None or patch.size == 0:
        return False
    ink = patch < 128 if patch.dtype == np.uint8 else patch.astype(np.uint8) < 128
    height, width = ink.shape[:2]
    if height < 4 or width < 2 or width / float(height) > 0.5:
        return False
    centroids = []
    for row in range(height):
        columns = np.flatnonzero(ink[row])
        if columns.size:
            centroids.append(float(columns.mean()))
    if len(centroids) < height * 0.6:
        return False
    drift = (max(centroids) - min(centroids)) / max(1.0, float(width))
    return drift > 0.45


def _keep_main_cluster(boxes: list[Rect], max_gap_ratio: float = 1.6) -> list[Rect]:
    """Drop boxes that are isolated from the digit run (background specks).

    Digits of one number are evenly spaced; a speck separated by a gap wider than a
    glyph is not part of the number and must not become a digit.
    """
    if len(boxes) < 3:
        return boxes
    widths = np.array([box[2] - box[0] for box in boxes], dtype=np.float32)
    median_width = float(np.median(widths)) or 1.0
    max_gap = max(4.0, max_gap_ratio * median_width)
    clusters: list[list[Rect]] = [[boxes[0]]]
    for box in boxes[1:]:
        previous = clusters[-1][-1]
        gap = box[0] - previous[2]
        if gap <= max_gap:
            clusters[-1].append(box)
        else:
            clusters.append([box])
    best = max(clusters, key=lambda cluster: (len(cluster), sum(b[2] - b[0] for b in cluster)))
    return best


def _model_width(width: int) -> int:
    """Side length the classifier expects for a glyph patch (square, resized from the box)."""
    if width % 2 == 0:
        width += 1
    return int(np.clip(width, 21, 61))


def _read_variant(
    big: np.ndarray,
    binary: np.ndarray,
    *,
    engine: TesseractEngine,
    classifier: PersianDigitClassifier | None,
    whitelist: str,
    variant: str,
    separator: str | None,
    min_confidence: float,
    second_opinion: bool = False,
) -> tuple[list[Glyph], float]:
    """Segment + classify one binarisation of the field."""
    boxes = segment_digit_glyphs(binary)
    glyphs: list[Glyph] = []
    for x0, y0, x1, y1 in boxes:
        patch = big[max(0, y0) : y1, max(0, x0) : x1]
        if patch.size == 0:
            continue
        padded = cv2.copyMakeBorder(patch, 14, 14, 18, 18, cv2.BORDER_CONSTANT, value=255)

        value, confidence = "", 0.0
        options: list[tuple[str, float]] = []
        if separator and is_separator_glyph(patch):
            value, confidence = separator, 90.0
        elif classifier is not None:
            prediction = classifier.classify(patch)
            if prediction is not None and prediction.confidence >= min_confidence:
                value = prediction.digit
                confidence = prediction.confidence * (0.55 + 0.45 * min(1.0, prediction.margin * 4))
                options = prediction.options
        if not value:
            try:
                result = engine.read(padded, psm=10, whitelist=whitelist, variant=variant)
            except Exception:  # pragma: no cover - engine may be unavailable
                break
            value = normalize_persian(result.text.replace("\n", "")).strip()[:2]
            confidence = result.confidence
        elif second_opinion and classifier is not None:
            # the k-NN is confidently wrong on blurred glyphs (۱۷ for ۱۱); an
            # independent reader turns that certainty into an *option* the field
            # validator can then accept or reject.
            try:
                result = engine.read(padded, psm=10, whitelist=whitelist, variant=f"{variant}:second")
            except Exception:  # pragma: no cover - engine may be unavailable
                result = None
            if result is not None:
                other = normalize_persian(result.text.replace("\n", "")).strip()[:1]
                if other and other != value:
                    share = 0.25 + 0.5 * min(1.0, max(0.0, result.confidence) / 100.0)
                    options = _merge_option(options, value, other, share)
        if not value:
            continue
        glyphs.append(Glyph(box=(x0, y0, x1, y1), value=value, confidence=confidence, options=options))
    if not glyphs:
        return [], 0.0
    weights = [max(1, len(glyph.value)) for glyph in glyphs]
    confidence = sum(w * g.confidence for w, g in zip(weights, glyphs, strict=True)) / sum(weights)
    return glyphs, round(confidence, 1)


def read_digit_glyphs(
    engine: TesseractEngine,
    roi_gray: np.ndarray,
    *,
    whitelist: str,
    upscale_factor: float = 3.0,
    max_glyphs: int = 20,
    variant: str = "glyphs",
    classifier: PersianDigitClassifier | None = None,
    min_confidence: float = 45.0,
    separator: str | None = None,
    expected_digits: int | None = None,
    second_opinion: bool = False,
) -> GlyphReading:
    """Read a numeric field glyph by glyph.

    The field is segmented on two independent binarisations (Sauvola and Otsu); the
    reading whose glyph count matches the field's known digit count wins, otherwise
    the most confident reading does.  Per-glyph alternatives (runner-up classes,
    dropped specks) are returned so the field validator can make the final call.
    """
    if roi_gray is None or roi_gray.size == 0:
        return GlyphReading()
    if classifier is None:
        candidate = get_digit_classifier()
        classifier = candidate if candidate.available else None

    big = upscale(roi_gray, max(1.5, upscale_factor))
    big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    window = max(15, (big.shape[1] // 8) | 1)
    binaries = [
        ("sauvola", despeckle(sauvola(big, window=window, k=0.2), min_area_px=8)),
        ("otsu", despeckle(otsu(big), min_area_px=8)),
    ]

    readings: list[tuple[str, list[Glyph], float]] = []
    for name, binary in binaries:
        glyphs, confidence = _read_variant(
            big,
            binary,
            engine=engine,
            classifier=classifier,
            whitelist=whitelist,
            variant=f"{variant}:{name}",
            separator=separator,
            min_confidence=min_confidence,
            second_opinion=second_opinion,
        )
        if glyphs:
            readings.append((name, glyphs, confidence))
    if not readings:
        return GlyphReading()

    chosen = max(readings, key=lambda item: item[2])
    if expected_digits:
        matching = [item for item in readings if len(item[1]) == expected_digits]
        if matching:
            chosen = max(matching, key=lambda item: item[2])
    name, glyphs, confidence = chosen
    glyphs = glyphs[:max_glyphs]
    text = "".join(glyph.value for glyph in glyphs)
    alternatives = (
        _spurious_alternatives(glyphs)
        + _runner_up_alternatives(glyphs)
        + _combination_alternatives(glyphs)
    )
    if separator and expected_digits:
        # a date is 8 digits in 4/2/2 groups; knowing that repairs the extra
        # glyphs that segmentation adds next to a glyph with a long tail
        alternatives = (
            _separator_alternatives(glyphs, separator=separator, limits=_date_limits(expected_digits))
            + _date_group_alternatives(glyphs, separator=separator)
            + alternatives
        )
    return GlyphReading(
        text=text,
        confidence=confidence,
        glyphs=glyphs,
        variant=f"glyphs:{name}",
        alternatives=alternatives,
    )


def _merge_option(
    options: list[tuple[str, float]], primary: str, other: str, share: float
) -> list[tuple[str, float]]:
    """Rebuild the ``(digit, share)`` options with a second reader's opinion."""
    scores: dict[str, float] = {}
    for digit, weight in options:
        scores[digit] = max(scores.get(digit, 0.0), weight)
    scores[primary] = max(scores.get(primary, 0.0), 1.0 - share)
    scores[other] = max(scores.get(other, 0.0), share)
    total = sum(scores.values()) or 1.0
    ordered = sorted(scores.items(), key=lambda item: -item[1])
    return [(digit, round(weight / total, 4)) for digit, weight in ordered]


def _spurious_alternatives(glyphs: list[Glyph], limit: int = 6) -> list[str]:
    """Strings obtained by dropping one glyph.

    Segmentation occasionally keeps a speck next to the number (which adds a digit).
    Instead of deciding here, the *validator* decides: the field finaliser accepts the
    candidate that passes the mod-11 checksum / Jalali calendar.
    """
    values = [glyph.value for glyph in glyphs]
    if len(values) < 3 or limit <= 0:
        return []
    alternatives: list[str] = []
    # the most likely specks sit at the ends of the run
    order = [len(values) - 1, 0, *(range(1, len(values) - 1))]
    for index in order[:limit]:
        alternatives.append("".join(value for position, value in enumerate(values) if position != index))
    return alternatives


def _runner_up_alternatives(glyphs: list[Glyph], limit: int = 8) -> list[str]:
    """Strings where one glyph is replaced by its runner-up class.

    Validators (mod-11 checksum, Jalali calendar) are strong enough to pick the right
    variant, which fixes the classic Persian confusions (۵/۳, ۶/۱, ۴/۲) without
    guessing inside the classifier.
    """
    candidates: list[tuple[float, int, str]] = []
    for index, glyph in enumerate(glyphs):
        if len(glyph.options) < 2:
            continue
        best_share = glyph.options[0][1]
        for runner_up, share in glyph.options[1:]:
            if runner_up == glyph.value:
                continue
            candidates.append((best_share - share, index, runner_up))
            break
    candidates.sort()  # most ambiguous glyphs first
    alternatives: list[str] = []
    for _, index, runner_up in candidates[:limit]:
        values = [glyph.value for glyph in glyphs]
        values[index] = runner_up
        alternatives.append("".join(values))
    return alternatives


def _combination_alternatives(
    glyphs: list[Glyph],
    *,
    max_positions: int = 4,
    min_share: float = 0.12,
    limit: int = 24,
) -> list[str]:
    """Strings where *several* ambiguous glyphs flip at once.

    Single-glyph alternatives cannot repair a field that has two doubtful digits
    (a blurred date is the typical case: ``۱۳۷۸/۱۱/۲۷`` read as ``۱۳۷۸/۱۷/۲۷``).
    The validator is the oracle here, so it is worth handing it the combinations of
    the few most ambiguous glyphs — 2^4 at most, which costs microseconds.
    """
    positions: list[tuple[float, int, str]] = []
    for index, glyph in enumerate(glyphs):
        if len(glyph.options) < 2:
            continue
        best_share = glyph.options[0][1]
        for runner_up, share in glyph.options[1:]:
            if runner_up == glyph.value or share < min_share:
                continue
            positions.append((best_share - share, index, runner_up))
            break
    if not positions:
        return []
    positions.sort()
    positions = positions[:max_positions]
    base = [glyph.value for glyph in glyphs]

    scored: list[tuple[float, str]] = []
    for mask in range(1, 1 << len(positions)):
        cost = 0.0
        values = list(base)
        for bit, (delta, index, runner_up) in enumerate(positions):
            if mask & (1 << bit):
                values[index] = runner_up
                cost -= delta
        scored.append((cost, "".join(values)))
    scored.sort()
    alternatives: list[str] = []
    for _, text in scored[:limit]:
        if text not in alternatives:
            alternatives.append(text)
    return alternatives


def _date_limits(expected_digits: int) -> tuple[int, int, int]:
    """Group sizes of a date field: ``(4, 2, 2)`` for ``YYYYMMDD``, otherwise even thirds."""
    if expected_digits == 8:
        return (4, 2, 2)
    return (expected_digits, 0, 0)


def _separator_alternatives(
    glyphs: list[Glyph],
    *,
    separator: str = "/",
    limits: tuple[int, int, int] = (4, 2, 2),
    pool: int = 6,
    limit: int = 24,
) -> list[str]:
    """Strings where two doubtful glyphs are treated as the *date separators*.

    The ``/`` of a real card is a thin diagonal that the classifier happily calls a
    ``۸`` (it has the same silhouette as the diagonal stroke of that digit).  The
    reading then has two glyphs too many and no group of 4/2/2, so the calendar
    validator can never see a date.  Instead of guessing from geometry, the two least
    confident glyphs of a few candidates are re-labelled as separators and the resulting
    ``YYYY/MM/DD`` strings are handed to the validator, which is the real oracle.
    """
    expected = sum(limits)
    if len(glyphs) != expected + 2 or min(limits) <= 0:
        return []
    values = [glyph.value for glyph in glyphs]
    ranked = sorted(range(len(glyphs)), key=lambda index: glyphs[index].confidence)
    candidates = sorted(ranked[:pool])
    strings: list[str] = []
    for first, second in itertools.combinations(candidates, 2):
        if second - first < 2:
            continue
        parts: list[str] = []
        current: list[str] = []
        for index, value in enumerate(values):
            if index in (first, second):
                parts.append("".join(current))
                current = []
            else:
                current.append(value)
        parts.append("".join(current))
        if [len(part) for part in parts] == list(limits):
            strings.append(separator.join(parts))
    return strings[:limit]


def _date_group_alternatives(
    glyphs: list[Glyph],
    *,
    separator: str = "/",
    limits: tuple[int, int, int] = (4, 2, 2),
    limit: int = 12,
) -> list[str]:
    """Strings that respect the ``YYYY/MM/DD`` structure of a date.

    Segmentation happily splits a ``۵`` into two blobs or keeps a speck, which
    makes the glyph count wrong (``۱۱۵۵۰`` for ``۱۳۵۰``).  The date structure is
    known — 4 digits, separator, 2, separator, 2 — so the excess glyphs are
    dropped in confidence order and every combination is offered to the validator.
    """
    groups: list[list[Glyph]] = [[]]
    separators: list[str] = []
    for glyph in glyphs:
        if glyph.value == separator:
            separators.append(separator)
            groups.append([])
        else:
            groups[-1].append(glyph)
    if len(groups) != 3 or len(separators) != 2:
        return []

    excess = [len(group) - top for group, top in zip(groups, limits, strict=True)]
    if any(value < 0 for value in excess):
        return []  # a missing digit: the ordinary alternatives handle that
    if not any(excess):
        return []

    # per group: which glyph indices may be dropped, most doubtful first
    drop_options: list[list[list[int]]] = []
    for group, extra in zip(groups, excess, strict=True):
        if extra <= 0:
            drop_options.append([[]])
            continue
        ranked = sorted(range(len(group)), key=lambda index: group[index].confidence)
        combinations = []
        for chosen in itertools.combinations(ranked[: extra + 2], extra):
            combinations.append(sorted(chosen))
        drop_options.append(combinations[:6])

    values = [glyph.value for glyph in glyphs]
    offsets = []
    position = 0
    for group in groups:
        offsets.append(position)
        position += len(group) + 1  # +1 for the separator

    scored: list[tuple[float, str]] = []
    for combination in itertools.product(*drop_options):
        dropped: set[int] = set()
        penalty = 0.0
        for group_index, chosen in enumerate(combination):
            for local in chosen:
                dropped.add(offsets[group_index] + local)
                penalty += groups[group_index][local].confidence
        text = "".join(value for index, value in enumerate(values) if index not in dropped)
        scored.append((-penalty, text))
    scored.sort()
    unique: list[str] = []
    for _, text in scored:
        if text not in unique:
            unique.append(text)
    return unique[:limit]
