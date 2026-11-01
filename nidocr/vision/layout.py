"""Card templates and field ROI geometry.

A template is a small JSON document with *normalised* boxes (0..1) for the label
and the value of every field on the canonical, perspective-corrected card::

    {
      "name": "national_card_front",
      "canvas": {"width": 1012, "height": 638},
      "field_layout": {
        "national_code": {"label": [0.40, 0.10, 0.63, 0.15],
                          "value": [0.36, 0.15, 0.63, 0.25], "align": "right"}
      }
    }

Because the card is warped to a fixed canvas before layout, normalised boxes are
scale invariant.  ``refine_rect`` then snaps a box onto the actual ink inside it,
which absorbs the small print/scan shifts that remain after warping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from nidocr.config import Settings, get_settings
from nidocr.errors import TemplateNotFoundError

Rect = tuple[int, int, int, int]


@dataclass
class FieldGeometry:
    key: str
    label_box: tuple[float, float, float, float]
    value_box: tuple[float, float, float, float]
    align: str = "right"


@dataclass
class TemplateLayout:
    name: str
    title_fa: str
    title_en: str
    canvas: tuple[int, int]
    field_layout: dict[str, FieldGeometry]
    raw: dict = field(default_factory=dict)
    path: Path | None = None

    # ── geometry helpers ─────────────────────────────────────────────
    def _to_pixels(self, box, width: int, height: int) -> Rect:
        x0 = int(round(min(box[0], box[2]) * width))
        y0 = int(round(min(box[1], box[3]) * height))
        x1 = int(round(max(box[0], box[2]) * width))
        y1 = int(round(max(box[1], box[3]) * height))
        x0, x1 = max(0, min(x0, width - 2)), max(1, min(x1, width - 1))
        y0, y1 = max(0, min(y0, height - 2)), max(1, min(y1, height - 1))
        return x0, y0, x1, y1

    def value_rect(self, key: str, width: int, height: int) -> Rect | None:
        geometry = self.field_layout.get(key)
        if geometry is None:
            return None
        return self._to_pixels(geometry.value_box, width, height)

    def label_rect(self, key: str, width: int, height: int) -> Rect | None:
        geometry = self.field_layout.get(key)
        if geometry is None:
            return None
        return self._to_pixels(geometry.label_box, width, height)

    def keys(self) -> list[str]:
        return list(self.field_layout)

    def normalised(self, rect: Rect, width: int, height: int) -> dict[str, float]:
        x0, y0, x1, y1 = rect
        return {
            "x": round(x0 / width, 4),
            "y": round(y0 / height, 4),
            "width": round((x1 - x0) / width, 4),
            "height": round((y1 - y0) / height, 4),
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        }


@lru_cache(maxsize=8)
def _load_template(path_str: str, mtime: float) -> TemplateLayout:
    path = Path(path_str)
    data = json.loads(path.read_text(encoding="utf-8"))
    canvas = data.get("canvas", {})
    width = int(canvas.get("width", 1012))
    height = int(canvas.get("height", 638))
    layout: dict[str, FieldGeometry] = {}
    for key, item in data.get("field_layout", {}).items():
        label = item.get("label") or item.get("label_box")
        value = item.get("value") or item.get("value_box")
        if not label or not value:
            continue
        layout[key] = FieldGeometry(
            key=key,
            label_box=tuple(float(v) for v in label),  # type: ignore[arg-type]
            value_box=tuple(float(v) for v in value),  # type: ignore[arg-type]
            align=str(item.get("align", "right")),
        )
    return TemplateLayout(
        name=str(data.get("name", path.stem)),
        title_fa=str(data.get("title_fa", "کارت ملی هوشمند")),
        title_en=str(data.get("title_en", path.stem.replace("_", " ").title())),
        canvas=(width, height),
        field_layout=layout,
        raw=data,
        path=path,
    )


def load_template(name_or_path: str | Path | None = None, settings: Settings | None = None) -> TemplateLayout:
    """Load a template by name (``national_card_front``) or by file path."""
    settings = settings or get_settings()
    path = settings.template_path(str(name_or_path) if name_or_path else None)
    if not Path(path).is_file():
        raise TemplateNotFoundError(
            f"Template {name_or_path!r} not found",
            detail={"searched": str(path), "template_dir": str(settings.template_dir)},
        )
    return _load_template(str(path), Path(path).stat().st_mtime)


def available_templates(settings: Settings | None = None) -> list[str]:
    settings = settings or get_settings()
    directory = Path(settings.template_dir)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


# ── ink aware refinement ─────────────────────────────────────────────────────
def _runs(flags: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs of ``flags``, merging gaps up to ``max_gap``."""
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


def _close_1d(flags: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphological closing of a 1-D boolean profile (bridges small gaps)."""
    if kernel_size <= 1 or not flags.any():
        return flags
    kernel = np.ones((1, int(kernel_size)), np.uint8)
    closed = cv2.morphologyEx(flags.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, kernel)
    return closed.ravel().astype(bool)


def _main_run(profile: np.ndarray, kernel_size: int, threshold_ratio: float = 0.12) -> tuple[int, int] | None:
    """Largest contiguous high-ink run of a projection profile."""
    peak = float(profile.max()) if profile.size else 0.0
    if peak <= 0:
        return None
    flags = profile > max(0.004, threshold_ratio * peak)
    flags = _close_1d(flags, kernel_size)
    runs = [(a, b) for a, b in _runs(flags, max_gap=0) if b - a >= 1]
    if not runs:
        return None
    weights = [float(profile[a : b + 1].sum()) for a, b in runs]
    # keep the run with most ink, then absorb neighbouring runs that are not specks
    best_index = int(np.argmax(weights))
    best_weight = weights[best_index]
    left, right = runs[best_index]
    for index in range(best_index - 1, -1, -1):
        if weights[index] < 0.06 * best_weight:
            break
        left = runs[index][0]
    for index in range(best_index + 1, len(runs)):
        if weights[index] < 0.06 * best_weight:
            break
        right = runs[index][1]
    return left, right


def refine_rect(
    binary: np.ndarray,
    rect: Rect,
    *,
    pad: int = 3,
    row_close_ratio: float = 0.20,
    column_close_ratio: float = 0.030,
    expand_y: float = 0.16,
    allow_columns: bool = True,
) -> Rect:
    """Tighten ``rect`` onto the text it contains, without clipping dots.

    Persian letters carry dots that sit *above/below* the body of the word, so a
    naive projection trim cuts them off (and with them the letter identity: ب ↔ پ ↔ ت).
    The projection is therefore morphologically closed before the run is selected,
    and the resulting band is expanded a little.

    ``allow_columns=False`` keeps the horizontal extent untouched and only trims
    rows.  Cursive Persian words are joined within a word but *not* between words,
    so the column projection happily collapses a two-word name onto a single word
    (or onto one letter) and the field then reads as empty.  Free-flowing text
    fields therefore use rows-only refinement.
    """
    height, width = binary.shape[:2]
    x0, y0, x1, y1 = rect
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return rect
    roi = binary[y0:y1, x0:x1]
    ink = (roi < 128).astype(np.float32)
    if ink.sum() < 8:
        return rect

    box_height, box_width = y1 - y0, x1 - x0
    row_band = _main_run(ink.mean(axis=1), max(1, int(round(row_close_ratio * box_height))))
    if row_band is None:
        return rect
    start, end = row_band
    extra = int(round(expand_y * (end - start + 1)))
    start = max(0, start - extra)
    end = min(box_height - 1, end + extra)

    if allow_columns:
        column_band = _main_run(
            ink[start : end + 1].mean(axis=0), max(2, int(round(column_close_ratio * box_width)))
        )
        if column_band is None:
            return rect
        col_start, col_end = column_band
    else:
        col_start, col_end = 0, box_width - 1

    new = (
        max(0, x0 + col_start - pad),
        max(0, y0 + start - pad),
        min(width, x0 + col_end + 1 + pad),
        min(height, y0 + end + 1 + pad),
    )
    if new[2] - new[0] < 6 or new[3] - new[1] < 6:
        return rect
    return new


def value_rect_from_label(
    label_rect: Rect,
    canvas: tuple[int, int],
    *,
    direction: str = "below",
    gap: float = 0.15,
    height_scale: float = 1.9,
    width_scale: float = 2.6,
) -> Rect:
    """Derive a value ROI from a detected label box (RTL layout aware)."""
    width, height = canvas
    x0, y0, x1, y1 = label_rect
    label_height = max(6, y1 - y0)
    if direction == "below":
        top = int(round(y1 + label_height * gap))
        bottom = int(round(top + label_height * height_scale))
        left = int(round(x0 - (x1 - x0) * 0.15))
        right = int(round(x1 + (x1 - x0) * 0.75))
    else:  # "left": value sits to the left of the label on the same baseline
        top = int(round(y0 - label_height * 0.25))
        bottom = int(round(y1 + label_height * 0.25))
        right = int(round(x0 - label_height * gap))
        left = int(round(right - (x1 - x0) * width_scale))
    left = max(0, left)
    right = min(width, max(left + 8, right))
    top = max(0, top)
    bottom = min(height, max(top + 8, bottom))
    return left, top, right, bottom


def rect_ink(binary: np.ndarray, rect: Rect) -> float:
    height, width = binary.shape[:2]
    x0, y0, x1, y1 = rect
    roi = binary[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)]
    if roi.size == 0:
        return 0.0
    return float(np.mean(roi < 128))


def draw_rect(image: np.ndarray, rect: Rect, color, thickness: int = 2) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
