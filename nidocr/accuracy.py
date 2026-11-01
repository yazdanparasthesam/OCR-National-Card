"""Accuracy measurement shared by the UI, the CLI and ``scripts/benchmark.py``.

The project is judged on *measured* accuracy, so the measurement lives in the
package instead of a notebook or a one-off script:

* :func:`evaluate` scores any list of ``(image, ground truth)`` pairs and returns
  a :class:`AccuracyReport` — per field **and** overall;
* :func:`run_synthetic` renders its own cards (with ground truth) and evaluates
  them, which is what the Streamlit «دقت» tab and ``nidocr bench`` use;
* :func:`compare_card` scores a *single* card against values typed by the user,
  so the UI can show «this card: 6/7 fields, 97 % characters».

Every metric is computed on the normalised values (Persian/Arabic digits, ZWNJ and
punctuation folded), so «۱۳۴۴۰۰۵۹۸۵» and «1344005985» compare equal.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nidocr.fields.definitions import FIELD_ORDER, FIELD_SPECS
from nidocr.metrics import character_accuracy, exact_match

#: signature of the optional progress callback (done, total, label)
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class FieldAccuracy:
    """Accuracy of one field across a dataset.

    ``exact`` and ``characters`` are rates in ``0..1``: a field is *exact* when the
    whole value matches the ground truth, and ``characters`` is the mean 1−CER, so
    a single wrong digit in a 10-digit code still scores 0.9.
    """

    key: str
    label_fa: str
    samples: int = 0
    exact: float = 0.0
    characters: float = 0.0
    confidence: float = 0.0
    validated: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flat dict for tables/JSON (percentages rounded to one decimal)."""
        return {
            "key": self.key,
            "label_fa": self.label_fa,
            "samples": self.samples,
            "exact": round(self.exact * 100, 1),
            "characters": round(self.characters * 100, 1),
            "confidence": round(self.confidence, 1),
            "validated": round(self.validated * 100, 1),
        }


@dataclass
class Failure:
    """One field that did not match, with what was expected and what was read."""

    image: str
    field: str
    expected: str
    predicted: str
    confidence: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flat dict for tables/JSON."""
        return {
            "image": self.image,
            "field": self.field,
            "expected": self.expected,
            "predicted": self.predicted,
            "confidence": round(self.confidence, 1),
        }


@dataclass
class AccuracyReport:
    """The result of one measurement run: overall rates plus the per-field breakdown."""

    cards: int = 0
    fields: list[FieldAccuracy] = field(default_factory=list)
    exact: float = 0.0
    characters: float = 0.0
    perfect_cards: float = 0.0
    mean_latency_ms: float = 0.0
    difficulty: str = "custom"
    failures: list[Failure] = field(default_factory=list)

    def field(self, key: str) -> FieldAccuracy | None:
        """Per-field row by key (``None`` when the field was not measured)."""
        return next((row for row in self.fields if row.key == key), None)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form (percentages, per-field rows and the worst failures)."""
        return {
            "cards": self.cards,
            "difficulty": self.difficulty,
            "exact": round(self.exact * 100, 2),
            "characters": round(self.characters * 100, 2),
            "perfect_cards": round(self.perfect_cards * 100, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "fields": [row.as_row() for row in self.fields],
            "failures": [item.as_row() for item in self.failures],
        }

    def to_markdown(self, *, title: str = "NIDOCR accuracy") -> str:
        """Render the report as the markdown table committed under ``reports/``."""
        lines = [
            f"# {title}",
            "",
            f"- cards: **{self.cards}**",
            f"- difficulty: `{self.difficulty}`",
            f"- field exact match: **{self.exact * 100:.2f}%**",
            f"- field character accuracy: **{self.characters * 100:.2f}%**",
            f"- fully correct cards: **{self.perfect_cards * 100:.2f}%**",
            f"- mean latency: {self.mean_latency_ms:.0f} ms / card",
            "",
            "| field | exact match | char accuracy | mean confidence | validator pass |",
            "|---|---|---|---|---|",
        ]
        for row in self.fields:
            lines.append(
                f"| `{row.key}` | {row.exact * 100:.1f}% | {row.characters * 100:.1f}% "
                f"| {row.confidence:.1f} | {row.validated * 100:.1f}% |"
            )
        if self.failures:
            lines += ["", "### Misreads", "", "| image | field | expected | predicted | conf |", "|---|---|---|---|---|"]
            for item in self.failures[:40]:
                lines.append(
                    f"| {item.image} | `{item.field}` | {item.expected} | {item.predicted or '—'} "
                    f"| {item.confidence:.0f} |"
                )
        return "\n".join(lines) + "\n"


def _new_row(key: str) -> FieldAccuracy:
    """Empty accumulator for one field, pre-filled with its Persian label."""
    spec = FIELD_SPECS.get(key)
    return FieldAccuracy(key=key, label_fa=spec.label_fa if spec else key)


def evaluate(
    records: Iterable[tuple[Any, dict]],
    *,
    ocr,
    difficulty: str = "custom",
    progress: ProgressCallback | None = None,
    keep_failures: int = 60,
) -> AccuracyReport:
    """Score a list of ``(image, ground_truth)`` pairs and return an :class:`AccuracyReport`.

    ``ground_truth`` may be either the full dataset payload (with a ``fields`` key,
    as written by ``scripts/generate_synthetic_cards.py``) or a plain
    ``{field: value}`` mapping.  ``image`` is anything :meth:`NationalCardOCR.extract`
    accepts (path, bytes, numpy array).
    ``progress(done, total, label)`` is called after every card so a UI can show a bar.
    """
    pairs = list(records)
    per_field: dict[str, list[tuple[bool, float, float, bool]]] = defaultdict(list)
    card_rates: list[float] = []
    timings: list[float] = []
    failures: list[Failure] = []

    for index, (image, payload) in enumerate(pairs, start=1):
        expected: dict[str, str] = dict(payload.get("fields", payload) or {})
        template = payload.get("template") if isinstance(payload, dict) else None
        started = time.perf_counter()
        result = ocr.extract(image, template=template)
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)

        hits = 0
        total = 0
        name = _label_of(image)
        for key in FIELD_ORDER:
            if key not in expected:
                continue
            field_result = result.fields.get(key)
            predicted = field_result.value if field_result else ""
            reference = str(expected[key])
            matched = exact_match(reference, predicted) if predicted else False
            per_field[key].append(
                (
                    matched,
                    character_accuracy(reference, predicted) if predicted else 0.0,
                    field_result.confidence if field_result else 0.0,
                    bool(field_result.validated) if field_result else False,
                )
            )
            hits += int(matched)
            total += 1
            if not matched and len(failures) < keep_failures:
                failures.append(
                    Failure(
                        image=name,
                        field=key,
                        expected=reference,
                        predicted=predicted,
                        confidence=field_result.confidence if field_result else 0.0,
                    )
                )
        if total:
            card_rates.append(hits / total)
        if progress is not None:
            progress(index, len(pairs), name)

    rows = [_new_row(key) for key in FIELD_ORDER if key in per_field]
    for row in rows:
        samples = per_field[row.key]
        row.samples = len(samples)
        row.exact = statistics.fmean(item[0] for item in samples)
        row.characters = statistics.fmean(item[1] for item in samples)
        row.confidence = statistics.fmean(item[2] for item in samples)
        row.validated = statistics.fmean(item[3] for item in samples)

    all_samples = [item for samples in per_field.values() for item in samples]
    return AccuracyReport(
        cards=len(card_rates),
        fields=rows,
        exact=statistics.fmean(item[0] for item in all_samples) if all_samples else 0.0,
        characters=statistics.fmean(item[1] for item in all_samples) if all_samples else 0.0,
        perfect_cards=statistics.fmean(card_rates) if card_rates else 0.0,
        mean_latency_ms=statistics.fmean(timings) if timings else 0.0,
        difficulty=difficulty,
        failures=failures,
    )


def _label_of(image: Any) -> str:
    """Best-effort display name of an image (file name, or ``«حافظه»`` for arrays)."""
    if isinstance(image, (str, Path)):
        return Path(image).name
    return "حافظه"


def run_synthetic(
    *,
    count: int = 8,
    difficulty: str = "clean",
    seed: int = 4242,
    ocr=None,
    progress: ProgressCallback | None = None,
) -> AccuracyReport:
    """Render ``count`` synthetic cards, read them and measure accuracy per field.

    This is the honest «accuracy on this machine right now» number: the ground
    truth is known because the identity was generated before the card was drawn.
    Cards are written with :func:`nidocr.synthetic.renderer.save_dataset` and read
    back from disk, so seeds, JPEG quality and degradation are *identical* to what
    ``scripts/benchmark.py --generate`` produces and the two numbers are comparable.
    ``difficulty`` is ``clean`` (flat scan), ``photo`` (phone photo) or ``hard``.
    """
    from nidocr.pipeline import get_ocr
    from nidocr.synthetic.renderer import save_dataset

    ocr = ocr or get_ocr()
    with tempfile.TemporaryDirectory(prefix="nidocr-accuracy-") as tmp:
        folder = Path(tmp)
        save_dataset(folder, count=max(1, int(count)), difficulty=difficulty, seed=int(seed))
        records = load_dataset(folder)
        return evaluate(records, ocr=ocr, difficulty=difficulty, progress=progress)


def compare_card(result, truth: dict[str, str]) -> dict[str, Any]:
    """Score one OCR result against values the user typed in.

    Returns a dict with ``fields`` (key, label, expected, predicted, exact,
    characters) plus the card totals ``exact_rate`` and ``characters`` (0..1),
    which is what the UI shows as «این کارت» accuracy.
    """
    rows: list[dict[str, Any]] = []
    hits = 0
    measured = 0
    for key, expected in truth.items():
        expected = str(expected or "").strip()
        if not expected:
            continue
        field_result = result.fields.get(key)
        predicted = field_result.value if field_result else ""
        matched = exact_match(expected, predicted)
        rows.append(
            {
                "key": key,
                "label_fa": FIELD_SPECS[key].label_fa if key in FIELD_SPECS else key,
                "expected": expected,
                "predicted": predicted,
                "exact": matched,
                "characters": character_accuracy(expected, predicted),
                "confidence": field_result.confidence if field_result else 0.0,
            }
        )
        hits += int(matched)
        measured += 1
    return {
        "fields": rows,
        "measured": measured,
        "correct": hits,
        "exact_rate": (hits / measured) if measured else 0.0,
        "characters": statistics.fmean(row["characters"] for row in rows) if rows else 0.0,
    }


def load_dataset(folder: str | Path) -> list[tuple[Path, dict]]:
    """Load ``image + ground-truth JSON`` pairs from a folder (see the benchmark script)."""
    dataset = Path(folder)
    records: list[tuple[Path, dict]] = []
    for truth_path in sorted(dataset.glob("*.json")):
        payload = json.loads(truth_path.read_text(encoding="utf-8"))
        image_name = payload.get("image") or truth_path.with_suffix(".jpg").name
        image_path = dataset / image_name
        if not image_path.is_file():
            for suffix in (".png", ".jpeg", ".JPG", ".bmp", ".tif"):
                candidate = dataset / Path(image_name).with_suffix(suffix).name
                if candidate.is_file():
                    image_path = candidate
                    break
        if image_path.is_file():
            records.append((image_path, payload))
    return records


def available_difficulties() -> Sequence[str]:
    """Difficulty levels the synthetic generator can produce (used by the UI selectbox)."""
    return ("clean", "photo", "hard")


__all__ = [
    "AccuracyReport",
    "Failure",
    "FieldAccuracy",
    "ProgressCallback",
    "available_difficulties",
    "compare_card",
    "evaluate",
    "load_dataset",
    "run_synthetic",
]


# ── estimated (no ground truth) accuracy ─────────────────────────────────────
#: Measured per-field scores of the shipped model, ``difficulty -> key -> (exact, characters)``.
#: Source: ``reports/benchmark_clean.md`` and ``reports/benchmark_photo.md`` (30 cards each).
#: They are the *priors* of :func:`estimate_field`: without a ground truth the UI can
#: still show a defensible number by starting from the measured rate and adjusting it
#: with what the field itself reports (confidence, validator, repair).
FIELD_PRIORS: dict[str, dict[str, tuple[float, float]]] = {
    "clean": {
        "national_code": (1.000, 1.000),
        "last_name": (0.967, 0.980),
        "first_name": (0.900, 0.942),
        "father_name": (0.900, 0.978),
        "birth_date": (0.800, 0.963),
        "serial_number": (0.933, 0.986),
        "gender": (1.000, 1.000),
    },
    "photo": {
        "national_code": (0.900, 0.927),
        "last_name": (0.900, 0.918),
        "first_name": (0.800, 0.875),
        "father_name": (0.767, 0.827),
        "birth_date": (0.567, 0.863),
        "serial_number": (0.533, 0.739),
        "gender": (0.800, 0.800),
    },
}
#: ``hard`` cards reuse the photo priors but are discounted: they are noisier than a
#: normal phone photo but there is no measured report for them yet.
_HARD_DISCOUNT = 0.85


@dataclass
class FieldEstimate:
    """Estimated accuracy of one field when no ground truth is available.

    ``exact``/``characters`` are rates in ``0..1`` and must be presented as
    *estimates* («تخمینی»), never as measured values.
    """

    key: str
    label_fa: str
    exact: float
    characters: float
    confidence: float
    reason_fa: str = ""

    def as_row(self) -> dict[str, Any]:
        """Flat dict for tables/JSON (percentages rounded to one decimal)."""
        return {
            "key": self.key,
            "label_fa": self.label_fa,
            "exact": round(self.exact * 100, 1),
            "characters": round(self.characters * 100, 1),
            "confidence": round(self.confidence, 1),
            "reason_fa": self.reason_fa,
        }


def _prior(key: str, difficulty: str) -> tuple[float, float]:
    """Measured prior ``(exact, characters)`` of a field for a difficulty level."""
    table = FIELD_PRIORS.get(difficulty if difficulty in FIELD_PRIORS else "photo", {})
    exact, characters = table.get(key, (0.80, 0.90))
    if difficulty == "hard":
        return exact * _HARD_DISCOUNT, characters * _HARD_DISCOUNT
    return exact, characters


def estimate_field(field, *, difficulty: str = "photo") -> FieldEstimate:
    """Estimate one field's accuracy from its measured prior plus its own signals.

    The adjustment is deliberately conservative: a high confidence can only remove
    part of the prior gap, and a field that failed its validator or had to be
    repaired is discounted, so the estimate never claims more than the measured
    baseline.
    """
    key = getattr(field, "key", "")
    label = getattr(field, "label_fa", key) or key
    prior_exact, prior_chars = _prior(key, difficulty)
    confidence = float(getattr(field, "confidence", 0.0) or 0.0)
    found = bool(getattr(field, "found", False))
    validated = bool(getattr(field, "validated", False))
    repaired = bool(getattr(field, "repaired", False))

    if not found:
        return FieldEstimate(key, label, 0.0, 0.0, confidence, "خوانده نشد")

    # 0.9 … 1.0 — a confident reading may recover up to a tenth of the prior gap.
    trust = 0.90 + 0.10 * min(max(confidence, 0.0), 100.0) / 100.0
    exact = prior_exact * trust
    characters = prior_chars + (1.0 - prior_chars) * (trust - 0.90) * 4.0
    reasons = []
    if repaired:
        exact *= 0.80
        characters *= 0.97
        reasons.append("اصلاح‌شده")
    elif not validated:
        exact *= 0.75
        characters *= 0.95
        reasons.append("بدون تأیید اعتبارسنج")
    if confidence < 55:
        exact *= 0.85
        reasons.append("اطمینان پایین")
    if not reasons:
        reasons.append("خواندن مستقیم")
    return FieldEstimate(
        key,
        label,
        min(exact, 0.99),
        min(characters, 0.995),
        confidence,
        " · ".join(reasons),
    )


def estimate_card(fields, *, difficulty: str = "photo") -> dict[str, Any]:
    """Per-field **and** overall estimated accuracy of one card (no ground truth needed).

    Returns ``{"fields": [FieldEstimate…], "exact": 0..1, "characters": 0..1,
    "difficulty": …}``; the overall value is the mean over the card's fields, which
    matches the definition used by :func:`evaluate` for measured runs.
    """
    rows = [estimate_field(field, difficulty=difficulty) for field in fields]
    return {
        "fields": rows,
        "exact": statistics.fmean(row.exact for row in rows) if rows else 0.0,
        "characters": statistics.fmean(row.characters for row in rows) if rows else 0.0,
        "difficulty": difficulty,
    }


__all__ += ["FIELD_PRIORS", "FieldEstimate", "estimate_card", "estimate_field"]
