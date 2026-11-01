"""Accuracy metrics used by the benchmark and the test-suite."""

from __future__ import annotations

from dataclasses import dataclass

from nidocr.ocr.text import normalize_for_compare


def levenshtein(reference: str, hypothesis: str) -> int:
    """Classic dynamic-programming edit distance."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """CER = edit distance / reference length (0.0 == perfect)."""
    ref = normalize_for_compare(reference) if normalize else reference
    hyp = normalize_for_compare(hypothesis) if normalize else hypothesis
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def character_accuracy(reference: str, hypothesis: str, *, normalize: bool = True) -> float:
    """1 − CER between two strings, computed after Persian-aware normalisation."""
    return max(0.0, 1.0 - character_error_rate(reference, hypothesis, normalize=normalize))


def exact_match(reference: str, hypothesis: str, *, normalize: bool = True) -> bool:
    """True when two values are equal once normalised (digits, ZWNJ, punctuation)."""
    if normalize:
        return normalize_for_compare(reference) == normalize_for_compare(hypothesis)
    return reference == hypothesis


@dataclass
class FieldScore:
    """Per-field comparison of a prediction against ground truth (exact flag + CER)."""
    key: str
    reference: str
    hypothesis: str
    exact: bool
    cer: float

    @property
    def accuracy(self) -> float:
        """Character accuracy of this field (0..1), derived from the stored CER."""
        return 1.0 - self.cer


def summarise(scores: list[FieldScore]) -> dict[str, float]:
    """Aggregate field scores into the headline numbers shown in the docs."""
    if not scores:
        return {"fields": 0, "exact_match": 0.0, "character_accuracy": 0.0}
    return {
        "fields": float(len(scores)),
        "exact_match": sum(score.exact for score in scores) / len(scores),
        "character_accuracy": sum(score.accuracy for score in scores) / len(scores),
    }
