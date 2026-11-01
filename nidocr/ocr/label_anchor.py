"""Label anchored layout detection.

Instead of relying only on fixed template boxes, the card is OCR'd once as a page;
the printed labels (``شماره ملی``, ``نام``, ``نام پدر`` …) are then located with a
fuzzy Persian matcher and each value ROI is derived from the label that owns it.
This makes the pipeline tolerant to card revisions, print shifts and small warps.
"""

from __future__ import annotations

from dataclasses import dataclass

from nidocr.config import Settings
from nidocr.fields.definitions import FIELD_SPECS, FieldSpec
from nidocr.ocr.tesseract_engine import OCRLine, TesseractEngine, TextToken
from nidocr.ocr.text import normalize_for_compare, similarity
from nidocr.vision.layout import Rect, rect_ink, refine_rect, value_rect_from_label


@dataclass
class Anchor:
    """A printed field label found on the card: which field it belongs to, the matched
    text, its pixel box, the similarity and the OCR confidence that produced it.
    """
    field_key: str
    label_text: str
    rect: Rect
    similarity: float
    confidence: float
    source_variant: str

    @property
    def weight(self) -> float:
        """Ranking weight of an anchor: similarity first, OCR confidence as a tie-breaker."""
        return self.similarity * (0.4 + 0.006 * max(0.0, self.confidence))


def _span_text(tokens: list[TextToken]) -> str:
    """Join the tokens of a label span into one normalised string."""
    return normalize_for_compare(" ".join(token.text for token in tokens))


def _matches(span: str, alias: str, threshold: float) -> float:
    """Similarity between an OCR token span and a label alias (0 when they cannot match)."""
    target = normalize_for_compare(alias)
    if not target or not span:
        return 0.0
    if span == target:
        return 1.0
    # guard against "نام" matching inside "نام خانوادگی"
    length_ratio = len(span) / len(target)
    if length_ratio < 0.68 or length_ratio > 1.45:
        return 0.0
    score = similarity(span, target)
    return score if score >= threshold else 0.0


def match_labels_in_line(
    line: OCRLine,
    specs: list[FieldSpec],
    *,
    threshold: float = 0.66,
    max_span: int = 3,
) -> list[Anchor]:
    """Find label matches inside one OCR line."""
    tokens = sorted(line.tokens, key=lambda token: token.left)  # RTL: right-most first
    anchors: list[Anchor] = []
    for spec in specs:
        best: Anchor | None = None
        for alias in spec.label_aliases:
            for span_size in range(1, max_span + 1):
                for start in range(0, len(tokens) - span_size + 1):
                    span_tokens = tokens[start : start + span_size]
                    span = _span_text(span_tokens)
                    score = _matches(span, alias, threshold)
                    if score <= 0:
                        continue
                    left = min(token.left for token in span_tokens)
                    top = min(token.top for token in span_tokens)
                    right = max(token.left + token.width for token in span_tokens)
                    bottom = max(token.top + token.height for token in span_tokens)
                    confidence = sum(token.confidence for token in span_tokens) / len(span_tokens)
                    anchor = Anchor(
                        field_key=spec.key,
                        label_text=alias,
                        rect=(left, top, right, bottom),
                        similarity=score,
                        confidence=confidence,
                        source_variant=line.text_rtl,
                    )
                    if best is None or anchor.weight > best.weight:
                        best = anchor
        if best is not None:
            anchors.append(best)
    return anchors


def detect_anchors(
    engine: TesseractEngine,
    images: dict[str, object],
    *,
    settings: Settings,
    canvas: tuple[int, int],
    binary,
) -> dict[str, tuple[Rect, Anchor]]:
    """OCR each provided card variant and derive value ROIs for every field."""
    specs = [FIELD_SPECS[key] for key in FIELD_SPECS]
    best: dict[str, tuple[Rect, Anchor]] = {}
    for variant, image in images.items():
        try:
            result = engine.read_page(image, psm=6, variant=variant)
        except Exception:  # pragma: no cover - OCR failure must never break the pipeline
            continue
        for line in result.lines:
            for anchor in match_labels_in_line(line, specs, threshold=settings.anchor_min_similarity):
                for direction in ("below", "left"):
                    candidate = value_rect_from_label(anchor.rect, canvas, direction=direction)
                    refined = refine_rect(binary, candidate)
                    ink = rect_ink(binary, refined)
                    if ink < 0.008:
                        continue
                    current = best.get(anchor.field_key)
                    if current is None or anchor.weight > current[1].weight:
                        best[anchor.field_key] = (refined, anchor)
                    break
    return best
