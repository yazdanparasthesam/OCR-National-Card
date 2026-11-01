"""End-to-end OCR pipeline.

``image → card detection → perspective warp → field localisation (label anchors +
template) → multi-variant OCR → Persian normalisation → validation/repair``
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

from nidocr.config import Settings, get_settings
from nidocr.errors import NIDOCRError
from nidocr.fields.definitions import FIELD_ORDER, FIELD_SPECS, FieldSpec, resolve_fields
from nidocr.fields.validators import FieldOutcome
from nidocr.image_io import load_image
from nidocr.fields.definitions import DIGIT_WHITELIST
from nidocr.ocr.digit_reader import read_digit_glyphs
from nidocr.ocr.label_anchor import Anchor, detect_anchors
from nidocr.ocr.tesseract_engine import OCRResult, TesseractEngine
from nidocr.ocr.text import (
    clean_ocr_text,
    normalize_for_compare,
    normalize_persian,
    only_digits,
    to_ascii_digits,
)
from nidocr.schemas import (
    BBox,
    Candidate,
    CardResult,
    FieldOnlyResult,
    FieldResult,
    QualityReport,
)
from nidocr.vision.annotate import draw_overlay, encode_jpeg_base64, encode_png_base64, field_color
from nidocr.vision.card_detector import CardDetection, detect_card, estimate_quality, warp_card
from nidocr.vision.enhance import (
    card_variants,
    crop_with_padding,
    field_variants,
    resize_to_max_side,
)
from nidocr.vision.layout import (
    Rect,
    TemplateLayout,
    load_template,
    rect_ink,
    refine_rect,
)

MAX_CANDIDATES_KEPT = 6
GLYPH_KINDS = {"digits", "date", "serial"}
#: Tesseract page-segmentation modes tried when the primary mode finds no text:
#: both are layout-independent and survive crops that the line segmenter rejects.
TEXT_FALLBACK_PSM = (11, 8)


@dataclass
class PreparedCard:
    """Everything the field readers need about one image."""

    original: np.ndarray
    color: np.ndarray
    gray: np.ndarray
    enhanced: np.ndarray
    binary: np.ndarray
    detection: CardDetection
    layout: TemplateLayout
    quality: QualityReport
    anchors: dict[str, tuple[Rect, Anchor]] = field(default_factory=dict)
    prep_ms: float = 0.0

    @property
    def canvas(self) -> tuple[int, int]:
        height, width = self.color.shape[:2]
        return width, height

    def bbox(self, rect: Rect) -> BBox:
        width, height = self.canvas
        x0, y0, x1, y1 = rect
        return BBox(
            x=round(x0 / width, 4),
            y=round(y0 / height, 4),
            width=round((x1 - x0) / width, 4),
            height=round((y1 - y0) / height, 4),
            x0=int(x0),
            y0=int(y0),
            x1=int(x1),
            y1=int(y1),
        )


class NationalCardOCR:
    """Main entry point of the library."""

    def __init__(self, settings: Settings | None = None, engine: TesseractEngine | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or TesseractEngine(self.settings)
        self._templates: dict[str, TemplateLayout] = {}

    # ── templates ────────────────────────────────────────────────────
    def template(self, name: str | None = None) -> TemplateLayout:
        key = name or self.settings.default_template
        if key not in self._templates:
            self._templates[key] = load_template(key, self.settings)
        return self._templates[key]

    # ── preparation ──────────────────────────────────────────────────
    def prepare(
        self,
        source,
        *,
        template: str | None = None,
        use_anchors: bool | None = None,
    ) -> PreparedCard:
        """Load, detect, warp and preprocess one image."""
        started = time.perf_counter()
        image = load_image(source)
        image = resize_to_max_side(image, self.settings.max_image_side)
        layout = self.template(template)

        detection = detect_card(image, self.settings)
        canvas = (layout.canvas[0], layout.canvas[1])
        color = warp_card(image, detection.quad, canvas, self.settings)
        variants = card_variants(color)
        gray_for_ocr = variants["enhanced"]

        gray = variants["gray"]
        quality_dict = estimate_quality(gray, detection)
        quality = QualityReport(
            sharpness=quality_dict["sharpness"],
            brightness=quality_dict["brightness"],
            contrast=quality_dict["contrast"],
            glare_ratio=quality_dict["glare_ratio"],
            resolution=(int(gray.shape[1]), int(gray.shape[0])),
            card_detected=detection.found,
            detection_score=detection.score,
            detection_method=detection.method,
            card_area_ratio=detection.area_ratio,
            warnings=list(quality_dict["warnings"]),
        )

        prepared = PreparedCard(
            original=image,
            color=color,
            gray=gray,
            enhanced=gray_for_ocr,
            binary=variants["binary"],
            detection=detection,
            layout=layout,
            quality=quality,
        )

        want_anchors = self.settings.use_label_anchors if use_anchors is None else use_anchors
        if want_anchors:
            try:
                prepared.anchors = detect_anchors(
                    self.engine,
                    {"enhanced": gray_for_ocr, "binary": variants["binary"]},
                    settings=self.settings,
                    canvas=canvas,
                    binary=variants["binary"],
                )
            except NIDOCRError:
                prepared.anchors = {}
        prepared.prep_ms = round((time.perf_counter() - started) * 1000, 1)
        return prepared

    # ── public API ───────────────────────────────────────────────────
    def extract(
        self,
        source,
        *,
        template: str | None = None,
        fields: list[str] | None = None,
        debug: bool = False,
        use_anchors: bool | None = None,
    ) -> CardResult:
        """Read every requested field of one card image."""
        started = time.perf_counter()
        prepared = self.prepare(source, template=template, use_anchors=use_anchors)
        specs = resolve_fields(fields)

        results: dict[str, FieldResult] = {}
        for spec in specs:
            results[spec.key] = self._read_field(prepared, spec, debug=debug)

        found = [item for item in results.values() if item.found]
        validated = [item for item in found if item.validated]
        mean_confidence = round(sum(item.confidence for item in found) / len(found), 1) if found else 0.0

        warnings = list(prepared.quality.warnings)
        for spec in specs:
            item = results[spec.key]
            if spec.required and not item.found:
                warnings.append(f"«{spec.label_fa}» استخراج نشد")
            elif item.found and not item.validated:
                warnings.append(f"«{spec.label_fa}» نیازمند بازبینی است")

        result = CardResult(
            success=bool(found),
            template=prepared.layout.name,
            template_title_fa=prepared.layout.title_fa,
            engine=self.engine.name,
            languages=self.settings.languages,
            fields=results,
            field_order=[spec.key for spec in specs],
            mean_confidence=mean_confidence,
            validated_fields=len(validated),
            total_fields=len(found),
            processing_ms=round((time.perf_counter() - started) * 1000, 1),
            quality=prepared.quality,
            warnings=warnings,
            image_size=(int(prepared.original.shape[1]), int(prepared.original.shape[0])),
            canonical_size=prepared.canvas,
        )
        if debug:
            self._attach_debug(prepared, result)
        return result

    def extract_field(
        self,
        source,
        field_key: str,
        *,
        template: str | None = None,
        debug: bool = True,
    ) -> FieldOnlyResult:
        """Read a single field (used by ``POST /api/v1/fields/{field}``)."""
        from nidocr.fields.definitions import get_field_spec

        started = time.perf_counter()
        spec = get_field_spec(field_key)
        prepared = self.prepare(source, template=template)
        item = self._read_field(prepared, spec, debug=debug)
        warnings = list(prepared.quality.warnings)
        if item.found and not item.validated:
            warnings.append(f"«{spec.label_fa}» نیازمند بازبینی است")
        return FieldOnlyResult(
            success=item.found,
            field=item,
            processing_ms=round((time.perf_counter() - started) * 1000, 1),
            quality=prepared.quality,
            warnings=warnings,
        )

    def extract_batch(self, sources: list, **kwargs) -> list[CardResult]:
        return [self.extract(source, **kwargs) for source in sources]

    def describe(self) -> list[dict]:
        return [
            {
                "key": spec.key,
                "path": spec.path,
                "label_fa": spec.label_fa,
                "label_en": spec.label_en,
                "kind": spec.kind,
                "required": spec.required,
                "description_fa": spec.description_fa,
                "example": spec.example,
            }
            for spec in FIELD_SPECS.values()
        ]

    # ── internals ────────────────────────────────────────────────────
    @staticmethod
    def _anchor_plausible(anchor_rect: Rect, template_rect: Rect) -> bool:
        """Reject anchors that point at a different row than the template expects.

        Label matching on a whole card is fuzzy (``نام`` is a prefix of
        ``نام خانوادگی``); a wildly displaced anchor would make the ROI read the
        neighbouring field, which is worse than using the calibrated template.
        """
        ax0, ay0, ax1, ay1 = anchor_rect
        tx0, ty0, tx1, ty1 = template_rect
        template_height = max(1, ty1 - ty0)
        anchor_height = max(1, ay1 - ay0)
        vertical_overlap = max(0, min(ay1, ty1) - max(ay0, ty0))
        if vertical_overlap / anchor_height < 0.35:
            return False
        horizontal_overlap = max(0, min(ax1, tx1) - max(ax0, tx0))
        if horizontal_overlap / max(1, ax1 - ax0) < 0.30:
            return False
        centre_shift = abs(((ay0 + ay1) / 2) - ((ty0 + ty1) / 2))
        return centre_shift <= 1.1 * template_height + 0.5 * anchor_height

    @staticmethod
    def _combine_rects(anchor_rect: Rect, template_rect: Rect) -> Rect:
        """Union of the anchor and template box, clamped near the template."""
        ax0, ay0, ax1, ay1 = anchor_rect
        tx0, ty0, tx1, ty1 = template_rect
        margin_x = max(6, int(0.10 * (tx1 - tx0)))
        margin_y = max(6, int(0.60 * (ty1 - ty0)))
        x0 = max(min(ax0, tx0), tx0 - margin_x)
        x1 = min(max(ax1, tx1), tx1 + margin_x)
        y0 = max(min(ay0, ty0), ty0 - margin_y)
        y1 = min(max(ay1, ty1), ty1 + margin_y)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return template_rect
        return x0, y0, x1, y1

    def _refine(self, binary: np.ndarray, rect: Rect, *, allow_columns: bool = True) -> Rect:
        """Snap a template/anchor box onto the ink it contains."""
        if rect_ink(binary, rect) < 0.004:
            return rect
        refined = refine_rect(binary, rect, allow_columns=allow_columns)
        return refined if rect_ink(binary, refined) >= 0.004 else rect

    def _text_candidates(self, ocr: OCRResult, spec: FieldSpec) -> list[str]:
        texts: list[str] = []
        for line in ocr.lines:
            texts.extend([line.text_rtl, line.text_ltr])
        if not texts and ocr.text:
            texts.append(ocr.text)

        collected: list[str] = []
        for raw in texts:
            cleaned = clean_ocr_text(raw)
            if not cleaned:
                continue
            stripped = self._strip_label(cleaned, spec)
            for value in (stripped, cleaned):
                if value and value not in collected:
                    collected.append(value)
            numeric = self._numeric_variant(cleaned, spec)
            if numeric and numeric not in collected:
                collected.append(numeric)
        if spec.kind == "text":
            # Tesseract occasionally emits a word in visual order ("یمساق" for
            # "قاسمی"); the lexicon-aware ranking rejects the reversed string
            # whenever the forward one is a real name.
            for value in list(collected):
                reversed_value = value[::-1]
                if reversed_value and reversed_value not in collected:
                    collected.append(reversed_value)
        return collected[:MAX_CANDIDATES_KEPT]

    def _read_glyph_candidates(
        self,
        roi_gray: np.ndarray,
        spec: FieldSpec,
        whitelist: str,
        source_name: str,
        refined: Rect,
        candidates: list[Candidate],
        best: tuple[float, Candidate, FieldOutcome, Rect, float] | None,
        *,
        second_opinion: bool = False,
    ) -> tuple[float, Candidate, FieldOutcome, Rect, float] | None:
        """Run the glyph reader and rank its string plus every alternative."""
        reading = read_digit_glyphs(
            self.engine,
            roi_gray,
            whitelist=whitelist,
            upscale_factor=self.settings.upscale,
            separator="/" if spec.kind == "date" else None,
            expected_digits=spec.digits_count,
            second_opinion=second_opinion,
        )
        for alt_index, glyph_text in enumerate([reading.text, *reading.alternatives]):
            if not glyph_text:
                continue
            outcome = spec.finalize(glyph_text)
            bonus = 0.35 if alt_index == 0 else 0.05
            score = self._rank(outcome, reading.confidence, spec) + bonus
            candidate = Candidate(
                variant=reading.variant,
                source=source_name,
                text=outcome.value or glyph_text,
                confidence=reading.confidence,
                validated=outcome.validated,
                score=round(score, 4),
            )
            candidates.append(candidate)
            if best is None or score > best[0]:
                best = (score, candidate, outcome, refined, reading.confidence)
        return best

    def _collect_text_candidates(
        self,
        variants: list[tuple[str, np.ndarray]],
        psm: int,
        spec: FieldSpec,
        source_name: str,
        candidates: list[Candidate],
        refined: Rect,
        best: tuple[float, Candidate, FieldOutcome, Rect, float] | None,
    ) -> tuple[float, Candidate, FieldOutcome, Rect, float] | None:
        """OCR every preprocessing variant at ``psm`` and keep the best reading."""
        for variant_name, variant_image in variants:
            ocr = self.engine.read(
                variant_image,
                psm=psm,
                whitelist=spec.whitelist,
                variant=variant_name,
            )
            if ocr.empty:
                continue
            for text in self._text_candidates(ocr, spec):
                outcome = spec.finalize(text)
                score = self._rank(outcome, ocr.confidence, spec)
                candidate = Candidate(
                    variant=variant_name,
                    source=source_name,
                    text=outcome.value or clean_ocr_text(text),
                    confidence=round(ocr.confidence, 1),
                    validated=outcome.validated,
                    score=score,
                )
                candidates.append(candidate)
                if best is None or score > best[0]:
                    best = (score, candidate, outcome, refined, ocr.confidence)
        return best

    @staticmethod
    def _strip_label(text: str, spec: FieldSpec) -> str:
        """Remove a label that leaked into the ROI (``شماره ملی ۰۰۱۲۳۴۵۶۷۸``)."""
        words = text.split()
        if not words:
            return text
        for alias in spec.label_aliases:
            alias_words = alias.split()
            size = len(alias_words)
            if len(words) <= size:
                continue
            head = " ".join(words[:size])
            tail = " ".join(words[-size:])
            if normalize_for_compare(head) == normalize_for_compare(alias):
                return " ".join(words[size:]).strip(" :،؛-")
            if normalize_for_compare(tail) == normalize_for_compare(alias):
                return " ".join(words[:-size]).strip(" :،؛-")
        return text

    @staticmethod
    def _numeric_variant(text: str, spec: FieldSpec) -> str | None:
        if spec.kind == "digits":
            return only_digits(text) or None
        if spec.kind == "date":
            cleaned = re.sub(r"[^0-9/\-]", "", to_ascii_digits(normalize_persian(text)))
            cleaned = re.sub(r"[/\-]{2,}", "/", cleaned).strip("/-")
            return cleaned or None
        if spec.kind == "serial":
            cleaned = re.sub(r"[^0-9A-Za-z]", "", to_ascii_digits(normalize_persian(text))).upper()
            return cleaned or None
        return None

    @staticmethod
    def _rank(outcome: FieldOutcome, confidence: float, spec: FieldSpec | None = None) -> float:
        score = 2.0 * outcome.score + min(max(confidence, 0.0), 100.0) / 100.0
        if spec is not None and spec.kind == "serial" and len(outcome.value) > 12:
            score -= 0.4  # implausibly long serial: usually merged glyphs
        if outcome.validated:
            score += 0.5
        if outcome.repaired:
            score -= 0.1
        if not outcome.value:
            score -= 1.5
        return round(score, 4)

    def _read_field(self, prepared: PreparedCard, spec: FieldSpec, *, debug: bool = False) -> FieldResult:
        """OCR one field: label anchor first, then the template box, then vote."""
        width, height = prepared.canvas
        options: list[tuple[Rect, str]] = []
        anchor = prepared.anchors.get(spec.key)
        template_rect = prepared.layout.value_rect(spec.key, width, height)
        if anchor is not None and template_rect is not None and not self._anchor_plausible(anchor[0], template_rect):
            anchor = None
        if anchor is not None:
            anchor_rect = anchor[0]
            if template_rect is not None:
                # the anchor follows the *printed* position (vertical shifts), the
                # template guarantees the value is not clipped horizontally
                anchor_rect = self._combine_rects(anchor_rect, template_rect)
            options.append((anchor_rect, "anchor"))
        if template_rect is not None:
            options.append((template_rect, "template"))
        if not options:
            return FieldResult.empty(spec, message="مکان این فیلد در قالب تعریف نشده است")

        candidates: list[Candidate] = []
        best: tuple[float, Candidate, FieldOutcome, Rect, float] | None = None
        whitelist = spec.whitelist or (DIGIT_WHITELIST if spec.kind in GLYPH_KINDS else None)

        # free text (names) keeps its full width: column trimming can collapse a
        # two-word Persian name onto a single letter and the field reads empty
        allow_columns = spec.kind in GLYPH_KINDS
        for rect, source_name in options:
            refined = self._refine(prepared.binary, rect, allow_columns=allow_columns)
            if rect_ink(prepared.binary, refined) < 0.003:
                continue
            # numeric fields get generous padding so edge glyphs are never clipped
            pad_ratio = 0.18 if spec.kind in GLYPH_KINDS else 0.11
            roi_gray = crop_with_padding(prepared.enhanced, refined, pad_ratio=pad_ratio)
            if roi_gray.size == 0:
                continue

            # 1) numeric fields: glyph-by-glyph reader (far more accurate for digits)
            if spec.kind in GLYPH_KINDS and whitelist:
                best = self._read_glyph_candidates(
                    roi_gray, spec, whitelist, source_name, refined, candidates, best
                )
                if best is not None and best[2].validated and best[2].score >= 0.99:
                    break  # validated by a checksum/calendar rule: no need for more passes

            # 2) whole-line passes with several preprocessing variants
            variants = field_variants(
                roi_gray,
                upscale_factor=self.settings.upscale,
                passes=self.settings.passes,
            )
            best = self._collect_text_candidates(
                variants, spec.psm, spec, source_name, candidates, refined, best
            )
            if best is None or not best[2].value or best[2].score < 0.9:
                # Tesseract's line segmenter sometimes reports *nothing* — or only a
                # fragment — for a perfectly legible RTL crop (a name that fills the
                # box edge to edge is the classic case).  The sparse-text modes see
                # the whole word, so retry whenever the reading is not convincing.
                for psm in TEXT_FALLBACK_PSM:
                    best = self._collect_text_candidates(
                        variants, psm, spec, source_name, candidates, refined, best
                    )
                    if best is not None and best[2].score >= 0.99:
                        break

        if spec.kind in GLYPH_KINDS and whitelist and (best is None or not best[2].validated):
            # the checksum / calendar rejected every reading: ask a second reader
            # for an independent opinion on each glyph before giving up.
            for rect, source_name in options:
                refined = self._refine(prepared.binary, rect, allow_columns=True)
                if rect_ink(prepared.binary, refined) < 0.003:
                    continue
                roi_gray = crop_with_padding(prepared.enhanced, refined, pad_ratio=0.18)
                if roi_gray.size == 0:
                    continue
                before = best[0] if best is not None else None
                best = self._read_glyph_candidates(
                    roi_gray,
                    spec,
                    whitelist,
                    source_name,
                    refined,
                    candidates,
                    best,
                    second_opinion=True,
                )
                if before is not None and best is not None and best[0] == before:
                    continue
                if best is not None and best[2].validated and best[2].score >= 0.99:
                    break

        if best is None:
            return FieldResult.empty(spec, message="متنی در این ناحیه پیدا نشد")

        _, candidate, outcome, refined, confidence = best
        reported_confidence = confidence * (0.85 if outcome.repaired else 1.0)
        result = FieldResult(
            key=spec.key,
            label_fa=spec.label_fa,
            label_en=spec.label_en,
            kind=spec.kind,
            value=outcome.value,
            raw_value=candidate.text,
            confidence=round(min(100.0, reported_confidence), 1),
            validated=outcome.validated,
            repaired=outcome.repaired,
            found=bool(outcome.value),
            required=spec.required,
            message=outcome.message,
            bbox=prepared.bbox(refined),
            source=candidate.source,
        )
        if debug:
            roi = crop_with_padding(prepared.color, refined, pad_ratio=0.05)
            if roi.size:
                result.roi_png = encode_png_base64(roi)
            candidates.sort(key=lambda item: item.score, reverse=True)
            result.candidates = candidates[:MAX_CANDIDATES_KEPT]
        return result.with_persian_digits()

    # ── debug artefacts ──────────────────────────────────────────────
    def _attach_debug(self, prepared: PreparedCard, result: CardResult) -> None:
        boxes: dict[str, Rect] = {}
        colors: dict[str, tuple[int, int, int]] = {}
        labels: dict[str, str] = {}
        for key, item in result.fields.items():
            if item.bbox is None:
                continue
            boxes[key] = (item.bbox.x0, item.bbox.y0, item.bbox.x1, item.bbox.y1)
            colors[key] = field_color(item.validated, item.confidence, item.repaired)
            labels[key] = f"{item.label_en}"
        if boxes:
            overlay = draw_overlay(prepared.color, boxes, labels=labels, colors=colors)
            result.overlay_png = encode_jpeg_base64(overlay, 86)
        result.processed_png = encode_jpeg_base64(
            np.dstack([prepared.enhanced] * 3) if prepared.enhanced.ndim == 2 else prepared.enhanced, 82
        )


_ocr: NationalCardOCR | None = None


def get_ocr(settings: Settings | None = None) -> NationalCardOCR:
    """Process-wide pipeline singleton (used by the API)."""
    global _ocr
    if _ocr is None or settings is not None:
        _ocr = NationalCardOCR(settings)
    return _ocr


def reset_ocr() -> None:
    global _ocr
    _ocr = None


__all__ = [
    "FIELD_ORDER",
    "NationalCardOCR",
    "PreparedCard",
    "get_ocr",
    "reset_ocr",
]
