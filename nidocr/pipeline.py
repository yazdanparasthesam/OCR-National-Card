"""End-to-end OCR pipeline.

``image → card detection → perspective warp → field localisation (label anchors +
template) → multi-variant OCR → Persian normalisation → validation/repair``
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace

import numpy as np

from nidocr.config import Settings, get_settings
from nidocr.errors import NIDOCRError, TemplateNotFoundError
from nidocr.fields.definitions import DIGIT_WHITELIST, FIELD_ORDER, FIELD_SPECS, FieldSpec, resolve_fields
from nidocr.fields.validators import FieldOutcome
from nidocr.image_io import load_image
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
    LINE_TARGET_HEIGHT,
    card_variants,
    crop_with_padding,
    field_variants,
    fit_line,
    normalize_background,
    resize_to_max_side,
    unsharp,
    upscale,
)
from nidocr.vision.layout import (
    Rect,
    TemplateLayout,
    available_templates,
    load_template,
    rect_ink,
    refine_rect,
)
from nidocr.vision.orient import best_rotation, looks_rotated, rotate, upright

#: How much darker one candidate template's photo region must be before ``auto``
#: trusts it: below this the two designs are indistinguishable and the default wins.
GLYPH_UPSCALE = 1.5
"""Magnification the glyph reader applies to an already normalised line."""

GLYPH_TALL_FACTOR = 1.6
"""Second line height for dates, relative to :data:`nidocr.vision.enhance.LINE_TARGET_HEIGHT`.

Measured on 30 degraded photos: 44 px alone reads 18/30 birth dates, 70 px alone 19/30,
and asking both and keeping the more confident reading reads 23/30 — the two heights fail
on different cards, so the tie-break has to be confidence, not a fixed scale."""

TEMPLATE_PHOTO_MARGIN = 0.06

#: A card narrower than this (in source pixels) is supersampled before reading:
#: measured on phone photos, 700 px is where glyph segmentation starts to merge digits.
SMALL_CARD_PX = 700

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
    rotation: int = 0
    osd_confidence: float = 0.0
    scale: float = 1.0
    upscale_boost: float = 1.0

    @property
    def roi_source(self) -> np.ndarray:
        """Whole-card image the field ROIs are cut from.

        Soft photos (a small or blurred source) read better on the *raw* grey canvas:
        the background flattening plus CLAHE that :attr:`enhanced` applies amplifies
        the guilloche pattern of the card into speckle, and Otsu inside
        :func:`field_variants` then works on that noise.  A crisp scan is the other way
        round — there the flattened image removes the printed background and gives
        Tesseract clean strokes.  The switch therefore follows the measured sharpness
        and the supersampling factor instead of being a fixed preference.
        """
        if self.scale > 1.0 or self.quality.sharpness < 120:
            return self.gray
        return self.enhanced

    @property
    def text_source(self) -> np.ndarray:
        """Canvas used for the *line* OCR of the name fields.

        Names go through Tesseract's line segmenter, which is sensitive to the
        background: on a small source photo (supersampled canvas) the raw grey keeps
        the strokes intact, while on a normal scan the flattened image is better.
        Numeric fields use :attr:`roi_source` instead — digits are segmented by our
        own glyph reader, which prefers the other side of that trade-off.
        """
        return self.gray if self.scale > 1.0 else self.enhanced

    @property
    def canvas(self) -> tuple[int, int]:
        """Size of the rectified card canvas (the coordinate system of every box)."""
        height, width = self.color.shape[:2]
        return width, height

    def bbox(self, rect: Rect) -> BBox:
        """Convert a pixel rect of the canvas to the normalised :class:`BBox` of the API."""
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
        """Bind a settings object and the Tesseract engine (both default to the singletons)."""
        self.settings = settings or get_settings()
        self.engine = engine or TesseractEngine(self.settings)
        self._templates: dict[str, TemplateLayout] = {}

    # ── templates ────────────────────────────────────────────────────
    def template(self, name: str | None = None) -> TemplateLayout:
        """Return a template layout by name, cached per process and invalidated on edit."""
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
        requested = str(template or self.settings.default_template)
        detection = detect_card(image, self.settings)
        # both card designs share the canonical canvas, so the warp can happen before
        # the template is known — which "auto" needs, because the design is recognised
        # from the *upright* text and the rotation is only fixed after the warp.
        canvas = (self.settings.canonical_width, self.settings.canonical_height)
        color = warp_card(image, detection.quad, canvas, self.settings)
        scale = _supersample_factor(detection.quad, canvas[0])
        if scale > 1.0:
            # a small photo (a card of ~500 px stretched to 1012 px) arrives soft, and
            # soft Persian digits touch each other, which breaks glyph segmentation.
            # One cubic supersample + a light unsharp is what makes those cards read.
            color = upscale(color, scale)
            color = unsharp(color, amount=0.6, radius=1.1)
        # the warp removes the perspective, not the rotation: a sideways or
        # upside-down card still has rotated text inside the canvas, so ask OSD
        # (one cheap call) and turn the canvas upright first.
        color, rotation, osd_confidence = upright(color, self.engine)
        variants = card_variants(color)
        want_anchors = self.settings.use_label_anchors if use_anchors is None else use_anchors
        anchors: dict[str, tuple[Rect, Anchor]] = {}
        auto = requested.lower() == "auto"
        if want_anchors:
            try:
                anchors = detect_anchors(
                    self.engine,
                    {"enhanced": variants["enhanced"], "binary": variants["binary"]},
                    settings=self.settings,
                    canvas=(variants["gray"].shape[1], variants["gray"].shape[0]),
                    binary=variants["binary"],
                )
            except NIDOCRError:
                anchors = {}
        layout = (
            self.template(self._detect_template(variants)) if auto else self._resolve_layout(requested)
        )
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
            rotation_degrees=rotation,
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
            rotation=rotation,
            osd_confidence=osd_confidence,
            scale=scale,
        )
        if rotation:
            prepared.quality.warnings.append(f"تصویر {rotation} درجه چرخانده شد تا متن افقی شود")

        if want_anchors and anchors:
            prepared.anchors = anchors
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
        result = self._build_result(prepared, specs, debug=debug, started=started)

        # rescue path: OSD found the card upright (or said nothing), yet the reading
        # is garbage — the classic signature of a sideways/upside-down photo. Probe
        # the four orientations and re-read once if another one clearly reads better.
        if prepared.rotation == 0 and prepared.detection.found and looks_rotated(result):
            alternative = self._rescue(prepared, specs, current=result, started=started, debug=debug)
            if alternative is not None:
                result = alternative

        # rescue path 2: a very small photo is read from a supersampled canvas, where the
        # ROI magnification was reduced on purpose.  If that still yields almost nothing,
        # one more read with the full magnification often recovers the digits (measured on
        # a 470-px card photo: national code and surname come out right).
        weak = result.mean_confidence < 50.0 or result.validated_fields <= 2
        if prepared.scale > 1.0 and weak and prepared.detection.found:
            boosted = self._build_result(
                replace(prepared, upscale_boost=prepared.scale**0.5),
                specs,
                debug=debug,
                started=started,
            )
            current_rank = (result.validated_fields, result.total_fields, result.mean_confidence)
            boosted_rank = (boosted.validated_fields, boosted.total_fields, boosted.mean_confidence)
            if boosted_rank > current_rank:
                result = boosted
        return result

    def _build_result(
        self,
        prepared: PreparedCard,
        specs: list[FieldSpec],
        *,
        debug: bool,
        started: float,
    ) -> CardResult:
        """Read ``specs`` from a prepared card and assemble the :class:`CardResult`."""
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
        """Run :meth:`extract` over several images and return one result per image."""
        return [self.extract(source, **kwargs) for source in sources]

    def describe(self) -> list[dict]:
        """Public description of every field (key, URL path, labels, kind, example)."""
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
        """Turn one OCR result into candidate strings for a text field.
        Adds the cleaned line, the line with a leaked label removed, a digits-only
        variant for numeric-looking fields and — because Tesseract occasionally emits a
        word in visual order — the reversed string, leaving the ranking to reject it.
        """
        texts: list[str] = []
        for line in ocr.lines:
            texts.append(line.text_rtl)
            # ``text_ltr`` is the same line in visual order, i.e. a *reversed* string
            # for Persian.  On a blurry photo it frequently out-scores the correct
            # reading, so it is offered only when the lexicon prefers it.
            if (
                line.text_ltr
                and line.text_ltr != line.text_rtl
                and self._lexicon_knows(line.text_ltr, spec)
                and not self._lexicon_knows(line.text_rtl, spec)
            ):
                texts.append(line.text_ltr)
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
            # "قاسمی").  The reversed string is only worth offering when the lexicon
            # actually recognises it *and* not the forward reading — otherwise it is
            # noise that can outrank a correct name on a blurry photo.
            for value in list(collected):
                reversed_value = value[::-1]
                if not reversed_value or reversed_value in collected:
                    continue
                if self._lexicon_knows(reversed_value, spec) and not self._lexicon_knows(value, spec):
                    collected.append(reversed_value)
        return collected[:MAX_CANDIDATES_KEPT]

    def _lexicon_knows(self, value: str, spec: FieldSpec) -> bool:
        """Is ``value`` a known name for this field's kind (never raises)?"""
        if not self.settings.use_name_lexicon:
            return False
        try:
            from nidocr.fields.lexicon import get_lexicon

            kind = "last" if spec.key == "last_name" else "first"
            return get_lexicon(self.settings.lexicon_dir).contains(value, kind)
        except Exception:  # pragma: no cover - the lexicon is optional
            return False

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
        upscale_factor: float | None = None,
    ) -> tuple[float, Candidate, FieldOutcome, Rect, float] | None:
        """Run the glyph reader and rank its string plus every alternative."""
        reading = read_digit_glyphs(
            self.engine,
            roi_gray,
            whitelist=whitelist,
            upscale_factor=upscale_factor or float(self.settings.upscale),
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
        """Digits-only / date-shaped / serial-shaped form of a line, or ``None`` when not applicable."""
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
        """Score a candidate so the best reading wins across ROIs, variants and psm modes.
        Weights: validator outcome first, then OCR confidence, plus bonuses for the
        primary reading and for anchors, minus a penalty for repaired values.
        """
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

    def _reoriented(self, prepared: PreparedCard, angle: int) -> PreparedCard:
        """Same prepared card, rotated ``angle`` degrees clockwise, with fresh variants.

        Everything derived from pixels is rebuilt (gray/enhanced/binary and the label
        anchors); the detection, layout and quality report are kept, and the quality
        report records the rotation so the API and the UI can explain what happened.
        """
        color = rotate(prepared.color, angle)
        variants = card_variants(color)
        quality = prepared.quality.model_copy(update={"rotation_degrees": angle})
        quality.warnings = [item for item in quality.warnings if "چرخانده شد" not in item]
        quality.warnings.append(f"تصویر {angle} درجه چرخانده شد تا متن افقی شود")
        card = replace(
            prepared,
            color=color,
            gray=variants["gray"],
            enhanced=variants["enhanced"],
            binary=variants["binary"],
            quality=quality,
            rotation=angle,
            anchors={},
        )
        if prepared.anchors:
            try:
                card.anchors = detect_anchors(
                    self.engine,
                    {"enhanced": card.enhanced, "binary": card.binary},
                    settings=self.settings,
                    canvas=card.canvas,
                    binary=card.binary,
                )
            except NIDOCRError:
                card.anchors = {}
        return card

    def _rescue(
        self,
        prepared: PreparedCard,
        specs: list[FieldSpec],
        *,
        current: CardResult,
        started: float,
        debug: bool,
    ) -> CardResult | None:
        """Try the best other orientation and keep it only if it reads *better*.

        «Better» is compared on ``(validated fields, found fields, mean confidence)``
        so a rotated guess that merely produces more text cannot win over a reading
        that actually passed the validators.  Returns ``None`` when the probe found
        no convincing alternative, which keeps the extra cost bounded (one probe per
        orientation, one extra read at most).
        """
        angle, _scores = best_rotation(prepared.color, self.engine)
        if not angle:
            return None
        candidate = self._build_result(
            self._reoriented(prepared, angle), specs, debug=debug, started=started
        )
        current_rank = (current.validated_fields, current.total_fields, current.mean_confidence)
        candidate_rank = (candidate.validated_fields, candidate.total_fields, candidate.mean_confidence)
        if candidate_rank <= current_rank:
            return None
        candidate.warnings.append(f"کارت با چرخش {angle} درجه بهتر خوانده شد")
        return candidate

    def _resolve_layout(self, requested: str):
        """Load the requested template, falling back to the default for a bad name.

        A stale client (or a typo in a query parameter) must never turn into a red
        traceback in the UI; the API still reports the unknown name through
        ``GET /templates`` where a caller can act on it.
        """
        try:
            return self.template(requested)
        except TemplateNotFoundError:
            return self.template(self.settings.default_template)

    def _detect_template(self, variants: dict[str, np.ndarray]) -> str:
        """Pick the card design from *where the ID photo sits* — no OCR involved.

        The two designs are mirrored in layout: the smart card prints the photo on
        the left and its six rows on the right, the older card the other way round.
        So each candidate template scores the darkness of its own ``decorations.photo``
        box on the warped canvas, and the darkest photo region wins.  This is
        deterministic, costs no OCR call, and — unlike label matching, which needs a
        good full-page pass — works on blurry photos as well as on rendered cards.
        Falls back to the configured default when the two regions are equally dark.
        """
        default = self.settings.default_template
        gray = variants.get("gray")
        if gray is None:  # pragma: no cover - defensive
            return default
        height, width = gray.shape[:2]
        scores: dict[str, float] = {}
        for name in available_templates(self.settings):
            layout = self.template(name)
            box = (layout.raw.get("decorations") or {}).get("photo")
            if not box:
                continue
            x0, y0, x1, y1 = layout._to_pixels(box, width, height)
            region = gray[y0:y1, x0:x1]
            if region.size == 0:
                continue
            scores[name] = float(255.0 - float(region.mean())) / 255.0
        if not scores:
            return default
        best_name = max(scores, key=scores.get)
        other = max((value for name, value in scores.items() if name != best_name), default=0.0)
        if scores[best_name] - other < TEMPLATE_PHOTO_MARGIN:
            return default
        return best_name

    @staticmethod
    def _glyph_line(roi_gray: np.ndarray, *, height_factor: float = 1.0) -> np.ndarray:
        """Prepare a numeric ROI for the glyph reader: flatten, then normalise the height.

        Both steps matter on a real card. The pattern behind the digits survives a
        binarisation and connects the glyphs into one blob (a ten-digit row then reads
        as three glyphs), and a line taller than ~60 px loses digits at its ends. On a
        flattened, 44-px line the same row yields every digit plus the two separators.
        ``height_factor`` builds the taller variant used as a second opinion.
        """
        target = int(round(LINE_TARGET_HEIGHT * height_factor))
        return fit_line(normalize_background(roi_gray), target_height=target)

    def _vote_text_variants(
        self,
        variants: list[tuple[str, np.ndarray]],
        spec: FieldSpec,
        source_name: str,
        candidates: list[Candidate],
        refined: Rect,
        best: tuple[float, Candidate, FieldOutcome, Rect, float] | None,
    ) -> tuple[float, Candidate, FieldOutcome, Rect, float] | None:
        """Read every preprocessing variant of one ROI and keep the best candidate.

        Tesseract's line segmenter sometimes reports *nothing* — or only a fragment —
        for a perfectly legible RTL crop (a name that fills the box edge to edge is the
        classic case), so the sparse-text modes are tried whenever the first reading is
        not convincing.
        """
        best = self._collect_text_candidates(variants, spec.psm, spec, source_name, candidates, refined, best)
        if best is None or not best[2].value or best[2].score < 0.9:
            for psm in TEXT_FALLBACK_PSM:
                best = self._collect_text_candidates(variants, psm, spec, source_name, candidates, refined, best)
                if best is not None and best[2].score >= 0.99:
                    break
        return best

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
            source = prepared.roi_source if spec.kind in GLYPH_KINDS else prepared.text_source
            roi_gray = crop_with_padding(source, refined, pad_ratio=pad_ratio)
            if roi_gray.size == 0:
                continue

            # 1) numeric fields: glyph-by-glyph reader (far more accurate for digits).
            #    The reader segments digits by column projection, so any guilloche that
            #    survives binarisation connects the glyphs into one blob: flattening the
            #    ROI background first is what makes a real card's date row readable.
            if spec.kind in GLYPH_KINDS and whitelist:
                best = self._read_glyph_candidates(
                    self._glyph_line(roi_gray),
                    spec,
                    whitelist,
                    source_name,
                    refined,
                    candidates,
                    best,
                    upscale_factor=GLYPH_UPSCALE,
                )
                if spec.kind == "date":
                    # two line heights disagree on different cards (blur vs. resolution),
                    # and either one can be calendar-valid while still wrong, so the
                    # taller reading is always asked and the vote decides on confidence
                    best = self._read_glyph_candidates(
                        self._glyph_line(roi_gray, height_factor=GLYPH_TALL_FACTOR),
                        spec,
                        whitelist,
                        source_name,
                        refined,
                        candidates,
                        best,
                        upscale_factor=GLYPH_UPSCALE,
                    )
                if best is not None and best[2].validated and best[2].score >= 0.99:
                    break  # validated by a checksum/calendar rule: no need for more passes

            # 2) whole-line passes with several preprocessing variants. Both line
            # heights are voted: gating the wide one behind a weak narrow reading saved
            # ~0.4 s but cost a photo point, and accuracy wins here.
            variants = field_variants(roi_gray, passes=self.settings.passes)
            best = self._vote_text_variants(variants, spec, source_name, candidates, refined, best)

        if spec.kind in GLYPH_KINDS and whitelist and (best is None or not best[2].validated):
            # the checksum / calendar rejected every reading: ask a second reader
            # for an independent opinion on each glyph before giving up.
            for rect, source_name in options:
                refined = self._refine(prepared.binary, rect, allow_columns=True)
                if rect_ink(prepared.binary, refined) < 0.003:
                    continue
                roi_gray = crop_with_padding(prepared.roi_source, refined, pad_ratio=0.18)
                if roi_gray.size == 0:
                    continue
                before = best[0] if best is not None else None
                best = self._read_glyph_candidates(
                    self._glyph_line(roi_gray),
                    spec,
                    whitelist,
                    source_name,
                    refined,
                    candidates,
                    best,
                    second_opinion=True,
                    upscale_factor=GLYPH_UPSCALE,
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
        """Attach the annotated overlay (field boxes) and the preprocessed card to a result."""
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


def _similarity(left: str, right: str) -> float:
    """Similarity of two short strings, ignoring spaces/ZWNJ (0..1)."""
    from difflib import SequenceMatcher

    def clean(value: str) -> str:
        """Strip spaces, ZWNJ and punctuation so only letters remain."""
        return "".join(ch for ch in (value or "") if ch.isalpha())

    a, b = clean(left), clean(right)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _supersample_factor(quad, canvas_width: int) -> float:
    """How much to enlarge the canvas so the card keeps the resolution it needs.

    Uses the *long* side of the detected quad: a 1012×638 canvas made from a 640-px
    card is fine, one made from a 460-px card is not.  Bounded to 2× so the extra
    cost stays small.
    """
    if quad is None:
        return 1.0
    points = np.asarray(quad, dtype=np.float32)
    if points.shape != (4, 2):  # pragma: no cover - defensive
        return 1.0
    longest = float(max(np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)))
    if longest <= 0:
        return 1.0
    factor = canvas_width / longest
    if longest >= canvas_width:
        return 1.0
    if longest >= SMALL_CARD_PX:
        return 1.0 if factor < 1.3 else 1.5
    return 2.0


def reset_ocr() -> None:
    """Forget the process-wide pipeline singleton (used by tests and after config changes)."""
    global _ocr
    _ocr = None


__all__ = [
    "FIELD_ORDER",
    "NationalCardOCR",
    "PreparedCard",
    "get_ocr",
    "reset_ocr",
]
