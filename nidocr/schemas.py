"""Pydantic models shared by the API, the UI and the CLI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nidocr.ocr.text import to_persian_digits


class BBox(BaseModel):
    """Field box in both normalised (0..1) and pixel coordinates."""

    x: float = Field(..., description="left, normalised 0..1")
    y: float = Field(..., description="top, normalised 0..1")
    width: float
    height: float
    x0: int
    y0: int
    x1: int
    y1: int
    space: Literal["card", "image"] = "card"


class Candidate(BaseModel):
    """One OCR alternative that was considered for a field (debug info)."""

    variant: str
    source: str
    text: str
    confidence: float
    validated: bool
    score: float


class FieldResult(BaseModel):
    """One field of the card as returned by the API/UI: value (Latin digits), the same
    value in Persian digits, confidence, validator flags, Persian message, bounding
    box, provenance and — in debug mode — the ROI crop and every candidate.
    """
    model_config = ConfigDict(populate_by_name=True)

    key: str
    label_fa: str
    label_en: str
    kind: str
    value: str = ""
    value_fa: str = ""
    raw_value: str = ""
    confidence: float = 0.0
    validated: bool = False
    repaired: bool = False
    found: bool = False
    required: bool = True
    message: str = ""
    bbox: BBox | None = None
    source: str = "template"
    roi_png: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)

    @classmethod
    def empty(cls, spec, *, message: str = "استخراج نشد") -> FieldResult:
        """Build an «not found» result for a field with a Persian reason."""
        return cls(
            key=spec.key,
            label_fa=spec.label_fa,
            label_en=spec.label_en,
            kind=spec.kind,
            required=spec.required,
            message=message,
        )

    def with_persian_digits(self) -> FieldResult:
        """Fill ``value_fa`` from ``value`` (dates/serials keep their separators)."""
        value_fa = to_persian_digits(self.value) if self.kind in {"digits", "date", "serial"} else self.value
        return self.model_copy(update={"value_fa": value_fa or self.value})


class QualityReport(BaseModel):
    """Image-quality facts about the processed photo (sharpness, exposure, glare,
    resolution, detection score) plus the Persian warnings derived from them.
    """
    sharpness: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    glare_ratio: float = 0.0
    resolution: tuple[int, int] = (0, 0)
    card_detected: bool = False
    detection_score: float = 0.0
    detection_method: str = "unknown"
    card_area_ratio: float = 0.0
    rotation_degrees: int = 0
    warnings: list[str] = Field(default_factory=list)


class CardResult(BaseModel):
    """The complete answer for one card: all fields in reading order, aggregate
    confidence/validation counters, timing, quality report and debug images.
    """
    success: bool = True
    template: str = "national_card_front"
    template_title_fa: str = "کارت ملی هوشمند"
    engine: str = "tesseract"
    languages: str = "fas"
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    field_order: list[str] = Field(default_factory=list)
    mean_confidence: float = 0.0
    validated_fields: int = 0
    total_fields: int = 0
    processing_ms: float = 0.0
    quality: QualityReport = Field(default_factory=QualityReport)
    warnings: list[str] = Field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)
    canonical_size: tuple[int, int] = (0, 0)
    overlay_png: str | None = None
    processed_png: str | None = None

    # ── convenience ──────────────────────────────────────────────────
    def ordered_fields(self) -> list[FieldResult]:
        """Fields in the order they are printed on the card (national code first)."""
        order = self.field_order or list(self.fields)
        return [self.fields[key] for key in order if key in self.fields]

    def flat(self) -> dict[str, str]:
        """``{field: value}`` mapping — handy for quick integrations."""
        return {key: field.value for key, field in self.fields.items()}

    def get(self, key: str, default: str = "") -> str:
        """Value of one field (``""`` when the field is missing) — the quick accessor for callers."""
        field = self.fields.get(key)
        return field.value if field else default

    def summary(self) -> dict[str, Any]:
        """Compact dict with the counters and warnings (used by logs and the UI header)."""
        return {
            "success": self.success,
            "template": self.template,
            "mean_confidence": self.mean_confidence,
            "validated_fields": self.validated_fields,
            "total_fields": self.total_fields,
            "processing_ms": self.processing_ms,
            "warnings": self.warnings,
        }


class FieldOnlyResult(BaseModel):
    """Response of ``POST /fields/{field}`` — a single field, same shape as in a card."""

    success: bool = True
    field: FieldResult
    processing_ms: float = 0.0
    quality: QualityReport = Field(default_factory=QualityReport)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Answer of ``GET /health``: engine readiness, models, templates and fields."""
    status: Literal["ok", "degraded"] = "ok"
    version: str
    engine: str
    engine_ready: bool
    engine_message: str
    languages: str
    available_languages: list[str] = Field(default_factory=list)
    tessdata_dir: str | None = None
    templates: list[str] = Field(default_factory=list)
    fields: list[str] = Field(default_factory=list)


class FieldInfo(BaseModel):
    """Metadata of a field (``GET /fields``): key, URL path, labels, kind, example."""
    key: str
    path: str
    label_fa: str
    label_en: str
    kind: str
    required: bool
    description_fa: str
    example: str


class TemplateInfo(BaseModel):
    """Metadata of a template: name, titles, canvas and the fields it defines."""
    name: str
    title_fa: str
    title_en: str
    canvas: tuple[int, int]
    fields: list[str]


class NationalCodeValidation(BaseModel):
    """Answer of ``POST /validate/national-code``: normalised value, validity and repair flag."""
    input: str
    value: str
    valid: bool
    repaired: bool
    message: str


class ErrorResponse(BaseModel):
    """Stable error envelope ``{detail, code, extra}`` returned by the API for every failure."""
    detail: str
    code: str = "error"
    extra: dict[str, Any] = Field(default_factory=dict)
