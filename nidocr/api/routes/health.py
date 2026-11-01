"""Health / readiness / capability discovery."""

from __future__ import annotations

from fastapi import APIRouter

from nidocr import __version__
from nidocr.fields.definitions import FIELD_SPECS
from nidocr.fields.lexicon import lexicon_status
from nidocr.ocr.tesseract_engine import get_engine
from nidocr.schemas import HealthResponse
from nidocr.vision.layout import available_templates

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="سرویس سالم است؟ / service health",
)
def health() -> HealthResponse:
    """Report readiness: engine name/binary, tessdata folder, languages, templates, fields.
    Returns ``status='degraded'`` (HTTP 200) when Tesseract cannot run, so a load
    balancer can keep the pod alive while the operator reads ``engine_message``.
    """
    engine = get_engine()
    ready, message = engine.ready()
    return HealthResponse(
        status="ok" if ready else "degraded",
        version=__version__,
        engine=engine.name,
        engine_ready=ready,
        engine_message=message,
        languages=engine.languages,
        available_languages=engine.available_languages(),
        tessdata_dir=str(engine.tessdata) if engine.tessdata else None,
        templates=available_templates(),
        fields=list(FIELD_SPECS),
    )


@router.get(
    "/capabilities",
    summary="نام‌های فرهنگ واژگان و امکانات فعال / lexicon + feature flags",
)
def capabilities() -> dict:
    """Report optional capabilities: lexicon sizes and the list of readable fields."""
    return {
        "version": __version__,
        "lexicon": lexicon_status(),
        "fields": [spec.key for spec in FIELD_SPECS.values()],
    }
