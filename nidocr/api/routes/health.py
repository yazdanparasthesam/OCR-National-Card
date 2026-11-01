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
    return {
        "version": __version__,
        "lexicon": lexicon_status(),
        "fields": [spec.key for spec in FIELD_SPECS.values()],
    }
