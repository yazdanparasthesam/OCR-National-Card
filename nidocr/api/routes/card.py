"""Whole-card endpoints: read every field at once."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException

from nidocr.api.deps import Pipeline, UploadFileDep, decode, read_upload
from nidocr.schemas import CardResult
from nidocr.vision.layout import available_templates

router = APIRouter(prefix="/card", tags=["card"])


@router.post(
    "",
    response_model=CardResult,
    summary="خواندن کامل کارت / read every field of the card",
    response_description="همه فیلدها همراه با اطمینان، اعتبارسنجی و کادر هر فیلد",
)
async def read_card(
    ocr: Pipeline,
    file: UploadFileDep,
    template: str | None = Form(None, description="نام قالب؛ پیش‌فرض national_card_front"),
    debug: bool = Form(False, description="افزودن تصویر پردازش‌شده و کادر فیلدها"),
) -> CardResult:
    """Detect the card, rectify it and read every field.

    The response contains Persian values (``value``), the same values in Persian
    digits (``value_fa``), per-field confidence (0–100), validator status and the
    normalised bounding box of every field.
    """
    image = decode(await read_upload(file))
    return ocr.extract(image, template=template, debug=bool(debug))


@router.post(
    "/batch",
    response_model=list[CardResult],
    summary="چند کارت در یک درخواست / several cards in one request",
)
async def read_cards(
    ocr: Pipeline,
    files: list[UploadFileDep],
    template: str | None = Form(None),
    debug: bool = Form(False),
) -> list[CardResult]:
    """Read several uploaded cards in one request (max 12) and return one result each.
    Each image is decoded and processed independently; failures do not abort the batch.
    """
    if len(files) > 12:
        raise HTTPException(status_code=413, detail="حداکثر ۱۲ تصویر در هر درخواست")
    return [
        ocr.extract(decode(await read_upload(item)), template=template, debug=bool(debug))
        for item in files
    ]


@router.get(
    "/templates",
    summary="قالب‌های موجود / available templates",
)
def card_templates() -> list[str]:
    """List the template names the pipeline knows about (front/back of the card or custom)."""
    return available_templates()
