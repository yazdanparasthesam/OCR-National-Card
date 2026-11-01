"""Per-field routes — one endpoint per printed field of the card."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException

from nidocr.api.deps import Pipeline, UploadFileDep, decode, read_upload
from nidocr.errors import FieldNotFoundError
from nidocr.fields.definitions import FIELD_SPECS, get_field_spec
from nidocr.schemas import FieldInfo, FieldOnlyResult

router = APIRouter(prefix="/fields", tags=["fields"])


@router.get(
    "",
    response_model=list[FieldInfo],
    summary="فهرست فیلدها / list of readable fields",
)
def list_fields() -> list[FieldInfo]:
    return [FieldInfo(**item) for item in _describe()]


def _describe() -> list[dict]:
    from nidocr.pipeline import get_ocr

    return get_ocr().describe()


@router.get(
    "/{field}",
    response_model=FieldInfo,
    summary="مشخصات یک فیلد / metadata of one field",
    responses={404: {"description": "field not found"}},
)
def field_info(field: str) -> FieldInfo:
    try:
        spec = get_field_spec(field)
    except FieldNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    return FieldInfo(
        key=spec.key,
        path=spec.path,
        label_fa=spec.label_fa,
        label_en=spec.label_en,
        kind=spec.kind,
        required=spec.required,
        description_fa=spec.description_fa,
        example=spec.example,
    )


@router.post(
    "/{field}",
    response_model=FieldOnlyResult,
    summary="خواندن یک فیلد / read a single field",
    responses={404: {"description": "field not found"}},
)
async def read_field(
    ocr: Pipeline,
    field: str,
    file: UploadFileDep,
    template: str | None = Form(None),
    debug: bool = Form(True, description="افزودن تصویر ناحیه فیلد"),
) -> FieldOnlyResult:
    """Read exactly one field of the card.

    Useful when only one value is needed (e.g. just the national code): it is the
    same pipeline as ``POST /card`` but only the requested region is OCR'ed, so it
    is noticeably faster.
    """
    try:
        get_field_spec(field)
    except FieldNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    image = decode(await read_upload(file))
    return ocr.extract_field(image, field, template=template, debug=bool(debug))
