"""Standalone validators (useful for form pre-checks without an image)."""

from __future__ import annotations

from fastapi import APIRouter, Body

from nidocr.fields.validators import validate_birth_date, validate_national_code
from nidocr.schemas import NationalCodeValidation

router = APIRouter(prefix="/validate", tags=["validate"])


@router.post(
    "/national-code",
    response_model=NationalCodeValidation,
    summary="اعتبارسنجی کد ملی / mod-11 national code check",
)
def national_code(value: str = Body(..., embed=True, examples=["0012345678"])) -> NationalCodeValidation:
    outcome = validate_national_code(value)
    return NationalCodeValidation(
        input=value,
        value=outcome.value,
        valid=outcome.validated,
        repaired=outcome.repaired,
        message=outcome.message,
    )


@router.post(
    "/birth-date",
    summary="اعتبارسنجی تاریخ تولد / Jalali date check",
)
def birth_date(value: str = Body(..., embed=True, examples=["1360/01/01"])) -> dict:
    outcome = validate_birth_date(value)
    return {
        "input": value,
        "value": outcome.value,
        "valid": outcome.validated,
        "repaired": outcome.repaired,
        "message": outcome.message,
    }
