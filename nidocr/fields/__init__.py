"""Field level validators, specifications and post-processing."""

from __future__ import annotations

from nidocr.fields.definitions import (
    FIELD_ORDER,
    FIELD_SPECS,
    FieldSpec,
    field_paths,
    get_field_spec,
    resolve_fields,
)
from nidocr.fields.validators import (
    FieldOutcome,
    is_valid_jalali_date,
    is_valid_national_code,
    make_national_code,
    national_code_checksum,
    normalize_name,
    normalize_serial,
    validate_birth_date,
    validate_gender,
    validate_name,
    validate_national_code,
    validate_serial,
)

__all__ = [
    "FIELD_ORDER",
    "FIELD_SPECS",
    "FieldOutcome",
    "FieldSpec",
    "field_paths",
    "get_field_spec",
    "is_valid_jalali_date",
    "is_valid_national_code",
    "make_national_code",
    "national_code_checksum",
    "normalize_name",
    "normalize_serial",
    "resolve_fields",
    "validate_birth_date",
    "validate_gender",
    "validate_name",
    "validate_national_code",
    "validate_serial",
]
