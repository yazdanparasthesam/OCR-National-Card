"""Machine readable description of every Iranian national card field."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from nidocr.errors import FieldNotFoundError
from nidocr.fields.validators import (
    FieldOutcome,
    make_name_validator,
    validate_birth_date,
    validate_expiry_date,
    validate_gender,
    validate_national_code,
    validate_serial,
)
from nidocr.ocr.text import ARABIC_DIGITS, LATIN_DIGITS, PERSIAN_DIGITS

DIGIT_WHITELIST = PERSIAN_DIGITS + ARABIC_DIGITS + LATIN_DIGITS
DATE_WHITELIST = DIGIT_WHITELIST + "/-"
SERIAL_WHITELIST = DIGIT_WHITELIST + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class FieldSpec:
    """Everything the pipeline needs to locate, read and verify one field."""

    key: str
    label_fa: str
    label_en: str
    kind: str  # digits | date | serial | text | enum
    finalize: Callable[[str], FieldOutcome]
    label_aliases: tuple[str, ...] = ()
    required: bool = True
    psm: int = 7
    whitelist: str | None = None
    description_fa: str = ""
    example: str = ""
    extra_aliases: tuple[str, ...] = field(default_factory=tuple)
    digits_count: int | None = None  # known digit count (numeric fields only)

    @property
    def path(self) -> str:
        """kebab-case segment used by the per-field API routes."""
        return self.key.replace("_", "-")

    def __post_init__(self) -> None:  # pragma: no cover - dataclass convenience
        """Derive the defaults that depend on the other fields of the spec.
        Fills ``path`` (URL slug) and ``digits_count`` from the label/kind when the
        caller did not set them, so a new field can be declared with three arguments.
        """
        if not self.label_aliases:
            object.__setattr__(self, "label_aliases", (self.label_fa,))


FIELD_ORDER: tuple[str, ...] = (
    "national_code",
    "last_name",
    "first_name",
    "father_name",
    "birth_date",
    "expiry_date",
    "serial_number",
    "gender",
)

FIELD_SPECS: dict[str, FieldSpec] = {
    "national_code": FieldSpec(
        key="national_code",
        label_fa="شماره ملی",
        label_en="National code",
        kind="digits",
        finalize=validate_national_code,
        label_aliases=("شماره ملی", "کد ملی", "شماره ملی کارت"),
        whitelist=DIGIT_WHITELIST,
        digits_count=10,
        description_fa="کد ملی ده‌رقمی با رقم کنترل (الگوریتم mod 11)",
        example="0013542419",
    ),
    "last_name": FieldSpec(
        key="last_name",
        label_fa="نام خانوادگی",
        label_en="Last name",
        kind="text",
        finalize=make_name_validator("last"),
        label_aliases=("نام خانوادگی", "نام‌خانوادگی", "فامیل", "نام خانوادگي"),
        description_fa="نام خانوادگی صاحب کارت",
        example="یزدانی",
    ),
    "first_name": FieldSpec(
        key="first_name",
        label_fa="نام",
        label_en="First name",
        kind="text",
        finalize=make_name_validator("first"),
        label_aliases=("نام",),
        description_fa="نام کوچک صاحب کارت",
        example="امیرحسین",
    ),
    "father_name": FieldSpec(
        key="father_name",
        label_fa="نام پدر",
        label_en="Father name",
        kind="text",
        finalize=make_name_validator("first"),
        label_aliases=("نام پدر", "نام پدری"),
        description_fa="نام پدر صاحب کارت",
        example="محمدرضا",
    ),
    "birth_date": FieldSpec(
        key="birth_date",
        label_fa="تاریخ تولد",
        label_en="Birth date",
        kind="date",
        finalize=validate_birth_date,
        label_aliases=("تاریخ تولد", "تاریخ تولد میلادی", "ت ولد"),
        whitelist=DATE_WHITELIST,
        digits_count=8,
        description_fa="تاریخ تولد شمسی به قالب YYYY/MM/DD",
        example="1365/04/12",
    ),
    "expiry_date": FieldSpec(
        key="expiry_date",
        label_fa="پایان اعتبار",
        label_en="Expiry date",
        kind="date",
        finalize=validate_expiry_date,
        label_aliases=("پایان اعتبار", "تاریخ انقضا", "اعتبار", "پایان اعتبار کارت"),
        required=False,
        whitelist=DATE_WHITELIST,
        digits_count=8,
        description_fa="پایان اعتبار کارت هوشمند (شمسی) — تنها در طرح کارت هوشمند",
        example="1406/03/14",
    ),
    "serial_number": FieldSpec(
        key="serial_number",
        label_fa="شماره سریال",
        label_en="Serial number",
        kind="serial",
        finalize=validate_serial,
        label_aliases=("شماره سریال", "سریال", "شماره سریال کارت"),
        whitelist=SERIAL_WHITELIST,
        description_fa="شماره سریال چاپ‌شده روی کارت",
        example="654789123",
    ),
    "gender": FieldSpec(
        key="gender",
        label_fa="جنسیت",
        label_en="Gender",
        kind="enum",
        finalize=validate_gender,
        label_aliases=("جنسیت",),
        required=False,
        description_fa="جنسیت (مرد/زن) — در برخی طرح‌های کارت وجود ندارد",
        example="مرد",
    ),
}

_ALIASES: dict[str, str] = {}
for _key, _spec in FIELD_SPECS.items():
    for _alias in {_key, _spec.path, _key.replace("_", ""), *_spec.label_aliases, *_spec.extra_aliases}:
        _ALIASES[_alias.replace("_", "").replace("-", "").replace(" ", "").lower()] = _key
_ALIASES.update(
    {
        "nationalcode": "national_code",
        "codemeli": "national_code",
        "melicode": "national_code",
        "firstname": "first_name",
        "name": "first_name",
        "lastname": "last_name",
        "surname": "last_name",
        "fathername": "father_name",
        "birthdate": "birth_date",
        "expiry": "expiry_date",
        "expirydate": "expiry_date",
        "validuntil": "expiry_date",
        "dateofbirth": "birth_date",
        "serial": "serial_number",
        "serialnumber": "serial_number",
    }
)


def get_field_spec(key: str) -> FieldSpec:
    """Look up a field by API key, kebab-case, label or common alias."""
    normalised = (key or "").replace("_", "").replace("-", "").replace(" ", "").lower()
    if normalised in _ALIASES:
        return FIELD_SPECS[_ALIASES[normalised]]
    raise FieldNotFoundError(
        f"Unknown field {key!r}",
        detail={"available": [spec.path for spec in FIELD_SPECS.values()]},
    )


def resolve_fields(keys: list[str] | tuple[str, ...] | None) -> list[FieldSpec]:
    """Resolve a list of requested field keys, keeping card order."""
    if not keys:
        return [FIELD_SPECS[key] for key in FIELD_ORDER]
    requested = {get_field_spec(key).key for key in keys}
    return [FIELD_SPECS[key] for key in FIELD_ORDER if key in requested]


def field_paths() -> list[str]:
    """Return ``{url_slug: field_key}`` for the API routes (``national-code`` → ``national_code``)."""
    return [FIELD_SPECS[key].path for key in FIELD_ORDER]
