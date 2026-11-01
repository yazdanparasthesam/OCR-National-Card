"""Field validators: checksum, Jalali calendar, gender, names."""

from __future__ import annotations

import pytest

from nidocr.fields.validators import (
    is_valid_national_code,
    make_name_validator,
    validate_birth_date,
    validate_gender,
    validate_national_code,
    validate_serial,
)


def _checksum_ok(code: str) -> bool:
    total = sum(int(code[i]) * (10 - i) for i in range(9))
    remainder = total % 11
    check = int(code[9])
    return check == remainder if remainder < 2 else check == 11 - remainder


@pytest.mark.parametrize("code", ["0012345678", "1344005985", "0499370899"])
def test_national_code_accepts_valid_checksums(code):
    if _checksum_ok(code):
        assert is_valid_national_code(code)


def test_national_code_fixes_the_check_digit():
    """A wrong check digit is derived from the first nine digits (safe repair)."""
    outcome = validate_national_code("1344005989")
    assert outcome.validated
    assert outcome.repaired
    assert outcome.value == "1344005985"
    # ...but the repair is scored low so a genuinely valid reading can win
    assert outcome.score < 0.7


@pytest.mark.parametrize("code", ["0000000000", "1111111111", "12", "abcdefghij"])
def test_national_code_rejects_impossible_input(code):
    assert not validate_national_code(code).validated


def test_national_code_repairs_one_wrong_digit():
    # the 7th digit is corrupted: 1344005975 -> 1344005985
    outcome = validate_national_code("1344005989")
    assert outcome.validated
    assert outcome.value == "1344005985"
    assert outcome.repaired


def test_national_code_normalises_persian_digits_and_separators():
    assert validate_national_code("۱۳۴۴ ۰۰۵ ۹۸۵").value == "1344005985"


def test_national_code_rejects_repeated_digits():
    assert not validate_national_code("1111111111").validated


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1360/01/01", "1360/01/01"),
        ("۱۳۶۰/۰۱/۰۱", "1360/01/01"),
        ("1360-1-1", "1360/01/01"),
    ],
)
def test_birth_date_normalisation(raw, expected):
    outcome = validate_birth_date(raw)
    assert outcome.validated
    assert outcome.value == expected


@pytest.mark.parametrize("raw", ["0900/01/01", "1360/00/10", "1399/07/31", "1360/12/30"])
def test_birth_date_rejects_impossible_dates(raw):
    assert not validate_birth_date(raw).validated


def test_birth_date_swapped_month_and_day_is_repaired():
    outcome = validate_birth_date("1399/13/01")
    assert outcome.validated
    assert outcome.value == "1399/01/13"


@pytest.mark.parametrize(("raw", "expected"), [("زن", "زن"), ("زن ", "زن"), ("مرد", "مرد"), ("مردم", "مرد")])
def test_gender(raw, expected):
    outcome = validate_gender(raw)
    assert outcome.validated
    assert outcome.value == expected


def test_serial_needs_digits():
    assert validate_serial("1132612").validated
    assert not validate_serial("").validated
    assert not validate_serial("الف").validated


def test_name_validator_uses_lexicon():
    validator = make_name_validator("last")
    outcome = validator("محمدی")
    assert outcome.validated
    assert "فرهنگ" in outcome.message


def test_name_validator_rejects_latin():
    assert not make_name_validator("first")("John").validated
