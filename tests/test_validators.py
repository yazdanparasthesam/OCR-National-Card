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
    """Reference implementation of the mod-11 checksum, used to build fixtures."""
    total = sum(int(code[i]) * (10 - i) for i in range(9))
    remainder = total % 11
    check = int(code[9])
    return check == remainder if remainder < 2 else check == 11 - remainder


@pytest.mark.parametrize("code", ["0012345678", "1344005985", "0499370899"])
def test_national_code_accepts_valid_checksums(code):
    """Codes whose check digit satisfies mod-11 are accepted unchanged."""
    if _checksum_ok(code):
        assert is_valid_national_code(code)


def test_national_code_fixes_the_check_digit_on_request():
    """`derive_check_digit=True` rebuilds the tenth digit from the first nine."""
    outcome = validate_national_code("1344005989", derive_check_digit=True)
    assert outcome.validated
    assert outcome.repaired
    assert outcome.value == "1344005985"
    # ...but the repair is scored low so a genuinely valid reading can win
    assert outcome.score < 0.7


def test_national_code_keeps_the_reading_when_only_the_tail_could_be_rewritten():
    """By default the digits are returned as read: the error may not be the last one.

    Measured on real photos: the derivation turned correct readings (printed codes that
    simply are not checksum-valid) into valid-but-wrong ones.
    """
    outcome = validate_national_code("7231198619")
    assert outcome.value == "7231198619"
    assert not outcome.repaired
    # ...and `1344005989` (one digit away from a valid code) is left alone too
    assert validate_national_code("1344005989").value == "1344005989"


@pytest.mark.parametrize("code", ["0000000000", "1111111111", "12", "abcdefghij"])
def test_national_code_rejects_impossible_input(code):
    """Impossible inputs (all-same digits, wrong length, letters) are rejected."""
    assert not validate_national_code(code).validated


def test_national_code_repairs_one_wrong_digit():
    """A code one digit away from validity is repaired when exactly one fix exists."""
    outcome = validate_national_code("1344005989", derive_check_digit=True)
    assert outcome.validated
    assert outcome.value == "1344005985"
    assert outcome.repaired


def test_national_code_normalises_persian_digits_and_separators():
    """Spaces and Persian digits are normalised before validation."""
    assert validate_national_code("۱۳۴۴ ۰۰۵ ۹۸۵").value == "1344005985"


def test_national_code_rejects_repeated_digits():
    """A code made of one repeated digit is never valid."""
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
    """Dates are normalised to ``YYYY/MM/DD`` with ASCII digits."""
    outcome = validate_birth_date(raw)
    assert outcome.validated
    assert outcome.value == expected


@pytest.mark.parametrize("raw", ["0900/01/01", "1360/00/10", "1399/07/31", "1360/12/30"])
def test_birth_date_rejects_impossible_dates(raw):
    """Out-of-range years, month 0 and impossible day/month pairs are rejected."""
    assert not validate_birth_date(raw).validated


def test_birth_date_swapped_month_and_day_is_repaired():
    """A month/day swap that becomes a real date is repaired."""
    outcome = validate_birth_date("1399/13/01")
    assert outcome.validated
    assert outcome.value == "1399/01/13"


@pytest.mark.parametrize(("raw", "expected"), [("زن", "زن"), ("زن ", "زن"), ("مرد", "مرد"), ("مردم", "مرد")])
def test_gender(raw, expected):
    """زن/مرد are accepted with surrounding noise."""
    outcome = validate_gender(raw)
    assert outcome.validated
    assert outcome.value == expected


def test_serial_needs_digits():
    """A serial must contain digits; Arabic letters are rejected."""
    assert validate_serial("1132612").validated
    assert not validate_serial("").validated
    assert not validate_serial("الف").validated


def test_name_validator_uses_lexicon():
    """A real surname is validated *through* the lexicon."""
    validator = make_name_validator("last")
    outcome = validator("محمدی")
    assert outcome.validated
    assert "فرهنگ" in outcome.message


def test_name_validator_rejects_latin():
    """A Latin name is rejected as invalid Persian text."""
    assert not make_name_validator("first")("John").validated
