"""Field level validators and self-healing corrections.

Iranian national card fields are strongly typed, so OCR output can be *verified*
instead of blindly trusted:

* ``national_code`` — 10 digits with a mod-11 check digit: we can detect a wrong
  digit and even repair a single substitution/insertion/deletion.
* ``birth_date`` — Jalali (Solar Hijri) date: month lengths and leap years are
  known, so impossible dates are rejected and re-ordered.
* ``serial_number`` — digits with optional Latin letter suffix.
* names / gender — Persian alphabet and a closed vocabulary respectively.

Every validator returns :class:`FieldOutcome` with a ``score`` in ``[0, 1]`` used
by the pipeline to rank competing OCR candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nidocr.config import get_settings
from nidocr.ocr.text import (
    ZWNJ,
    collapse_spaces,
    digit_ratio,
    is_persian_letter,
    normalize_persian,
    only_digits,
    persian_ratio,
    to_ascii_digits,
)

# برای ساخت رشتهٔ معتبر از روی یک ورودی خام
_PERSIAN_LETTER_SET = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأؤئ")


@dataclass
class FieldOutcome:
    """Result of normalising (+validating/reparing) one field."""

    value: str
    validated: bool = False
    repaired: bool = False
    score: float = 0.0
    messages: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        """Human readable Persian explanation of the outcome (the API returns it verbatim)."""
        return "؛ ".join(self.messages)


# ─────────────────────────────────────────────────────────────────────────────
# national code (کد ملی) — mod 11 check digit
# ─────────────────────────────────────────────────────────────────────────────
def national_code_checksum(code: str) -> int | None:
    """Return the expected check digit for the first 9 digits of ``code``."""
    digits = only_digits(code)
    if len(digits) < 9:
        return None
    total = sum(int(digit) * (10 - index) for index, digit in enumerate(digits[:9]))
    remainder = total % 11
    return remainder if remainder < 2 else 11 - remainder


def is_valid_national_code(code: str) -> bool:
    """Validate an Iranian national code (code ملی) including the check digit."""
    digits = only_digits(code)
    if len(digits) != 10:
        return False
    if len(set(digits)) == 1:  # 0000000000, 1111111111, ... are invalid
        return False
    return national_code_checksum(digits) == int(digits[9])


def _substitutions(digits: str) -> list[str]:
    """All strings obtained by replacing exactly one digit (used for unique-repair search)."""
    candidates: list[str] = []
    for index, original in enumerate(digits):
        for digit in "0123456789":
            if digit != original:
                candidates.append(f"{digits[:index]}{digit}{digits[index + 1:]}")
    return candidates


def _insertions(digits: str) -> list[str]:
    """All strings obtained by inserting one digit at any position (a lost glyph)."""
    candidates: list[str] = []
    for index in range(len(digits) + 1):
        for digit in "0123456789":
            candidates.append(f"{digits[:index]}{digit}{digits[index:]}")
    return candidates


def _deletions(digits: str) -> list[str]:
    """All strings obtained by deleting one digit (a spurious segmented glyph)."""
    return [f"{digits[:index]}{digits[index + 1:]}" for index in range(len(digits))]


def validate_national_code(raw: str) -> FieldOutcome:
    """Normalise, validate and (when possible) repair a national code.

    Repair is deliberately **tiered**, because a blind "find any valid code one
    digit away" search always succeeds (≈ 100/11 candidates pass the checksum) and
    would happily rewrite a correct code into a wrong one:

    1. the code is valid — nothing to do;
    2. the first nine digits are intact and only the check digit is wrong — the
       checksum itself gives the right digit, so this repair is safe;
    3. a single wrong digit somewhere: accepted **only** when exactly one valid
       candidate exists (otherwise the reading is ambiguous and is reported as
       invalid instead of guessed);
    4. a digit was lost / doubled (length 9 or 11): same uniqueness rule.
    """
    digits = only_digits(raw)
    if not digits:
        return FieldOutcome("", False, False, 0.0, ["شماره ملی خوانده نشد"])

    if len(digits) == 10 and is_valid_national_code(digits):
        return FieldOutcome(digits, True, False, 1.0, ["رقم کنترل معتبر است"])

    # 2) exactly one digit was read wrong, and only one digit can fix it:
    #    that is the most trustworthy repair, so it is tried first
    attempts: list[tuple[str, str]] = []
    if len(digits) == 10:
        attempts = [(candidate, "یک رقم اصلاح شد") for candidate in _substitutions(digits)]
    elif len(digits) == 9:
        attempts = [(candidate, "یک رقم جاافتاده بازسازی شد") for candidate in _insertions(digits)]
    elif len(digits) == 11:
        attempts = [(candidate, "یک رقم اضافه حذف شد") for candidate in _deletions(digits)]

    repairs = [candidate for candidate, _ in attempts if is_valid_national_code(candidate)]
    if len(repairs) == 1:
        message = next(message for candidate, message in attempts if candidate == repairs[0])
        return FieldOutcome(repairs[0], True, True, 0.78, [message])

    # 3) fall back to the checksum itself: keep the first nine digits, derive the
    #    check digit.  Scored low on purpose — the first nine digits may be the
    #    wrong ones, so a genuinely valid alternative must be able to win.
    if len(digits) == 10 and len(set(digits)) > 1:
        expected = str(national_code_checksum(digits[:9]))
        if expected != digits[9]:
            candidate = digits[:9] + expected
            if is_valid_national_code(candidate):
                return FieldOutcome(candidate, True, True, 0.62, ["رقم کنترل از روی ۹ رقم اول اصلاح شد"])

    score = 0.25 * digit_ratio(raw) + (0.1 if len(digits) == 10 else 0.0)
    return FieldOutcome(
        digits,
        False,
        False,
        round(score, 3),
        [f"رقم کنترل نامعتبر است (طول {len(digits)})"],
    )


def make_national_code(seed9: str) -> str:
    """Build a checksum-valid national code from 9 digits (used by tests/datasets)."""
    digits = only_digits(seed9).ljust(9, "3")[:9]
    return digits + str(national_code_checksum(digits))


# ─────────────────────────────────────────────────────────────────────────────
# Jalali dates (تاریخ تولد)
# ─────────────────────────────────────────────────────────────────────────────
def is_jalali_leap(year: int) -> bool:
    """True for leap years of the 33-year Jalali cycle."""
    return (year % 33) in {1, 5, 9, 13, 17, 22, 26, 30}


def jalali_month_days(year: int, month: int) -> int:
    """Number of days of a Jalali month (1–6 → 31, 7–11 → 30, 12 → 29/30 by leap rule)."""
    if month <= 6:
        return 31
    if month <= 11:
        return 30
    return 30 if is_jalali_leap(year) else 29


def is_valid_jalali_date(year: int, month: int, day: int) -> bool:
    """True when the year/month/day triple is a real Jalali date inside the birth-year window."""
    settings = get_settings()
    if not (settings.birth_year_min <= year <= settings.birth_year_max):
        return False
    if not (1 <= month <= 12):
        return False
    return 1 <= day <= jalali_month_days(year, month)


_SEPARATORS = re.compile(r"[\/\-.،,،\s_|]+")


def parse_date_numbers(raw: str) -> list[int]:
    """Extract the numbers of a date string (digits only, separators ignored)."""
    cleaned = to_ascii_digits(normalize_persian(raw))
    parts = [part for part in _SEPARATORS.split(cleaned) if part]
    numbers: list[int] = []
    for part in parts:
        digits = re.sub(r"\D", "", part)
        if digits:
            numbers.append(int(digits))
    if len(numbers) < 3:
        # a date printed without separators: 13650412 / 650412
        digits = "".join(re.sub(r"\D", "", part) for part in parts)
        if len(digits) == 8:
            numbers = [int(digits[:4]), int(digits[4:6]), int(digits[6:])]
        elif len(digits) == 6:
            numbers = [int(digits[:2]), int(digits[2:4]), int(digits[4:])]
    return numbers


def validate_birth_date(raw: str) -> FieldOutcome:
    """Normalise a Jalali birth date to ``YYYY/MM/DD``."""
    numbers = parse_date_numbers(raw)
    if len(numbers) < 3:
        return FieldOutcome("", False, False, 0.0, ["تاریخ تولد خوانده نشد"])

    year, month, day = numbers[0], numbers[1], numbers[2]
    messages: list[str] = []
    repaired = False

    if year < 100:  # 65/3/7 → 1365/03/07
        year += 1300
        repaired = True
        messages.append("سال دورقمی بازسازی شد")

    if not (1 <= month <= 12) and 1 <= day <= 12:  # swapped month/day
        month, day = day, month
        repaired = True
        messages.append("جای ماه و روز اصلاح شد")

    valid = is_valid_jalali_date(year, month, day)
    value = f"{year:04d}/{month:02d}/{day:02d}"
    if not valid:
        settings = get_settings()
        score = 0.3 if settings.birth_year_min <= year <= settings.birth_year_max else 0.1
        return FieldOutcome(value, False, repaired, score, ["تاریخ در تقویم جلالی معتبر نیست"])
    return FieldOutcome(value, True, repaired, 1.0, ["تاریخ معتبر است"] if not repaired else messages)


def validate_expiry_date(raw: str) -> FieldOutcome:
    """Normalise the smart card's «پایان اعتبار» date to ``YYYY/MM/DD``.

    Same Jalali rules as the birth date, except for the year: an expiry date is in
    the *future* while a birth date is in the past, so a two-digit year is rebuilt
    as 14xx and the ``birth_year_range`` setting is not applied.
    """
    numbers = parse_date_numbers(raw)
    if len(numbers) < 3:
        return FieldOutcome("", False, False, 0.0, ["تاریخ پایان اعتبار خوانده نشد"])
    year, month, day = numbers[0], numbers[1], numbers[2]
    messages: list[str] = []
    repaired = False
    if year < 100:  # 06/03/14 → 1406/03/14
        year += 1400
        repaired = True
        messages.append("سال دورقمی بازسازی شد")
    if not (1 <= month <= 12) and 1 <= day <= 12:
        month, day = day, month
        repaired = True
        messages.append("جای ماه و روز اصلاح شد")
    value = f"{year:04d}/{month:02d}/{day:02d}"
    if not is_valid_jalali_date(year, month, day):
        return FieldOutcome(value, False, repaired, 0.2, ["تاریخ در تقویم جلالی معتبر نیست"])
    if not 1390 <= year <= 1450:
        return FieldOutcome(value, False, repaired, 0.4, ["سال پایان اعتبار منطقی نیست"])
    return FieldOutcome(value, True, repaired, 1.0, ["تاریخ معتبر است"] if not repaired else messages)


# ─────────────────────────────────────────────────────────────────────────────
# serial number (شماره سریال)
# ─────────────────────────────────────────────────────────────────────────────
_SERIAL_ALLOWED = re.compile(r"[^0-9A-Za-z]")


def normalize_serial(raw: str) -> str:
    """Normalise a serial number: ASCII digits, upper-case Latin letters, no separators."""
    value = to_ascii_digits(normalize_persian(raw))
    value = _SERIAL_ALLOWED.sub("", value)
    # drop a leading label artefact such as "IR"
    return value.upper()


def validate_serial(raw: str) -> FieldOutcome:
    """Validate a card serial (digits/Latin, 5–14 characters) and report the cleaned value."""
    value = normalize_serial(raw)
    digits = sum(char.isdigit() for char in value)
    if not value:
        return FieldOutcome("", False, False, 0.0, ["شماره سریال خوانده نشد"])
    if 5 <= len(value) <= 14 and digits >= 4:
        return FieldOutcome(value, True, len(value) != len(raw.strip()), 1.0, ["قالب معتبر است"])
    score = 0.2 + 0.3 * min(digits / 6, 1.0)
    return FieldOutcome(value, False, False, round(score, 3), ["قالب شماره سریال نامعتبر است"])


# ─────────────────────────────────────────────────────────────────────────────
# names (نام / نام خانوادگی / نام پدر)
# ─────────────────────────────────────────────────────────────────────────────
_NAME_DROP = re.compile(r"[^آ-یءأؤئ" + ZWNJ + r" ]+")
_NAME_MAX = 45
LEXICON_SNAP_MIN = 0.86
"""Similarity a reading must reach before it is replaced by a lexicon entry.

Lower values were tempting (a surname outside the lexicon looks like a typo) but they
also rewrote *correct* reads: on the real card «شرجی» (0.75 similar to «سرخی») was
silently replaced. 0.86 keeps the useful repairs (a dropped dot or a missing letter in
long names) without inventing a different word for a short one."""


def normalize_name(raw: str) -> str:
    """Clean a Persian name: fold code points, drop punctuation, collapse spaces, cap words."""
    value = normalize_persian(raw)
    value = _NAME_DROP.sub(" ", value)
    value = collapse_spaces(value.replace("  ", " "))
    # a lone "ی"/"ا" at the end is a very common bidi artefact
    tokens = [token for token in value.split(" ") if token and (len(token) > 1 or token in {"و", "آ"})]
    return collapse_spaces(" ".join(tokens))[: _NAME_MAX + 10]


def validate_name(raw: str, *, required: bool = True, kind: str = "any", use_lexicon: bool = True) -> FieldOutcome:
    """Validate a Persian name, snapping near misses to the name lexicon."""
    value = normalize_name(raw)
    if not value:
        return FieldOutcome("", not required, False, 0.0, ["مقدار خوانده نشد"])
    letters = [char for char in value.replace(" ", "") if is_persian_letter(char)]
    if not letters:
        return FieldOutcome(value, False, False, 0.1, ["متن فارسی معتبر نیست"])
    length_ok = 2 <= len(value) <= _NAME_MAX

    if use_lexicon and kind in {"first", "last"} and get_settings().use_name_lexicon:
        lexicon_kind = "last" if kind == "last" else "first"
        from nidocr.fields.lexicon import get_lexicon

        lexicon = get_lexicon()
        if lexicon.available(lexicon_kind):
            stored = lexicon.canonical(value, lexicon_kind)
            if stored is not None:
                # keep the lexicon's spelling: OCR drops the madda of «آ»
                return FieldOutcome(
                    normalize_name(stored), length_ok, False, 1.0, ["نام در فرهنگ واژگان یافت شد"]
                )
            snapped, score = lexicon.snap(value, lexicon_kind)
            if snapped and score >= LEXICON_SNAP_MIN:
                # a snap is a *repair*, never a validation: the value is different from
                # what was read, so it must stay visibly less certain than an exact hit
                return FieldOutcome(
                    normalize_name(snapped),
                    False,
                    True,
                    0.55 + 0.3 * score,
                    [f"با فرهنگ واژگان اصلاح شد ({score:.0%} تشابه)"],
                )
    one_token = " " not in value
    messages: list[str] = []
    if not one_token:
        messages.append("چند کلمه تشخیص داده شد")
    if not length_ok:
        messages.append("طول متن غیرعادی است")
    # a plausible word outside the lexicon is *weaker evidence* than a real name
    score = 0.82 if use_lexicon and kind in {"first", "last"} else 1.0
    score -= 0.15 * max(0, len(value.split(" ")) - 3)
    if not length_ok:
        score -= 0.4
    if len(value) <= 2:
        score -= 0.2
    return FieldOutcome(
        value,
        length_ok,
        value != collapse_spaces(normalize_persian(raw)),
        round(max(0.05, min(score, 1.0)), 3),
        messages or ["نام فارسی معتبر است (خارج از فرهنگ واژگان)"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# gender (جنسیت)
# ─────────────────────────────────────────────────────────────────────────────
_GENDER_MAP = {
    "مرد": "مرد",
    "م": "مرد",
    "male": "مرد",
    "m": "مرد",
    "زن": "زن",
    "ز": "زن",
    "female": "زن",
    "f": "زن",
}


def validate_gender(raw: str) -> FieldOutcome:
    """Validate the gender field (زن / مرد), tolerating trailing letters and spacing."""
    value = normalize_persian(raw).replace(ZWNJ, "").strip(" .:،؛")
    key = value.lower()
    if key in _GENDER_MAP:
        return FieldOutcome(_GENDER_MAP[key], True, _GENDER_MAP[key] != value, 1.0, ["مقدار معتبر است"])
    if value:
        best = None
        for candidate in ("مرد", "زن"):
            if candidate in value or value in candidate:
                best = candidate
        if best:
            return FieldOutcome(best, True, True, 0.8, ["مقدار اصلاح شد"])
    return FieldOutcome(value, False, False, 0.0, ["جنسیت تشخیص داده نشد"])


# ─────────────────────────────────────────────────────────────────────────────
# generic helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_name_validator(kind: str):
    """Build a name finaliser bound to a lexicon kind (``first`` / ``last``)."""

    def _validate(raw: str, *, required: bool = True) -> FieldOutcome:
        """Finalise one name reading with the lexicon kind the field is bound to."""
        return validate_name(raw, required=required, kind=kind)

    _validate.__name__ = f"validate_{kind}_name"
    return _validate


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a score to the 0..1 range used by :class:`FieldOutcome`."""
    return max(low, min(high, value))


def persian_quality(text: str) -> float:
    """0..1 heuristic used to rank name candidates (more letters & fewer oddities)."""
    if not text:
        return 0.0
    ratio = persian_ratio(text)
    penalty = 0.1 * max(0, len(text.split()) - 3)
    return clamp(ratio - penalty)
