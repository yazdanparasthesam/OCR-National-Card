"""Persian/Arabic text normalisation utilities.

Tesseract with the ``fas`` model emits Arabic-script code points that are not the
Persian ones (``ي`` instead of ``ی``, ``ك`` instead of ``ک``), a mix of Arabic-Indic,
Persian and Latin digits, bidi control characters and stray punctuation.  Every
field of the card is normalised here before validation.
"""

from __future__ import annotations

import re
import unicodedata

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"

ZWNJ = "\u200c"  # نیم‌فاصله

PERSIAN_LETTERS = set(
    "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأؤئۀ"
)

# ── character level translation tables ───────────────────────────────────────
_DIGIT_TRANSLATION: dict[int, str] = {}
for _index, _char in enumerate(PERSIAN_DIGITS):
    _DIGIT_TRANSLATION[ord(_char)] = LATIN_DIGITS[_index]
for _index, _char in enumerate(ARABIC_DIGITS):
    _DIGIT_TRANSLATION[ord(_char)] = LATIN_DIGITS[_index]

_TO_PERSIAN_DIGIT_TRANSLATION = {ord(LATIN_DIGITS[i]): PERSIAN_DIGITS[i] for i in range(10)}

_CHAR_TRANSLATION = {
    ord("ي"): "ی",
    ord("ى"): "ی",
    ord("ﻯ"): "ی",
    ord("ﻰ"): "ی",
    ord("ﻲ"): "ی",
    ord("ك"): "ک",
    ord("ﻙ"): "ک",
    ord("ﻚ"): "ک",
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("ٱ"): "ا",
    ord("ة"): "ه",
    ord("ۀ"): "ه",
    ord("ؤ"): "و",
    ord("ئ"): "ی",
    ord("ـ"): None,  # tatweel / کشیده
}

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u06D6-\u06ED]")
_BIDI_MARKS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u200b]")
_WHITESPACE = re.compile(r"\s+")
_EDGE_PUNCT = " \t\n\r.,:;،؛ـ-_=~!?؟«»\"'`()[]{}|/\\<>*#٪%+&^$@"

# Latin letters that tesseract confuses with Persian ones inside digits/serials
_LATIN_TO_DIGIT = {"O": "0", "o": "0", "I": "1", "l": "1", "i": "1", "Z": "2", "S": "5", "B": "8"}


def to_ascii_digits(text: str) -> str:
    """Convert Persian/Arabic-Indic digits to Latin digits (۰۱۲ → 012)."""
    return text.translate(_DIGIT_TRANSLATION)


def to_persian_digits(text: str) -> str:
    """Convert Latin digits to Persian digits (012 → ۰۱۲)."""
    return text.translate(_TO_PERSIAN_DIGIT_TRANSLATION)


def strip_bidi(text: str) -> str:
    """Remove bidi control characters and zero-width spaces."""
    return _BIDI_MARKS.sub("", text or "")


def normalize_persian(text: str) -> str:
    """Full Persian normalisation: code points, diacritics, whitespace, digits kept."""
    if not text:
        return ""
    value = strip_bidi(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(_CHAR_TRANSLATION)
    value = _DIACRITICS.sub("", value)
    value = value.replace("\u00a0", " ")
    value = _WHITESPACE.sub(" ", value)
    return value.strip()


def normalize_for_compare(text: str) -> str:
    """Aggressive form used by accuracy metrics and fuzzy label matching."""
    value = normalize_persian(text)
    value = to_ascii_digits(value)
    value = value.replace(ZWNJ, " ")
    value = re.sub(r"[^\w\s]|_", " ", value)
    value = _WHITESPACE.sub(" ", value)
    return value.strip()


def is_persian_letter(char: str) -> bool:
    """True when ``char`` is a Persian/Arabic letter (not a digit/punctuation)."""
    return char in PERSIAN_LETTERS or ("\u0621" <= char <= "\u063a") or ("\u0641" <= char <= "\u064a")


def has_persian_letters(text: str) -> bool:
    """True when the string contains at least one Persian/Arabic letter."""
    return any(is_persian_letter(char) for char in text)


def persian_ratio(text: str) -> float:
    """Share of Persian letters among all letters/digits of ``text``."""
    relevant = [char for char in text if is_persian_letter(char) or char.isdigit() or char.isalpha()]
    if not relevant:
        return 0.0
    return sum(is_persian_letter(char) for char in relevant) / len(relevant)


def digit_ratio(text: str) -> float:
    """Share of digit characters among the non-space characters of ``text``."""
    relevant = [char for char in text if not char.isspace()]
    if not relevant:
        return 0.0
    return sum(char.isdigit() for char in relevant) / len(relevant)


def only_digits(text: str) -> str:
    """Keep digits only, normalised to Latin (and fixing classic OCR look-alikes)."""
    value = to_ascii_digits(normalize_persian(text))
    cleaned = []
    for char in value:
        if char.isdigit():
            cleaned.append(char)
        elif char in _LATIN_TO_DIGIT and char.isalpha() and not char.islower():
            cleaned.append(_LATIN_TO_DIGIT[char])
    return "".join(cleaned)


def only_alnum(text: str) -> str:
    """Keep Persian letters, Latin letters, digits and ZWNJ."""
    value = normalize_persian(text).replace(ZWNJ, " ")
    return "".join(char for char in value if char.isalnum() or char.isspace()).strip()


def strip_edges(text: str) -> str:
    """Remove leading/trailing punctuation and whitespace."""
    return (text or "").strip(_EDGE_PUNCT)


def collapse_spaces(text: str) -> str:
    """Collapse every run of whitespace to a single space and trim the ends."""
    return _WHITESPACE.sub(" ", text or "").strip()


def clean_ocr_text(text: str) -> str:
    """Normalise raw tesseract output without dropping semantic content."""
    value = normalize_persian(text)
    value = strip_edges(value)
    value = collapse_spaces(value)
    # a very common artefact: the label colon ends up glued to the value
    value = value.lstrip(":؛،.و ").strip()
    return value


def words(text: str) -> list[str]:
    """Split a string into non-empty words after normalisation."""
    return [word for word in re.split(r"[\s" + ZWNJ + r"]+", normalize_persian(text)) if word]


def sort_tokens_rtl(tokens: list[tuple[str, int]]) -> str:
    """Join ``(text, x_left)`` tokens in right-to-left reading order."""
    ordered = sorted(tokens, key=lambda item: -item[1])
    return collapse_spaces(" ".join(token for token, _ in ordered))


def sort_tokens_ltr(tokens: list[tuple[str, int]]) -> str:
    """Join ``(text, x_left)`` tokens in left-to-right order (mirror of the RTL helper)."""
    ordered = sorted(tokens, key=lambda item: item[1])
    return collapse_spaces(" ".join(token for token, _ in ordered))


def similarity(left: str, right: str) -> float:
    """Normalised similarity (0..1) between two short strings, Persian aware."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, normalize_for_compare(left), normalize_for_compare(right)).ratio()
