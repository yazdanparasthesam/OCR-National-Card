"""Persian name lexicon + fuzzy snapping.

Iranian national card name fields contain personal names, i.e. a *closed-ish
vocabulary*.  Tesseract's Persian model confuses the dot patterns of ``ب/پ/ت/ث/ی``
and ``ز/ژ``, which turns ``رضایی`` into ``رضابی``.  Snapping the OCR output to the
closest entry of a name lexicon fixes most of these errors deterministically.

The lexicon files live in ``data/lexicon`` (built by
``scripts/build_name_lexicon.py``) and can be replaced by your own list — for a
production deployment the civil-registry name list is the best choice.

If no lexicon file is present the module becomes a no-op, so the pipeline keeps
working without it.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from nidocr.config import PROJECT_ROOT, get_settings
from nidocr.ocr.text import ZWNJ, normalize_for_compare, normalize_persian

DEFAULT_LEXICON_DIR = PROJECT_ROOT / "data" / "lexicon"
KINDS = ("first", "last")

#: similarity below this is never snapped (only genuine near-misses are corrected)
MIN_SIMILARITY = 0.74

#: letters Tesseract's Persian model mixes up (dot placement/count).  A snap is
#: only accepted when *every* difference is one of these confusion pairs, which
#: keeps the correction precision-first: we never invent or drop a letter, we
#: only re-dot a glyph that OCR read wrong.
_CONFUSION_GROUPS = (
    # the classic one-dot family: Tesseract reads ی/ب/پ/ت/ث/ن for one another
    "بتثنپیئ",
    "سش",
    "صض",
    "طظ",
    "قغ",
    "کگ",
    "هة",
    "اآأإ",
    "دذ",
    "زژ",
    "ذز",
    "حخج",
    "چجح",
)
_CONFUSION: dict[str, set[str]] = {}
for _group in _CONFUSION_GROUPS:
    for _char in _group:
        _CONFUSION.setdefault(_char, set()).update(group_char for group_char in _group if group_char != _char)


def is_confusable(left: str, right: str) -> bool:
    """True when ``left``/``right`` are the same OCR-prone letter."""
    return left == right or right in _CONFUSION.get(left, ())


def confusion_distance(source: str, target: str) -> int | None:
    """Number of confusion-style substitutions turning ``source`` into ``target``.

    Returns ``None`` when the strings cannot be related by letter re-dotting
    alone (different length, or a substitution outside the confusion table).
    """
    if len(source) != len(target):
        return None
    diff = 0
    for left, right in zip(source, target, strict=True):
        if left == right:
            continue
        if not is_confusable(left, right):
            return None
        diff += 1
    return diff
#: a snap must also be noticeably better than the raw form's plausibility
MIN_LENGTH_GAIN = 0.0


#: letters that Persian keyboards/OCR swap freely at the start of a word
_LOOKALIKE = str.maketrans({"آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا", "ة": "ه", "ؤ": "و", "ئ": "ی", "ى": "ی"})


def _key(text: str) -> str:
    """Comparison key: normalised, look-alikes folded, spaces and ZWNJ removed."""
    return normalize_for_compare(text).replace(" ", "").replace(ZWNJ, "").translate(_LOOKALIKE)


class NameLexicon:
    """Fuzzy matcher over a list of Persian names."""

    def __init__(self, first_names: list[str] | None = None, last_names: list[str] | None = None) -> None:
        self._sets: dict[str, set[str]] = {}
        self._lists: dict[str, list[str]] = {}
        self._lookup: dict[str, dict[str, str]] = {}
        self._buckets: dict[str, dict[str, list[str]]] = {}
        for kind, names in (("first", first_names or []), ("last", last_names or [])):
            self._install(kind, names)

    # ── construction ─────────────────────────────────────────────────
    def _install(self, kind: str, names: list[str]) -> None:
        # de-duplicate on the space-free key and keep the cleanest spelling: the
        # upstream first-name list contains stray spaces ("جو ا د") and we want
        # "جواد" to win over them.
        chosen: dict[str, str] = {}
        for name in names:
            value = normalize_persian(name).strip()
            key = _key(value)
            if len(key) < 2:
                continue
            previous = chosen.get(key)
            if previous is None or value.count(" ") < previous.count(" "):
                chosen[key] = value
        cleaned = list(chosen.values())
        self._lists[kind] = cleaned
        self._sets[kind] = set(chosen)
        self._lookup[kind] = dict(chosen)
        buckets: dict[str, list[str]] = {}
        for name in cleaned:
            first = _key(name)[:1]
            # index under every confusing reading of the first letter: ب/پ/ت/ی
            # are exactly the pairs OCR mixes up, so bucketing by the literal
            # first character would hide the right name from the lookup.
            for variant in {first, *_CONFUSION.get(first, ())}:
                buckets.setdefault(variant, []).append(name)
        self._buckets[kind] = buckets

    @classmethod
    def load(cls, directory: str | Path | None = None) -> NameLexicon:
        settings = get_settings()
        base = Path(directory) if directory else (
            Path(settings.lexicon_dir) if settings.lexicon_dir else DEFAULT_LEXICON_DIR
        )
        first_names = _read_lines(base / "first_names.txt")
        last_names = _read_lines(base / "last_names.txt")
        return cls(first_names, last_names)

    # ── queries ──────────────────────────────────────────────────────
    def available(self, kind: str = "first") -> bool:
        return bool(self._lists.get(kind))

    def size(self, kind: str = "first") -> int:
        return len(self._lists.get(kind, []))

    def canonical(self, name: str, kind: str = "first") -> str | None:
        """Return the lexicon's own spelling of ``name`` (``ارمین`` → ``آرمین``)."""
        if not self.available(kind):
            return None
        return self._lookup[kind].get(_key(name))

    def contains(self, name: str, kind: str = "first") -> bool:
        if not self.available(kind):
            return False
        return _key(name) in self._sets[kind]

    def snap(self, name: str, kind: str = "first") -> tuple[str | None, float]:
        """Return ``(best_name, similarity)`` when ``name`` is a near miss."""
        if not self.available(kind) or not name:
            return None, 0.0
        key = _key(name)
        if not key:
            return None, 0.0
        candidates = self._buckets[kind].get(key[:1], [])
        # fall back to a full scan for very short names / unusual first letters
        if len(candidates) < 5:
            candidates = self._lists[kind]
        best_name, best_score, best_diff = None, 0.0, None
        # the surname list is ordered by frequency, so a later candidate must be
        # *clearly* better before it replaces an earlier (more common) one
        for candidate in candidates:
            candidate_key = _key(candidate)
            if len(candidate_key) != len(key):
                continue  # never add/drop a letter: that is not a re-dotting error
            diff = confusion_distance(key, candidate_key)
            if diff is None or diff == 0:
                continue
            score = SequenceMatcher(None, key, candidate_key).ratio()
            if score > best_score + 0.02:
                best_name, best_score, best_diff = candidate, score, diff
        if best_name is None or best_score < MIN_SIMILARITY:
            return None, best_score
        # allow a single confused letter, or two in longer names
        if best_diff is not None and best_diff > (1 if len(key) < 6 else 2):
            return None, best_score
        return best_name, best_score


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=4)
def get_lexicon(directory: str | None = None) -> NameLexicon:
    """Process-wide cached lexicon."""
    return NameLexicon.load(directory)


def lexicon_status() -> dict[str, object]:
    """Small diagnostic dict for the health endpoint / UI."""
    try:
        lexicon = get_lexicon()
    except Exception as exc:  # pragma: no cover - defensive
        return {"available": False, "error": str(exc)}
    return {
        "available": lexicon.available("first") or lexicon.available("last"),
        "first_names": lexicon.size("first"),
        "last_names": lexicon.size("last"),
    }


def normalise_for_lookup(text: str) -> str:
    return normalize_for_compare(unicodedata.normalize("NFKC", text or ""))
