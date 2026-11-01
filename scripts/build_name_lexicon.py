"""Build the Persian name lexicon used to snap OCR results to real names.

The lexicon makes Iranian-card OCR dramatically more accurate on the name fields:
``رضابی`` (a typical ``ی``/``ب`` dot confusion) becomes ``رضایی`` because only the
second is a real surname.

Source data (download once, they are *not* shipped):

* first names — https://github.com/nikahd99/iranian-Names-Database-By-Gender
* surnames    — https://github.com/farbodbj/iranian-surname-frequencies

Usage::

    python scripts/build_name_lexicon.py \
        --first-names-csv /path/iranianNamesDataset.csv \
        --surnames-csv /path/iranian-surname-frequencies.csv \
        --max-surnames 30000

Writes ``data/lexicon/first_names.txt`` and ``data/lexicon/last_names.txt``
(one normalised name per line, most common first).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nidocr.config import PROJECT_ROOT  # noqa: E402
from nidocr.ocr.text import normalize_persian  # noqa: E402

PERSIAN_RANGE = set("آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیءأؤئ‌")


def clean(name: str) -> str:
    """Normalise one source name and drop rows that are not clean Persian words."""
    value = normalize_persian(name).replace("\u200c", " ").strip()
    value = " ".join(part for part in value.split() if part)
    if not value or not all(char in PERSIAN_RANGE or char == " " for char in value):
        return ""
    if len(value) < 2 or len(value) > 30:
        return ""
    return value


def read_first_names(path: Path) -> list[str]:
    """Read the gendered first-name CSV and return unique normalised names."""
    names: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().lower() in {"names", "name"}:
                continue
            value = clean(row[0])
            if value and value not in seen:
                seen.add(value)
                names.append(value)
    return names


def read_surnames(path: Path, limit: int) -> list[str]:
    """Read the surname CSV, sort by frequency and keep the ``limit`` most common."""
    rows: list[tuple[float, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = clean(row.get("name", ""))
            if not value:
                continue
            try:
                frequency = float(row.get("frequency") or 0)
            except ValueError:
                frequency = 0.0
            rows.append((frequency, value))
    rows.sort(key=lambda item: -item[0])
    names: list[str] = []
    seen: set[str] = set()
    for _, value in rows:
        if value not in seen:
            seen.add(value)
            names.append(value)
        if len(names) >= limit:
            break
    return names


def main() -> int:
    """Write ``first_names.txt`` and ``last_names.txt`` into the lexicon folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-names-csv", required=True)
    parser.add_argument("--surnames-csv", required=True)
    parser.add_argument("--max-surnames", type=int, default=30000)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "lexicon"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_names = read_first_names(Path(args.first_names_csv))
    surnames = read_surnames(Path(args.surnames_csv), args.max_surnames)

    (out_dir / "first_names.txt").write_text("\n".join(first_names) + "\n", encoding="utf-8")
    (out_dir / "last_names.txt").write_text("\n".join(surnames) + "\n", encoding="utf-8")
    print(f"first names: {len(first_names)} → {out_dir / 'first_names.txt'}")
    print(f"surnames   : {len(surnames)} → {out_dir / 'last_names.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
