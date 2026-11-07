"""Turn a CSV of card labels into the benchmark dataset format.

A labelled set usually arrives as *a folder of images* plus *one CSV*::

    id,national_id,first_name,last_name,date_of_birth,father_name,expiry_date
    1,5116168397,مبینا,اسدی,1380/04/01,نریمان,1409/10/19
    2,5182864095,سمانه,خزایی,1385/08/04,یزدان,1414/11/23

``nidocr.accuracy.load_dataset`` — used by ``scripts/benchmark.py`` and by the «دقت»
tab of the UI — reads one JSON per image instead::

    {"image": "1.jpg", "fields": {"national_code": "5116168397", ...}}

This script bridges the two: it writes one ``<id>.json`` per labelled row and links
the matching image next to it, so the result can be measured directly::

    python scripts/csv_to_dataset.py --csv labels.csv --images cards/ --out data/real
    python scripts/benchmark.py --dataset data/real --difficulty real

Column names are matched case-insensitively against a table of aliases (Persian
headers included); unknown columns are ignored, and a row whose image is missing is
reported and skipped instead of silently producing an empty record.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: CSV header (normalised) → field key used by the pipeline.
COLUMN_ALIASES: dict[str, str] = {
    "national_id": "national_code",
    "national_code": "national_code",
    "nationalcode": "national_code",
    "nationalnumber": "national_code",
    "شمارهملی": "national_code",
    "کدملی": "national_code",
    "first_name": "first_name",
    "firstname": "first_name",
    "name": "first_name",
    "نام": "first_name",
    "last_name": "last_name",
    "lastname": "last_name",
    "surname": "last_name",
    "family": "last_name",
    "نامخانوادگی": "last_name",
    "father_name": "father_name",
    "fathername": "father_name",
    "نامپدر": "father_name",
    "date_of_birth": "birth_date",
    "birth_date": "birth_date",
    "birthdate": "birth_date",
    "dob": "birth_date",
    "تاریخ_تولد": "birth_date",
    "تاریختولد": "birth_date",
    "expiry_date": "expiry_date",
    "expirydate": "expiry_date",
    "valid_until": "expiry_date",
    "پایاناعتبار": "expiry_date",
    "serial_number": "serial_number",
    "serial": "serial_number",
    "شمارهسریال": "serial_number",
    "gender": "gender",
    "sex": "gender",
    "جنسیت": "gender",
}

#: Column that identifies the image file (an ``id`` of 12 is looked up as ``12.jpg``).
ID_COLUMNS = ("image", "file", "filename", "id", "نامفایل", "شماره")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

_FIELD_KEYS = (
    "national_code",
    "first_name",
    "last_name",
    "father_name",
    "birth_date",
    "expiry_date",
    "serial_number",
    "gender",
)


def normalize_header(name: str) -> str:
    """Lower-case a CSV header and strip the punctuation that separates Persian words.

    ``shomare_melli``, ``Shomare Melli`` and ``شماره ملی`` all collapse to one token, so
    the alias table only has to be written once per spelling.
    """
    return re.sub(r"[\s\-_.\u200c:/]+", "", (name or "").strip().lower())


#: The alias table keyed by *normalised* header, so ``national_id`` and ``nationalId``
#: both resolve (the raw spellings above are normalised once, here).
_ALIASES_BY_TOKEN: dict[str, str] = {
    normalize_header(alias): key for alias, key in COLUMN_ALIASES.items()
}
_ID_TOKENS = {normalize_header(alias) for alias in ID_COLUMNS}


def map_columns(header: list[str]) -> dict[str, str]:
    """Return ``csv column → field key`` for every column we understand."""
    mapping: dict[str, str] = {}
    for column in header:
        key = _ALIASES_BY_TOKEN.get(normalize_header(column))
        if key:
            mapping[column] = key
    return mapping


def find_image(images: Path, stem: str) -> Path | None:
    """Locate the image for one row, tolerating a different suffix or case."""
    for suffix in IMAGE_SUFFIXES:
        for candidate in (images / f"{stem}{suffix}", images / f"{stem}{suffix.upper()}"):
            if candidate.is_file():
                return candidate
    for candidate in images.iterdir():
        if candidate.is_file() and candidate.stem == stem:
            return candidate
    return None


def convert(
    csv_path: Path,
    images: Path,
    out: Path,
    *,
    copy: bool = False,
    keep_spaces: bool = True,
    template: str = "auto",
) -> tuple[int, list[str]]:
    """Write one ground-truth JSON per labelled row; return ``(written, skipped)``."""
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        mapping = map_columns(header)
        if not mapping:
            raise SystemExit(f"no recognised column in {csv_path.name}: {header}")
        id_column = next((name for name in header if normalize_header(name) in _ID_TOKENS), None)
        for row in reader:
            fields = {
                key: (row[column] or "").strip()
                for column, key in mapping.items()
                if (row.get(column) or "").strip()
            }
            if not keep_spaces:
                fields = {key: re.sub(r"\s+", "", value) for key, value in fields.items()}
            stem = (row.get(id_column) or "").strip() if id_column else ""
            if not stem:
                skipped.append("<row without an id column value>")
                continue
            image = find_image(images, stem)
            if image is None:
                skipped.append(stem)
                continue
            target = out / image.name
            if not target.exists():
                if copy:
                    shutil.copyfile(image, target)
                else:
                    target.symlink_to(image.resolve())
            # ``template`` is part of the payload contract: without it the benchmark
            # falls back to ``Settings.default_template``, which is the old front design
            # — and every smart card is then read one row off.  A real folder mixes
            # designs, so "auto" is the only honest default.
            payload = {
                "image": image.name,
                "fields": fields,
                "difficulty": "custom",
                "template": template,
            }
            (out / f"{image.stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            written += 1
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``--csv`` + ``--images`` → a ``--out`` dataset folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path, help="label CSV (one row per card)")
    parser.add_argument("--images", required=True, type=Path, help="folder holding the card images")
    parser.add_argument("--out", required=True, type=Path, help="dataset folder to write")
    parser.add_argument("--copy", action="store_true", help="copy the images instead of linking them")
    parser.add_argument(
        "--template",
        default="auto",
        help="template the measurement should use for every card (default: auto-detect)",
    )
    parser.add_argument(
        "--strip-spaces",
        action="store_true",
        help="remove spaces inside labels (some CSVs write «محمد جواد» where the card has «محمدمهدی»)",
    )
    args = parser.parse_args(argv)

    written, skipped = convert(args.csv, args.images, args.out, copy=args.copy,
                               keep_spaces=not args.strip_spaces, template=args.template)
    print(f"{written} record(s) → {args.out}")
    if skipped:
        preview = ", ".join(skipped[:8])
        print(f"{len(skipped)} row(s) skipped (image not found): {preview}"
              + (" …" if len(skipped) > 8 else ""))
    if written:
        print(f"measure it with: python scripts/benchmark.py --dataset {args.out} --difficulty custom")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
