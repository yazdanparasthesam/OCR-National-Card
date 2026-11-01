"""Download the model files the OCR stack needs (Tesseract data + digit classifier).

    python scripts/fetch_models.py                 # everything that is missing
    python scripts/fetch_models.py --force         # re-download
    python scripts/fetch_models.py --tesseract-only

The repository ships ``models/persian_digits.npz`` (the Persian glyph classifier)
and ``models/tessdata/{fas,eng,osd}.traineddata``.  This script is here for two
reasons:

* a fresh clone with ``.gitignore``d models can restore them in one command;
* ``--best`` fetches the (slower, marginally more accurate) Tesseract models.

Downloads come from the official GitHub repositories; if you are behind a proxy
set ``HTTPS_PROXY`` as usual.  A locally installed ``tesseract`` distribution also
works: point ``NIDOCR_TESSDATA_DIR`` at it instead of downloading.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESSDATA_DIR = PROJECT_ROOT / "models" / "tessdata"
DIGIT_MODEL = PROJECT_ROOT / "models" / "persian_digits.npz"

SOURCES = {
    "fast": "https://codeload.github.com/tesseract-ocr/tessdata_fast/tar.gz/refs/heads/main",
    "best": "https://codeload.github.com/tesseract-ocr/tessdata_best/tar.gz/refs/heads/main",
}
LANGUAGES = ("fas", "eng", "osd")
CONFIGS = ("tessconfigs",)


def _download(url: str, target: Path) -> None:
    print(f"↓ {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https URL
        target.write_bytes(response.read())


def fetch_tessdata(kind: str, *, force: bool) -> int:
    missing = [code for code in LANGUAGES if force or not (TESSDATA_DIR / f"{code}.traineddata").is_file()]
    if not missing:
        print("tessdata: already present")
        return 0
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "tessdata.tar.gz"
        _download(SOURCES[kind], archive)
        with tarfile.open(archive) as tar:
            members = [
                member
                for member in tar.getmembers()
                if member.name.split("/")[-1] in {f"{code}.traineddata" for code in missing}
            ]
            if not members:
                raise SystemExit("archive did not contain the expected .traineddata files")
            for member in members:
                extracted = tar.extractfile(member)
                if extracted is None:  # pragma: no cover - defensive
                    continue
                target = TESSDATA_DIR / Path(member.name).name
                target.write_bytes(extracted.read())
                print(f"  ✓ {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


def fetch_configs() -> int:
    """The `configs` folder holds psm helpers tesseract looks up by name."""
    target = TESSDATA_DIR / "configs"
    if target.is_dir() and any(target.iterdir()):
        print("configs: already present")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    (target / "digits").write_text("tessedit_char_whitelist 0123456789\n", encoding="utf-8")
    (target / "persian_digits").write_text("tessedit_char_whitelist ۰۱۲۳۴۵۶۷۸۹\n", encoding="utf-8")
    print(f"  ✓ wrote {target}")
    return 0


def check_digit_model() -> int:
    if DIGIT_MODEL.is_file():
        print(f"digit classifier: present ({DIGIT_MODEL.stat().st_size / 1024:.0f} kB)")
        return 0
    print("digit classifier: MISSING — rebuild it with")
    print("  python scripts/train_digit_model.py --cards 120 --augmentations 8 --per-class 600")
    return 1


def find_system_tesseract() -> str | None:
    found = shutil.which("tesseract")
    if found:
        print(f"system tesseract : {found}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="fast", choices=sorted(SOURCES), help="tessdata variant")
    parser.add_argument("--best", action="store_true", help="shorthand for --kind best")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--tesseract-only", action="store_true")
    args = parser.parse_args()

    kind = "best" if args.best else args.kind
    status = fetch_tessdata(kind, force=args.force)
    fetch_configs()
    if not args.tesseract_only:
        status |= check_digit_model()
    find_system_tesseract()
    print("\ndone. Run `nidocr doctor` to verify the setup.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
