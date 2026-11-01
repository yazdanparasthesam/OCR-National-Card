"""Command line interface: ``nidocr …``.

    nidocr fields                                   # what can be read
    nidocr read card.jpg                            # whole card, Persian table
    nidocr read card.jpg --json out.json            # machine readable
    nidocr field national_code card.jpg             # one field
    nidocr check 0012345678                         # mod-11 checksum only
    nidocr serve --port 8000                        # FastAPI (uvicorn)
    nidocr ui --port 8501                           # Streamlit interface
    nidocr bench --generate 20 --difficulty photo   # accuracy benchmark
    nidocr doctor                                   # environment diagnostics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nidocr import __version__
from nidocr.config import PROJECT_ROOT, get_settings
from nidocr.fields.definitions import FIELD_SPECS, get_field_spec
from nidocr.fields.lexicon import lexicon_status
from nidocr.fields.validators import validate_birth_date, validate_national_code
from nidocr.ocr.text import to_persian_digits
from nidocr.vision.layout import available_templates

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ── helpers ──────────────────────────────────────────────────────────────────
def _collect_images(targets: list[str]) -> list[Path]:
    """Expand the CLI arguments into image paths (folders are scanned for image files)."""
    images: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            images.append(path)
        else:
            raise SystemExit(f"مسیر پیدا نشد / path not found: {target}")
    return images


def _print_card(result, *, digits: bool = True) -> None:
    """Print one card as a Persian aligned table (value, confidence, validator message).

    ``digits=False`` keeps ASCII digits for terminals without Persian shaping.
    """
    if digits:
        from nidocr.ocr.text import to_persian_digits as fa

    else:

        def fa(text: str) -> str:
            """Identity formatter: print ASCII digits unchanged."""
            return text

    print(f"\n{'=' * 62}")
    print(f" {result.template_title_fa}  ·  {result.processing_ms:.0f} ms  ·  {result.engine}")
    print(f"{'=' * 62}")
    for field in result.ordered_fields():
        value = fa(field.value) if field.value else "—"
        status = "✓" if field.validated else ("~" if field.found else "✗")
        print(f" {status} {field.label_fa:<18} {value:<24} {field.confidence:5.1f}%  {field.message}")
    if result.warnings:
        print("\n هشدارها / warnings:")
        for warning in result.warnings:
            print(f"   • {warning}")


# ── commands ─────────────────────────────────────────────────────────────────
def cmd_read(args: argparse.Namespace) -> int:
    """Command ``nidocr read``: read one image or a folder, optionally write JSON.
    Exit code 1 when a card fails completely, so batch scripts can react.
    """
    from nidocr.pipeline import get_ocr

    ocr = get_ocr()
    images = _collect_images(args.images)
    if not images:
        raise SystemExit("تصویری پیدا نشد / no images")

    payloads = []
    failures = 0
    for path in images:
        result = ocr.extract(path, template=args.template, debug=args.debug)
        failures += 0 if result.success else 1
        if not args.quiet:
            _print_card(result, digits=not args.latin_digits)
        payloads.append(result.model_dump(exclude={"overlay_png", "processed_png"}))

    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = payloads[0] if len(payloads) == 1 else payloads
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON → {target}")

    if len(images) > 1:
        ok = len(images) - failures
        print(f"\n{ok}/{len(images)} کارت خوانده شد")
    return 0 if failures == 0 else 1


def cmd_field(args: argparse.Namespace) -> int:
    """Command ``nidocr field``: read a single field and print value/raw/confidence."""
    from nidocr.pipeline import get_ocr

    spec = get_field_spec(args.field)
    result = get_ocr().extract_field(args.image, spec.key, template=args.template)
    field = result.field
    if args.json:
        print(field.model_dump_json(indent=2, exclude={"roi_png"}))
    else:
        print(f"{field.label_fa}: {to_persian_digits(field.value) or '—'}")
        print(f"  خام / raw      : {field.raw_value}")
        print(f"  اطمینان        : {field.confidence:.1f}%")
        print(f"  معتبر          : {'بله' if field.validated else 'خیر'}")
        print(f"  پیام           : {field.message}")
    return 0 if field.found else 1


def cmd_fields(_args: argparse.Namespace) -> int:
    """Command ``nidocr fields``: list the readable fields with their URL paths."""
    from nidocr.pipeline import get_ocr

    print(f"{'key':<16}{'path':<18}{'label':<18}{'kind':<8}required")
    print("-" * 70)
    for item in get_ocr().describe():
        print(
            f"{item['key']:<16}{item['path']:<18}{item['label_fa']:<18}"
            f"{item['kind']:<8}{'yes' if item['required'] else 'no'}"
        )
    print(f"\n{len(FIELD_SPECS)} fields · templates: {', '.join(available_templates())}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Command ``nidocr check``: mod-11 validation of a national code (exit 1 when invalid)."""
    outcome = validate_national_code(args.value)
    print(json.dumps(
        {
            "input": args.value,
            "value": outcome.value,
            "valid": outcome.validated,
            "repaired": outcome.repaired,
            "message": outcome.message,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if outcome.validated else 1


def cmd_date(args: argparse.Namespace) -> int:
    """Command ``nidocr date``: Jalali validation of a birth date (exit 1 when invalid)."""
    outcome = validate_birth_date(args.value)
    print(json.dumps(
        {
            "input": args.value,
            "value": outcome.value,
            "valid": outcome.validated,
            "repaired": outcome.repaired,
            "message": outcome.message,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if outcome.validated else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Command ``nidocr serve``: start the FastAPI service with uvicorn."""
    import uvicorn

    print(f"NIDOCR API → http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "nidocr.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Command ``nidocr ui``: start the Streamlit interface as a subprocess."""
    app_path = PROJECT_ROOT / "nidocr" / "ui" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"NIDOCR UI → http://{args.host}:{args.port}")
    return subprocess.call(command)  # noqa: S603 - fixed argv


def cmd_bench(args: argparse.Namespace) -> int:
    """Command ``nidocr bench``: delegate to ``scripts/benchmark.py`` with the same arguments."""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "benchmark.py")]
    if args.dataset:
        command += ["--dataset", args.dataset]
    if args.generate:
        command += ["--generate", str(args.generate), "--difficulty", args.difficulty]
    if args.out:
        command += ["--out", args.out]
    if args.title:
        command += ["--title", args.title]
    return subprocess.call(command)  # noqa: S603 - fixed argv


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Command ``nidocr doctor``: report engine, models, lexicon, templates and settings.
    Exit code 1 when the OCR engine is not usable, which makes it CI friendly.
    """
    settings = get_settings()
    from nidocr.ocr.digit_classifier import get_digit_classifier
    from nidocr.ocr.tesseract_engine import get_engine

    engine = get_engine()
    ready, message = engine.ready()
    classifier = get_digit_classifier()
    checks = {
        "version": __version__,
        "python": sys.version.split()[0],
        "tesseract binary": engine.binary or "—",
        "tesseract ready": ready,
        "tesseract message": message,
        "tessdata dir": str(engine.tessdata or "—"),
        "languages": ", ".join(engine.available_languages()) or "—",
        "digit classifier": (
            f"{classifier.metadata.get('samples', '?')} samples" if classifier.available else "missing"
        ),
        "digit holdout accuracy": (
            f"{classifier.accuracy:.3f}" if classifier.accuracy is not None else "—"
        ),
        "lexicon": lexicon_status(),
        "templates": available_templates(),
        "fields": list(FIELD_SPECS),
        "upscale": settings.upscale,
        "passes": settings.passes,
        "cache": settings.cache_ocr,
    }
    width = max(len(key) for key in checks)
    for key, value in checks.items():
        print(f"{key:<{self_width(width)}} : {value}")
    return 0 if ready else 1


def self_width(_width: int) -> int:
    """Padding helper (Persian aware: the terminal pads bytes, not glyphs)."""
    return max(18, _width)


def cmd_templates(_args: argparse.Namespace) -> int:
    """Command ``nidocr templates``: print the available template names."""
    for name in available_templates():
        print(name)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree: one sub-command per ``cmd_*`` function."""
    parser = argparse.ArgumentParser(prog="nidocr", description="Iranian national card OCR (کارت ملی)")
    parser.add_argument("--version", action="version", version=f"nidocr {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    read = sub.add_parser("read", help="خواندن کامل کارت (all fields)")
    read.add_argument("images", nargs="+", help="image file(s) or a directory")
    read.add_argument("--template", default=None, help="template name (default: national_card_front)")
    read.add_argument("--json", default=None, help="write the result as JSON")
    read.add_argument("--debug", action="store_true", help="include ROI/overlay images in the JSON")
    read.add_argument("--quiet", action="store_true", help="no table, just the JSON file")
    read.add_argument("--latin-digits", action="store_true", help="print 1234 instead of ۱۲۳۴")
    read.set_defaults(func=cmd_read)

    field = sub.add_parser("field", help="خواندن یک فیلد (single field)")
    field.add_argument("field", help="national_code | last_name | first_name | father_name | birth_date | serial_number | gender")
    field.add_argument("image", help="path to the card image")
    field.add_argument("--template", default=None)
    field.add_argument("--json", action="store_true")
    field.set_defaults(func=cmd_field)

    fields = sub.add_parser("fields", help="فهرست فیلدها (list fields)")
    fields.set_defaults(func=cmd_fields)

    check = sub.add_parser("check", help="اعتبارسنجی کد ملی (mod-11)")
    check.add_argument("value")
    check.set_defaults(func=cmd_check)

    date = sub.add_parser("date", help="اعتبارسنجی تاریخ تولد (Jalali)")
    date.add_argument("value")
    date.set_defaults(func=cmd_date)

    serve = sub.add_parser("serve", help="اجرای API (FastAPI)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=cmd_serve)

    ui = sub.add_parser("ui", help="اجرای رابط کاربری (Streamlit)")
    ui.add_argument("--host", default="0.0.0.0")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(func=cmd_ui)

    bench = sub.add_parser("bench", help="سنجش دقت (accuracy benchmark)")
    bench.add_argument("--dataset", default=None)
    bench.add_argument("--generate", type=int, default=0)
    bench.add_argument("--difficulty", default="clean", choices=["clean", "photo", "hard"])
    bench.add_argument("--out", default=None)
    bench.add_argument("--title", default=None)
    bench.set_defaults(func=cmd_bench)

    doctor = sub.add_parser("doctor", help="بررسی محیط (diagnostics)")
    doctor.set_defaults(func=cmd_doctor)

    templates = sub.add_parser("templates", help="قالب‌های موجود (list templates)")
    templates.set_defaults(func=cmd_templates)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse the arguments and dispatch to the selected command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
