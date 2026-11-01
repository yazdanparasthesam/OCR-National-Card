"""Build a template JSON for a new card revision from a real (or synthetic) card.

The pipeline ships with ``national_card_front`` / ``national_card_back``, but new
print runs move the fields around.  This script rectifies one or more cards the
same way the pipeline does, finds the printed labels with the Persian label
matcher, and writes a template whose boxes are the *measured* label/value
positions (normalised 0..1 so any image size works).

    python scripts/calibrate_template.py card.jpg --name my_card --out nidocr/templates/my_card.json

Inspect the result with ``nidocr read card.jpg --template my_card``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from nidocr.config import PROJECT_ROOT, get_settings  # noqa: E402
from nidocr.fields.definitions import FIELD_SPECS  # noqa: E402
from nidocr.ocr.tesseract_engine import get_engine  # noqa: E402
from nidocr.vision.binarize import otsu  # noqa: E402
from nidocr.vision.card_detector import detect_card, warp_card  # noqa: E402
from nidocr.vision.enhance import enhance_gray, to_gray  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _norm(rect, width: int, height: int) -> list[float]:
    """Normalise a pixel box to 0..1 coordinates for the template file."""
    x0, y0, x1, y1 = rect
    return [
        round(x0 / width, 4),
        round(y0 / height, 4),
        round(x1 / width, 4),
        round(y1 / height, 4),
    ]


def measure(
    image_path: Path, settings, position: str = "below"
) -> dict[str, tuple[list[float], list[float]]]:
    """Return ``{field: (label_box, value_box)}`` in normalised coordinates."""
    from nidocr.image_io import load_image
    from nidocr.ocr.label_anchor import detect_anchors

    image = load_image(image_path)
    canvas_size = (settings.canonical_width, settings.canonical_height)
    detection = detect_card(image, settings)
    card = warp_card(image, detection.quad, size=canvas_size, settings=settings)
    gray = enhance_gray(to_gray(card))
    binary = otsu(gray)

    # the anchors are detected on canvas-sized images so their coordinates are the
    # template's coordinates (no upscale, no padding offset)
    anchors = detect_anchors(
        get_engine(),
        {"enhanced": gray, "otsu": binary},
        settings=settings,
        canvas=canvas_size,
        binary=binary,
    )
    from nidocr.vision.layout import value_rect_from_label

    measured = {}
    for key, (ink_rect, anchor) in anchors.items():
        # use the *derived* box, not the ink-tight one: a template must fit any
        # name, so the width comes from the label geometry and not from the name
        # that happened to be printed on this particular card
        box = value_rect_from_label(anchor.rect, canvas_size, direction=position)
        top, bottom = box[1], box[3]
        right = min(canvas_size[0], anchor.rect[2] + int(0.004 * canvas_size[0]))
        ink_width = ink_rect[2] - ink_rect[0]
        width = max(int(0.34 * canvas_size[0]), int(ink_width + 0.08 * canvas_size[0]))
        left = max(0, right - width)
        measured[key] = (
            _norm(anchor.rect, *canvas_size),
            _norm((left, top, right, bottom), *canvas_size),
        )
    print(f"  {image_path.name}: matched {len(measured)}/{len(FIELD_SPECS)} labels")
    return measured


def merge(measurements: list[dict], width: int, height: int) -> dict:
    """Median of every box across the inspected cards (robust to one bad read)."""
    layout: dict[str, dict] = {}
    for spec in FIELD_SPECS.values():
        labels = [m[spec.key][0] for m in measurements if spec.key in m]
        values = [m[spec.key][1] for m in measurements if spec.key in m]
        if not labels or not values:
            continue
        label_box = [round(statistics.median(axis), 4) for axis in zip(*labels, strict=True)]
        value_box = [round(statistics.median(axis), 4) for axis in zip(*values, strict=True)]
        layout[spec.key] = {"label": label_box, "value": value_box, "align": "right"}
    return layout


def main() -> int:
    """Calibrate a template from card photos and write the JSON (see ``--value-position``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="card image(s) or a directory")
    parser.add_argument("--name", default="custom_card", help="template name")
    parser.add_argument("--title-fa", default="کارت ملی هوشمند")
    parser.add_argument("--title-en", default="Iranian National Card — custom")
    parser.add_argument("--base", default=None, help="template to copy decorations from")
    parser.add_argument(
        "--value-position",
        default="below",
        choices=["below", "left"],
        help="where the printed value sits relative to its label (Iranian cards: below)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    settings = get_settings()
    paths: list[Path] = []
    for target in args.images:
        path = Path(target)
        if path.is_dir():
            paths.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
        else:
            paths.append(path)
    if not paths:
        raise SystemExit("no images found")

    print(f"calibrating '{args.name}' from {len(paths)} card(s)")
    measurements = [measure(path, settings, args.value_position) for path in paths]
    field_layout = merge(measurements, settings.canonical_width, settings.canonical_height)
    if not field_layout:
        raise SystemExit("no labels matched — try sharper images or a lower threshold")

    base_path = Path(args.base) if args.base else PROJECT_ROOT / "nidocr" / "templates" / "national_card_front.json"
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.is_file() else {}
    template = {
        "name": args.name,
        "title_fa": args.title_fa,
        "title_en": args.title_en,
        "canvas": [settings.canonical_width, settings.canonical_height],
        "field_layout": field_layout,
        "decorations": base.get("decorations", {}),
    }
    out = Path(args.out) if args.out else PROJECT_ROOT / "nidocr" / "templates" / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    missing = [key for key in FIELD_SPECS if key not in field_layout]
    print(f"template → {out}")
    print(f"fields  : {', '.join(field_layout)}")
    if missing:
        print(f"missing : {', '.join(missing)} (they will fall back to the shipped template)")
    print("note    : review the boxes (e.g. with `nidocr read card.jpg --template "
          f"{args.name} --debug`) before relying on them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
