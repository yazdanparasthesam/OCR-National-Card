"""Train the Persian digit classifier used by the glyph reader.

Run::

    python scripts/train_digit_model.py --cards 90 --per-class 320
    python scripts/train_digit_model.py --holdout Bold --report reports/digits.md

Training data is generated **through the same pipeline the runtime uses**: synthetic
cards are rendered (several Persian font families), enhanced exactly like
:func:`nidocr.vision.enhance.enhance_gray`, the numeric fields are cropped and the
glyphs are segmented with the runtime segmenter.  This matters — a classifier trained
on clean standalone text loses several points once the guilloche background and the
card-level contrast normalisation are in the loop.

The model is a k-NN table (normalised 20×20 glyph bitmaps), stored as a small
``.npz``; a held-out font family is evaluated and the measured accuracy is written
into the model metadata and the markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nidocr.config import PROJECT_ROOT  # noqa: E402
from nidocr.ocr.digit_classifier import FEATURE_SIZE, featurize  # noqa: E402
from nidocr.ocr.digit_reader import is_separator_glyph, segment_digit_glyphs  # noqa: E402
from nidocr.ocr.text import PERSIAN_DIGITS, to_persian_digits  # noqa: E402
from nidocr.synthetic.renderer import CardSpec, FontSet, random_spec, render_card  # noqa: E402
from nidocr.vision.binarize import despeckle, sauvola  # noqa: E402
from nidocr.vision.enhance import crop_with_padding, enhance_gray, upscale  # noqa: E402
from nidocr.vision.layout import load_template, refine_rect  # noqa: E402

NUMERIC_FIELDS = ("national_code", "serial_number", "birth_date")


def discover_fonts(extra_dirs: list[Path]) -> list[Path]:
    folders = [PROJECT_ROOT / "assets" / "fonts" / "train", PROJECT_ROOT / "assets" / "fonts", *extra_dirs]
    fonts: list[Path] = []
    for folder in folders:
        if Path(folder).is_dir():
            fonts.extend(sorted(Path(folder).glob("*.ttf")))
    seen: set[str] = set()
    unique: list[Path] = []
    for font in fonts:
        key = font.stem.lower()
        if key not in seen:
            seen.add(key)
            unique.append(font)
    return unique


def augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Geometric + photometric augmentation.

    Stroke weight is the single most important factor for cross-font generalisation
    (a bold card font looks like a light font that has been dilated), so erosion and
    dilation are applied aggressively.
    """
    out = patch.copy()
    height, width = out.shape[:2]
    angle = rng.uniform(-5, 5)
    scale = rng.uniform(0.86, 1.16, size=2)
    shear = rng.uniform(-0.16, 0.16)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    matrix[0, 1] += shear * scale[1]
    matrix[0, 0] *= scale[0]
    matrix[1, 1] *= scale[1]
    out = cv2.warpAffine(out, matrix, (width, height), borderValue=255)

    operation = rng.integers(0, 4)
    if operation:
        kernel = np.ones((2, 2), np.uint8)
        iterations = int(rng.integers(1, 3))
        out = cv2.erode(out, kernel, iterations=iterations) if operation == 1 else cv2.dilate(
            out, kernel, iterations=iterations
        )

    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0.3, 1.8)))
    if rng.random() < 0.6:
        noise = rng.normal(0, rng.uniform(2, 14), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:  # contrast / gamma variation from photo lighting
        gamma = float(rng.uniform(0.75, 1.35))
        table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)], np.uint8)
        out = cv2.LUT(out, table)
    return out


def segment_field_glyphs(enhanced_card: np.ndarray, rect) -> list[np.ndarray]:
    """Crop + segment one numeric field exactly like the runtime does."""
    roi = crop_with_padding(enhanced_card, rect, pad_ratio=0.11)
    big = upscale(roi, 3.0)
    big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    binary = despeckle(sauvola(big, window=max(15, (big.shape[1] // 8) | 1), k=0.2), min_area_px=8)
    boxes = segment_digit_glyphs(binary)
    return [big[y0:y1, x0:x1] for x0, y0, x1, y1 in boxes if (x1 - x0) >= 2 and (y1 - y0) >= 2]


def collect_from_card(
    spec: CardSpec,
    font: Path,
    layout,
    *,
    rng: np.random.Generator,
    augmentations: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Render one synthetic card and harvest labelled digit glyphs."""
    card = render_card(spec, layout, fonts=FontSet(value=font, label=font, title=font))
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    enhanced = enhance_gray(gray)
    width, height = layout.canvas
    features: list[np.ndarray] = []
    labels: list[int] = []

    for key in NUMERIC_FIELDS:
        text = spec.fields.get(key, "")
        if not text:
            continue
        digits = [char for char in to_persian_digits(text) if char in PERSIAN_DIGITS]
        rect = layout.value_rect(key, width, height)
        if rect is None:
            continue
        refined = refine_rect(enhanced < 128, rect) if True else rect
        crops = [crop for crop in segment_field_glyphs(enhanced, refined) if not is_separator_glyph(crop)]
        if len(crops) != len(digits):  # a missed or merged glyph: unlabelable
            continue
        for crop, digit in zip(crops, digits, strict=True):
            for _ in range(augmentations):
                vector = featurize(augment(crop, rng), FEATURE_SIZE)
                if vector is not None:
                    features.append(vector)
                    labels.append(PERSIAN_DIGITS.index(digit))
    return features, labels


def build_dataset(
    fonts: list[Path],
    layout,
    *,
    cards: int,
    augmentations: int,
    seed: int,
    limit_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[int] = []
    counts = np.zeros(10, dtype=int)
    for index in range(cards):
        font = fonts[index % len(fonts)]
        spec = random_spec(int(rng.integers(0, 10**6)))
        card_features, card_labels = collect_from_card(
            spec, font, layout, rng=rng, augmentations=augmentations
        )
        for vector, label in zip(card_features, card_labels, strict=True):
            if limit_per_class is not None and counts[label] >= limit_per_class:
                continue
            features.append(vector)
            labels.append(label)
            counts[label] += 1
    print(f"  harvested {len(labels)} glyphs · per class {counts.tolist()}")
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def evaluate(features: np.ndarray, labels: np.ndarray, test_features: np.ndarray, test_labels: np.ndarray) -> dict:
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    gallery = features / norms
    predictions = []
    for vector in test_features:
        similarities = gallery @ (vector / (np.linalg.norm(vector) + 1e-8))
        top = min(7, similarities.shape[0])
        indices = np.argpartition(-similarities, top - 1)[:top]
        scores = np.zeros(10, dtype=np.float32)
        for index in indices:
            scores[int(labels[index])] += max(0.0, float(similarities[index])) ** 3
        predictions.append(int(np.argmax(scores)) if scores.sum() > 0 else -1)
    predictions_array = np.array(predictions)
    accuracy = float((predictions_array == test_labels).mean())
    per_class = {
        PERSIAN_DIGITS[index]: round(float((predictions_array[test_labels == index] == index).mean()), 4)
        for index in range(10)
        if (test_labels == index).any()
    }
    return {"accuracy": accuracy, "per_class": per_class, "samples": int(len(test_labels))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the Persian digit k-NN model")
    parser.add_argument("--fonts-dir", action="append", default=[], help="extra folder with .ttf fonts")
    parser.add_argument("--cards", type=int, default=120, help="synthetic cards rendered for training")
    parser.add_argument("--augmentations", type=int, default=8, help="augmented copies per glyph")
    parser.add_argument("--per-class", type=int, default=600, help="cap training glyphs per digit")
    parser.add_argument("--holdout", default="Gandom", help="font (stem substring) kept out of training")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "models" / "persian_digits.npz"))
    parser.add_argument("--report", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    fonts = discover_fonts([Path(folder) for folder in args.fonts_dir])
    if not fonts:
        print("No .ttf fonts found — add some to assets/fonts/train/")
        return 1
    holdout = [font for font in fonts if args.holdout and args.holdout.lower() in font.stem.lower()]
    training = [font for font in fonts if font not in holdout] or fonts
    print(f"fonts: {len(fonts)} · training {[f.stem for f in training]}")
    print(f"holdout: {[f.stem for f in holdout] or '—'}")

    layout = load_template("national_card_front")
    print("· harvesting training glyphs")
    train_features, train_labels = build_dataset(
        training,
        layout,
        cards=args.cards,
        augmentations=args.augmentations,
        seed=args.seed,
        limit_per_class=args.per_class,
    )
    print("· harvesting holdout glyphs")
    holdout_features, holdout_labels = build_dataset(
        holdout or training,
        layout,
        cards=max(6, args.cards // 6),
        augmentations=1,
        seed=args.seed + 1,
    )
    print("· evaluating")
    metrics = evaluate(train_features, train_labels, holdout_features, holdout_labels)
    print(f"  held-out glyph accuracy: {metrics['accuracy'] * 100:.2f}%  per class: {metrics['per_class']}")

    quantised = np.clip(np.round(train_features * 255.0), 0, 255).astype(np.uint8)
    metadata = {
        "version": 2,
        "kind": "persian-digit-knn",
        "feature_size": FEATURE_SIZE,
        "samples": int(len(train_labels)),
        "training_fonts": [font.stem for font in training],
        "holdout_fonts": [font.stem for font in holdout],
        "holdout_accuracy": round(metrics["accuracy"], 5),
        "holdout_per_class": metrics["per_class"],
        "trained_from": "synthetic cards (pipeline-identical preprocessing)",
        "cards": args.cards,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=quantised, labels=train_labels, meta=np.array(json.dumps(metadata)))
    print(f"  saved {out_path} ({out_path.stat().st_size / 1024:.0f} kB, {len(train_labels)} glyphs)")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        rows = "\n".join(f"| {digit} | {value * 100:.1f}% |" for digit, value in metrics["per_class"].items())
        report_path.write_text(
            "# Persian digit classifier\n\n"
            f"- training glyphs: {len(train_labels)}\n"
            f"- fonts: {', '.join(metadata['training_fonts'])}\n"
            f"- held-out font(s): {', '.join(metadata['holdout_fonts']) or '—'}\n"
            f"- held-out glyph accuracy: **{metrics['accuracy'] * 100:.2f}%** "
            f"({metrics['samples']} glyphs)\n\n"
            "| digit | accuracy |\n|---|---|\n" + rows + "\n",
            encoding="utf-8",
        )
        print(f"  report → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
