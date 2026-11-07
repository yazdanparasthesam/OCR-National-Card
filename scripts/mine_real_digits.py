"""Harvest labelled digit glyphs from real card photos.

The digit reader is a k-NN table trained on *rendered* glyphs.  Real photos add
motion blur, glare, JPEG ringing and a hand-held warp, so a table built on renders
alone is measured on renders and guessed on phones.  This script mines the real
crops instead — from the pictures and the labels the user already has.

Labelling needs no OCR: for a national code, a birth date or an expiry date the
*number of glyphs is known from the label* (ten digits, eight digits), so a field
whose segmentation yields exactly that many non-separator glyphs can be labelled
positionally.  A field whose glyph count does not match is dropped, which is what
keeps a misaligned box out of the training set.

    python scripts/mine_real_digits.py --csv labels.csv --images cards/ --out real_digits.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nidocr.config import PROJECT_ROOT, get_settings  # noqa: E402
from nidocr.image_io import load_image  # noqa: E402
from nidocr.ocr.digit_classifier import FEATURE_SIZE, featurize  # noqa: E402
from nidocr.ocr.digit_reader import is_separator_glyph, segment_digit_glyphs  # noqa: E402
from nidocr.ocr.text import PERSIAN_DIGITS, to_persian_digits  # noqa: E402
from nidocr.vision.binarize import despeckle, sauvola  # noqa: E402
from nidocr.vision.card_detector import detect_card, warp_card  # noqa: E402
from nidocr.vision.enhance import (  # noqa: E402
    card_variants,
    crop_with_padding,
    resize_to_max_side,
    upscale,
)
from nidocr.vision.layout import load_template, refine_rect  # noqa: E402

#: CSV header → the field key the templates use.  Only the fields that carry digits.
COLUMNS = {
    "national_id": "national_code",
    "national_code": "national_code",
    "شناسه ملی": "national_code",
    "کد ملی": "national_code",
    "date_of_birth": "birth_date",
    "birth_date": "birth_date",
    "تاریخ تولد": "birth_date",
    "expiry_date": "expiry_date",
    "تاریخ انقضا": "expiry_date",
}
#: Both designs are tried: which one a card is does not matter when the label decides.
LAYOUTS = ("national_card_smart", "national_card_front")
REPORT_EVERY = 100


def read_labels(path: Path) -> list[dict[str, str]]:
    """Read the CSV, mapping its header onto pipeline field keys."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping = {
            column: COLUMNS[column.strip().lower()]
            for column in (reader.fieldnames or [])
            if column.strip().lower() in COLUMNS
        }
        rows = []
        for raw in reader:
            row = {key: (raw.get(column) or "").strip() for column, key in mapping.items()}
            if row.get("id") or raw.get("id"):
                row["id"] = (raw.get("id") or "").strip()
            rows.append(row)
    return rows


def augment(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric/geometric jitter, mirroring the synthetic trainer's augmentation."""
    height, width = patch.shape[:2]
    angle = float(rng.uniform(-7, 7))
    scale = float(rng.uniform(0.9, 1.1))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    out = cv2.warpAffine(patch, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), float(rng.uniform(0.0, 0.8)))
    noise = rng.normal(0.0, float(rng.uniform(0.0, 6.0)), out.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def field_glyphs(enhanced: np.ndarray, rect, *, max_glyphs: int = 12) -> list[np.ndarray]:
    """Crop and segment one numeric field exactly like the runtime does."""
    roi = crop_with_padding(enhanced, rect, pad_ratio=0.11)
    if roi.size == 0:
        return []
    big = upscale(roi, 3.0)
    big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    binary = despeckle(sauvola(big, window=max(15, (big.shape[1] // 8) | 1), k=0.2), min_area_px=8)
    boxes = segment_digit_glyphs(binary, max_glyphs=max_glyphs)
    crops = []
    for x0, y0, x1, y1 in boxes:
        if (x1 - x0) < 2 or (y1 - y0) < 2:
            continue
        patch = big[y0:y1, x0:x1]
        if not is_separator_glyph(patch):
            crops.append(patch)
    return crops


def mine_card(
    path: Path,
    labels: dict[str, str],
    layouts: dict,
    settings,
    rng: np.random.Generator,
    *,
    augmentations: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Mine one photo: warp, try both designs, keep the fields whose glyphs match the label."""
    image = resize_to_max_side(load_image(path), settings.max_image_side)
    detection = detect_card(image, settings)
    canvas = warp_card(image, detection.quad, (settings.canonical_width, settings.canonical_height), settings)
    enhanced = card_variants(canvas)["enhanced"]
    height, width = enhanced.shape[:2]
    features: list[np.ndarray] = []
    labels_out: list[int] = []
    for layout in layouts.values():
        for key, text in labels.items():
            if key not in ("national_code", "birth_date", "expiry_date") or not text:
                continue
            digits = [char for char in to_persian_digits(text) if char in PERSIAN_DIGITS]
            if not digits:
                continue
            rect = layout.value_rect(key, width, height)
            if rect is None:
                continue
            refined = refine_rect(enhanced < 128, rect)
            crops = field_glyphs(enhanced, refined, max_glyphs=len(digits) + 3)
            if len(crops) != len(digits):
                continue
            heights = np.array([crop.shape[0] for crop in crops], dtype=float)
            median = float(np.median(heights))
            if median < 8 or np.any(heights < 0.55 * median) or np.any(heights > 1.8 * median):
                continue
            for crop, char in zip(crops, digits, strict=True):
                for _ in range(augmentations):
                    vector = featurize(augment(crop, rng), FEATURE_SIZE)
                    if vector is not None:
                        features.append(vector)
                        labels_out.append(PERSIAN_DIGITS.index(char))
    return features, labels_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine labelled digit glyphs from real cards")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "real_digits.npz")
    parser.add_argument("--limit", type=int, default=0, help="stop after N cards (0 = all)")
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    settings = get_settings()
    layouts = {name: load_template(name) for name in LAYOUTS}
    rng = np.random.default_rng(args.seed)
    rows = read_labels(args.csv)
    features: list[np.ndarray] = []
    labels: list[int] = []
    used = 0
    for index, row in enumerate(rows, start=1):
        image_path = args.images / f"{row.get('id', '')}.jpg"
        if not image_path.exists():
            continue
        card_features, card_labels = mine_card(
            image_path, row, layouts, settings, rng, augmentations=args.augmentations
        )
        if card_labels:
            used += 1
            features.extend(card_features)
            labels.extend(card_labels)
        if index % REPORT_EVERY == 0:
            print(f"  {index} rows → {used} usable cards, {len(labels)} glyphs", flush=True)
        if args.limit and used >= args.limit:
            break
    if not labels:
        print("no labelled glyphs mined — is the CSV in the expected format?", file=sys.stderr)
        return 1
    matrix = np.array(features, dtype=np.uint8)
    np.savez_compressed(args.out, features=matrix, labels=np.array(labels, dtype=np.uint8))
    counts = np.bincount(np.array(labels, dtype=np.uint8), minlength=len(PERSIAN_DIGITS))
    print(f"{len(labels)} glyphs from {used} cards → {args.out}")
    print("per digit:", dict(zip(PERSIAN_DIGITS, counts.tolist(), strict=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
