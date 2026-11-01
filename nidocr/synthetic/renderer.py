"""Synthetic national card renderer.

Real Iranian national cards cannot be shipped with a repository, so the project
ships a renderer that draws cards with the *same template geometry the OCR
pipeline expects*, Persian shaped text, a guilloche background, a photo placeholder
and — optionally — realistic phone-camera degradations (perspective, blur, glare,
noise, JPEG).  The generated dataset carries ground truth JSON, which is what the
accuracy benchmark and the test-suite compare against.
"""

from __future__ import annotations

import zlib

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nidocr.config import PROJECT_ROOT
from nidocr.fields.validators import make_national_code
from nidocr.ocr.text import to_persian_digits
from nidocr.synthetic import names as name_bank
from nidocr.vision.layout import TemplateLayout, load_template

DEFAULT_VALUE_FONT = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Medium.ttf"
DEFAULT_LABEL_FONT = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Regular.ttf"
DEFAULT_TITLE_FONT = PROJECT_ROOT / "assets" / "fonts" / "Vazirmatn-Bold.ttf"


@dataclass
class CardSpec:
    """The logical content of one card."""

    fields: dict[str, str]
    gender: str = "مرد"
    photo_hue: int = 25

    @property
    def national_code(self) -> str:
        return self.fields.get("national_code", "")

    def to_dict(self) -> dict:
        return {"fields": dict(self.fields), "gender": self.gender, "photo_hue": self.photo_hue}

    @classmethod
    def from_dict(cls, payload: dict) -> CardSpec:
        return cls(
            fields=dict(payload.get("fields", {})),
            gender=payload.get("gender", "مرد"),
            photo_hue=int(payload.get("photo_hue", 25)),
        )


# ── random data ──────────────────────────────────────────────────────────────
def random_spec(seed: int | None = None) -> CardSpec:
    rng = np.random.default_rng(seed)
    female = bool(rng.integers(0, 2))
    first = name_bank.FEMALE_FIRST_NAMES[int(rng.integers(0, len(name_bank.FEMALE_FIRST_NAMES)))] if female else name_bank.MALE_FIRST_NAMES[int(rng.integers(0, len(name_bank.MALE_FIRST_NAMES)))]
    last = name_bank.LAST_NAMES[int(rng.integers(0, len(name_bank.LAST_NAMES)))]
    father = name_bank.FATHER_NAMES[int(rng.integers(0, len(name_bank.FATHER_NAMES)))]
    seed9 = "".join(str(d) for d in rng.integers(0, 10, size=9))
    if len(set(seed9)) == 1:
        seed9 = "12345678" + seed9[-1]
    year = int(rng.integers(1330, 1385))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29 if month == 12 else 31))
    serial = "".join(str(d) for d in rng.integers(0, 10, size=int(rng.integers(7, 10))))
    return CardSpec(
        fields={
            "national_code": make_national_code(seed9),
            "first_name": first,
            "last_name": last,
            "father_name": father,
            "birth_date": f"{year:04d}/{month:02d}/{day:02d}",
            "serial_number": serial,
            "gender": "زن" if female else "مرد",
        },
        gender="زن" if female else "مرد",
        photo_hue=int(rng.integers(0, 180)),
    )


# ── Persian text rendering ───────────────────────────────────────────────────
def shape(text: str, *, persian_digits: bool = False) -> str:
    """Shape + bidi-reorder Persian text so PIL draws it correctly."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Synthetic card rendering needs `arabic-reshaper` and `python-bidi` "
            "(pip install -r requirements-dev.txt)"
        ) from exc
    value = to_persian_digits(text) if persian_digits else text
    return get_display(arabic_reshaper.reshape(value))


@dataclass
class FontSet:
    value: Path = DEFAULT_VALUE_FONT
    label: Path = DEFAULT_LABEL_FONT
    title: Path = DEFAULT_TITLE_FONT
    cache: dict = field(default_factory=dict)

    def get(self, role: str, size: int) -> ImageFont.FreeTypeFont:
        key = (role, size)
        if key not in self.cache:
            path = getattr(self, role)
            self.cache[key] = ImageFont.truetype(str(path), size)
        return self.cache[key]


# ── background artwork ───────────────────────────────────────────────────────
def _guilloche(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Interference-like lath-work pattern, similar to a real card background."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    field = np.zeros((height, width), np.float32)
    for _ in range(5):
        freq_x = rng.uniform(0.018, 0.075)
        freq_y = rng.uniform(0.014, 0.05)
        phase = rng.uniform(0, 6.283)
        amplitude = rng.uniform(0.4, 1.0)
        modulation = rng.uniform(0.15, 0.55)
        field += amplitude * np.sin(
            xx * freq_x * 6.283
            + modulation * np.sin(yy * freq_y * 6.283 + phase)
            + phase
        )
    field -= field.min()
    field /= max(1e-6, field.max())
    return field


def _background(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base_top = np.array([196, 214, 178], np.float32)  # BGR: pale green
    base_bottom = np.array([170, 186, 210], np.float32)  # pale blue
    if rng.random() < 0.5:
        base_top, base_bottom = base_bottom, base_top
    gradient = (yy / max(1, height - 1))[..., None]
    canvas = base_top * (1 - gradient) + base_bottom * gradient
    pattern = _guilloche(width, height, rng)
    canvas += (pattern[..., None] - 0.5) * rng.uniform(10, 22)
    # diagonal security stripes
    stripes = (np.sin((xx + yy) * rng.uniform(0.01, 0.03)) > 0.93).astype(np.float32)
    canvas -= stripes[..., None] * rng.uniform(6, 14)
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _draw_photo(canvas: np.ndarray, box, hue: int) -> None:
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width < 8 or height < 8:
        return
    photo = np.zeros((height, width, 3), np.uint8)
    top = np.array([90, 120, 160], np.float32)
    bottom = np.array([150, 175, 200], np.float32)
    gradient = (np.mgrid[0:height, 0:width][0] / max(1, height - 1))[..., None]
    photo[:] = (top * (1 - gradient) + bottom * gradient).astype(np.uint8)
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue) % 180
    photo = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # head and shoulders silhouette
    cv2.ellipse(photo, (width // 2, int(height * 0.36)), (int(width * 0.19), int(height * 0.24)), 0, 0, 360, (120, 132, 148), -1)
    cv2.ellipse(photo, (width // 2, int(height * 1.02)), (int(width * 0.44), int(height * 0.42)), 0, 0, 360, (96, 108, 126), -1)
    canvas[y0:y1, x0:x1] = photo


def _draw_emblem(canvas: np.ndarray, box) -> None:
    x0, y0, x1, y1 = box
    centre = ((x0 + x1) // 2, (y0 + y1) // 2)
    radius = max(4, min(x1 - x0, y1 - y0) // 2)
    cv2.circle(canvas, centre, radius, (120, 96, 70), 2, cv2.LINE_AA)
    for angle in range(-90, 91, 30):
        radians = np.deg2rad(angle)
        end = (int(centre[0] + radius * 0.8 * np.cos(radians + np.pi / 2)), int(centre[1] + radius * 0.8 * np.sin(radians + np.pi / 2)))
        cv2.line(canvas, centre, end, (110, 88, 66), 1, cv2.LINE_AA)
    cv2.circle(canvas, centre, max(1, radius // 6), (110, 88, 66), -1, cv2.LINE_AA)


# ── card rendering ───────────────────────────────────────────────────────────
def render_card(
    spec: CardSpec,
    template: TemplateLayout,
    *,
    fonts: FontSet | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Render one card face (BGR) at the template canvas size."""
    fonts = fonts or FontSet()
    canvas_width, canvas_height = canvas_size or template.canvas
    # a *stable* seed: Python's hash() is randomised per process, which would make
    # the same spec render differently (and benchmarks irreproducible)
    signature = "|".join(f"{key}={value}" for key, value in sorted(spec.fields.items()))
    rng = np.random.default_rng(zlib.crc32(signature.encode("utf-8")))

    canvas = _background(canvas_width, canvas_height, rng)
    decorations = template.raw.get("decorations", {})

    def scale_box(box):
        if not box:
            return None
        x0 = int(round(box[0] * canvas_width))
        y0 = int(round(box[1] * canvas_height))
        x1 = int(round(box[2] * canvas_width))
        y1 = int(round(box[3] * canvas_height))
        return x0, y0, x1, y1

    if decorations.get("photo"):
        _draw_photo(canvas, scale_box(decorations["photo"]), spec.photo_hue)
    if decorations.get("emblem"):
        _draw_emblem(canvas, scale_box(decorations["emblem"]))
    if decorations.get("barcode"):
        box = scale_box(decorations["barcode"])
        if box:
            x0, y0, x1, y1 = box
            positions = np.arange(x0, x1, 3)
            for index, x in enumerate(positions):
                if index % 2 == 0:
                    cv2.line(canvas, (int(x), y0), (int(x), y1), (40, 40, 40), 1)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    def draw_text(text: str, box, *, size: int, font_role: str, persian_digits: bool = False, colour=(38, 38, 46)):
        if not box:
            return
        x0, y0, x1, y1 = box
        shaped = shape(text, persian_digits=persian_digits)
        font = fonts.get(font_role, max(9, int(size)))
        bounds = draw.textbbox((0, 0), shaped, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        x = x1 - text_width  # right aligned (RTL card)
        y = y0 + max(0, ((y1 - y0) - text_height) // 2) - bounds[1]
        draw.text((x, y), shaped, font=font, fill=colour)

    for key, geometry in template.field_layout.items():
        label_box = scale_box(geometry.label_box)
        value_box = scale_box(geometry.value_box)
        label_height = (label_box[3] - label_box[1]) if label_box else 24
        value_height = (value_box[3] - value_box[1]) if value_box else 40
        if label_box:
            draw_text(
                _label_text(template, key),
                label_box,
                size=int(label_height * 0.92),
                font_role="label",
                colour=(78, 82, 96),
            )
        value = spec.fields.get(key, "")
        if not value:
            continue
        draw_text(
            value,
            value_box,
            size=int(value_height * 0.86),
            font_role="value",
            persian_digits=key in {"national_code", "birth_date", "serial_number"},
        )

    title_box = scale_box(decorations.get("title"))
    if title_box:
        draw_text(
            template.title_fa,
            title_box,
            size=int((title_box[3] - title_box[1]) * 0.9),
            font_role="title",
            colour=(60, 60, 72),
        )
    footer_box = scale_box(decorations.get("footer"))
    if footer_box:
        draw_text(
            "جمهوری اسلامی ایران",
            footer_box,
            size=int((footer_box[3] - footer_box[1]) * 0.75),
            font_role="label",
            colour=(80, 84, 98),
        )

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _label_text(template: TemplateLayout, key: str) -> str:
    from nidocr.fields.definitions import FIELD_SPECS

    spec = FIELD_SPECS.get(key)
    raw = template.raw.get("field_layout", {}).get(key, {})
    return raw.get("label_text") or (spec.label_fa if spec else key)


# ── degradations ─────────────────────────────────────────────────────────────
DIFFICULTIES = ("clean", "photo", "hard")


def degrade(image: np.ndarray, *, difficulty: str = "photo", seed: int | None = None) -> np.ndarray:
    """Turn a flat card render into something that looks photographed."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if difficulty == "clean":
        return image
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]

    # 1. perspective + rotation, placed on a desk background
    spread = 0.016 if difficulty == "photo" else 0.032
    source = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    target = source.copy()
    for index in range(4):
        target[index] += rng.uniform(-spread, spread, size=2) * np.array([width, height])

    desk_scale = 1.20 if difficulty == "photo" else 1.34
    desk = np.zeros((int(height * desk_scale), int(width * desk_scale), 3), np.uint8)
    wood = np.array([int(rng.integers(90, 150)), int(rng.integers(110, 170)), int(rng.integers(130, 195))])
    desk[:] = np.clip(wood + rng.normal(0, 6, desk.shape), 0, 255).astype(np.uint8)
    offset_x = (desk.shape[1] - width) // 2
    offset_y = (desk.shape[0] - height) // 2
    matrix = cv2.getPerspectiveTransform(source, target + np.float32([offset_x, offset_y]))
    warped = cv2.warpPerspective(image, matrix, (desk.shape[1], desk.shape[0]), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full((height, width), 255, np.uint8), matrix, (desk.shape[1], desk.shape[0]), borderValue=0
    )
    desk[mask > 0] = warped[mask > 0]
    result = desk

    # 2. lighting: shadow gradient + optional specular glare
    yy, xx = np.mgrid[0 : result.shape[0], 0 : result.shape[1]].astype(np.float32)
    gradient = 1.0 + 0.22 * ((xx / result.shape[1]) - 0.5) + 0.14 * ((yy / result.shape[0]) - 0.5)
    result = np.clip(result.astype(np.float32) * gradient[..., None], 0, 255)
    if rng.random() < (0.5 if difficulty == "photo" else 0.85):
        glare_x = int(rng.integers(0, result.shape[1]))
        glare_y = int(rng.integers(0, result.shape[0]))
        radius = int(min(result.shape[:2]) * rng.uniform(0.25, 0.5))
        distance = np.sqrt((xx - glare_x) ** 2 + (yy - glare_y) ** 2)
        glare = np.clip(1 - distance / max(1, radius), 0, 1) ** 2
        result += glare[..., None] * rng.uniform(60, 130)
        result = np.clip(result, 0, 255)

    # 3. optics: blur, noise, resolution and compression
    sigma = rng.uniform(0.4, 0.9) if difficulty == "photo" else rng.uniform(0.8, 1.6)
    result = cv2.GaussianBlur(result, (0, 0), sigma)
    result = np.clip(result + rng.normal(0, rng.uniform(1.5, 5.0) if difficulty == "photo" else rng.uniform(4, 9), result.shape), 0, 255)
    scale = rng.uniform(0.62, 1.0) if difficulty == "photo" else rng.uniform(0.45, 0.65)
    result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    quality = int(rng.integers(70, 93)) if difficulty == "photo" else int(rng.integers(45, 70))
    ok, buffer = cv2.imencode(".jpg", result.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        result = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return result


def save_dataset(
    out_dir: str | Path,
    *,
    count: int = 8,
    difficulty: str = "photo",
    template: str = "national_card_front",
    seed: int = 0,
    prefix: str = "card",
) -> list[dict]:
    """Render ``count`` cards with ground truth JSON next to them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layout = load_template(template)
    records: list[dict] = []
    for index in range(count):
        spec = random_spec(seed * 1000 + index)
        card = render_card(spec, layout)
        image = degrade(card, difficulty=difficulty, seed=seed * 1000 + index)
        name = f"{prefix}_{index + 1:03d}.jpg"
        path = out_dir / name
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:  # pragma: no cover
            raise RuntimeError(f"could not encode {name}")
        path.write_bytes(buffer.tobytes())
        record = {
            "image": name,
            "template": template,
            "difficulty": difficulty,
            "fields": dict(spec.fields),
        }
        (out_dir / f"{prefix}_{index + 1:03d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)
    return records
