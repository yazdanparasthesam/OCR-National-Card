"""Image loading helpers (bytes, path, file object, numpy, PIL)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

from nidocr.errors import InvalidImageError

MAX_PIXELS = 40_000_000  # ~40 MP guard against decompression bombs


def _to_array(pil_image) -> np.ndarray:
    """Convert a PIL image to a BGR ``uint8`` numpy array (dropping alpha)."""
    array = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def load_image(source) -> np.ndarray:
    """Load ``source`` as a BGR ``uint8`` numpy array.

    Accepts a filesystem path, raw bytes, a file-like object, a PIL image or an
    already decoded numpy array.
    """
    if source is None:
        raise InvalidImageError("No image provided")

    if isinstance(source, np.ndarray):
        image = source
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return image

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"File not found: {path}")
        data = path.read_bytes()
        return _decode(data, name=path.name)

    if isinstance(source, (bytes, bytearray, memoryview)):
        return _decode(bytes(source))

    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            raise InvalidImageError("File object returned text, expected binary image data")
        return _decode(bytes(data))

    if hasattr(source, "convert") or hasattr(source, "mode"):
        try:  # PIL image
            return _to_array(source)
        except Exception as exc:  # pragma: no cover - defensive
            raise InvalidImageError(f"Unsupported PIL image: {exc}") from exc

    raise InvalidImageError(f"Unsupported image source type: {type(source)!r}")


def _decode(data: bytes, name: str | None = None) -> np.ndarray:
    """Decode image bytes with OpenCV, falling back to PIL for exotic formats.
    Also enforces the 40-megapixel guard so a huge upload cannot exhaust memory.
    """
    if not data:
        raise InvalidImageError("Empty image payload")
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        # last resort: PIL handles some formats OpenCV does not
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as pil_image:
                image = _to_array(pil_image)
        except Exception as exc:
            raise InvalidImageError(
                f"Could not decode image{f' ({name})' if name else ''}: {exc}"
            ) from exc
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise InvalidImageError("Image is too large (more than 40 megapixels)")
    return image


def save_image(path: str | Path, image: np.ndarray, quality: int = 92) -> Path:
    """Write an image to disk as JPEG/PNG, creating parent folders and JPEG quality params."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if path.suffix.lower() in {".jpg", ".jpeg"} else []
    success, buffer = cv2.imencode(path.suffix or ".png", image, params)
    if not success:  # pragma: no cover
        raise InvalidImageError(f"Could not write image to {path}")
    path.write_bytes(buffer.tobytes())
    return path
