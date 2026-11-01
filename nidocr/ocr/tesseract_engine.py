"""Tesseract wrapper (``fas`` traineddata) returning text *and* word geometry.

The engine drives the ``tesseract`` executable directly instead of relying on a
higher level binding, for three reasons:

* input images are handed over as **BMP**, which every Leptonica build supports
  (PNG/TIFF support is optional at compile time and frequently missing in slim
  containers or self-built binaries),
* the TSV renderer gives us word boxes with confidences, which the label-anchor
  layout detection needs,
* errors and timeouts are reported precisely (a missing ``fas.traineddata`` is a
  very common setup mistake and deserves a clear message).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from nidocr.config import Settings, get_settings
from nidocr.errors import EngineUnavailableError, OCRError
from nidocr.ocr.text import normalize_persian, sort_tokens_ltr, sort_tokens_rtl

TSV_HEADER = (
    "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
)


@dataclass
class TextToken:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    line_key: tuple[int, int, int]

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height

    @property
    def centre(self) -> tuple[float, float]:
        return self.left + self.width / 2.0, self.top + self.height / 2.0


@dataclass
class OCRLine:
    text_rtl: str
    text_ltr: str
    confidence: float
    bbox: tuple[int, int, int, int]
    tokens: list[TextToken] = field(default_factory=list)


@dataclass
class OCRResult:
    lines: list[OCRLine] = field(default_factory=list)
    confidence: float = 0.0
    variant: str = ""
    psm: int = 7
    lang: str = "fas"
    elapsed_ms: float = 0.0
    empty: bool = False

    @property
    def text(self) -> str:
        return "\n".join(line.text_rtl for line in self.lines if line.text_rtl).strip()

    @property
    def text_ltr(self) -> str:
        return "\n".join(line.text_ltr for line in self.lines if line.text_ltr).strip()

    @property
    def tokens(self) -> list[TextToken]:
        return [token for line in self.lines for token in line.tokens]


def _weighted_confidence(tokens: list[TextToken]) -> float:
    if not tokens:
        return 0.0
    weights = [max(1, len(token.text)) for token in tokens]
    values = [token.confidence for token in tokens]
    return float(sum(w * v for w, v in zip(weights, values, strict=True)) / sum(weights))


def _parse_tsv(payload: str) -> list[OCRLine]:
    """Parse tesseract TSV output into lines of positioned words."""
    grouped: dict[tuple[int, int, int], list[TextToken]] = {}
    rows = [row for row in payload.splitlines() if row.strip()]
    if not rows:
        return []
    if rows[0].startswith("level"):
        rows = rows[1:]
    for row in rows:
        columns = row.split("\t")
        if len(columns) < 12:
            continue
        text = " ".join(columns[11:]).strip()
        if not text:
            continue
        try:
            confidence = float(columns[10])
        except ValueError:
            continue
        if confidence < 0:
            continue
        key = (int(columns[2]), int(columns[3]), int(columns[4]))
        grouped.setdefault(key, []).append(
            TextToken(
                text=normalize_persian(text),
                left=int(columns[6]),
                top=int(columns[7]),
                width=int(columns[8]),
                height=int(columns[9]),
                confidence=confidence,
                line_key=key,
            )
        )

    lines: list[OCRLine] = []
    for _, tokens in sorted(grouped.items()):
        ordered = sorted(tokens, key=lambda token: token.left)
        left = min(token.left for token in ordered)
        top = min(token.top for token in ordered)
        right = max(token.left + token.width for token in ordered)
        bottom = max(token.top + token.height for token in ordered)
        lines.append(
            OCRLine(
                text_rtl=sort_tokens_rtl([(token.text, token.left) for token in ordered]),
                text_ltr=sort_tokens_ltr([(token.text, token.left) for token in ordered]),
                confidence=_weighted_confidence(ordered),
                bbox=(left, top, right, bottom),
                tokens=ordered,
            )
        )
    return lines


class TesseractEngine:
    """Subprocess based Tesseract client with caching."""

    name = "tesseract"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._cache: dict[str, OCRResult] = {}
        self.binary = self.settings.resolved_tesseract_cmd()
        self.tessdata = self.settings.resolved_tessdata_dir()

    # ── setup / introspection ────────────────────────────────────────
    def ready(self) -> tuple[bool, str]:
        return self.settings.engine_ready()

    def ensure_ready(self) -> None:
        ready, message = self.ready()
        if not ready:
            raise EngineUnavailableError(message)

    @property
    def languages(self) -> str:
        return self.settings.languages or "fas"

    @property
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.tessdata:
            env["TESSDATA_PREFIX"] = str(self.tessdata)
        return env

    def available_languages(self) -> list[str]:
        if not self.binary:
            return []
        try:
            completed = subprocess.run(  # noqa: S603 - fixed binary path
                [self.binary, "--list-langs"],
                capture_output=True,
                text=True,
                timeout=20,
                env=self._env,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return []
        languages = []
        for line in (completed.stdout or "").splitlines():
            line = line.strip()
            if line and " " not in line and not line.startswith("List"):
                if line.endswith(".traineddata"):
                    line = line[: -len(".traineddata")]
                languages.append(line)
        return sorted(set(languages))

    def missing_languages(self) -> list[str]:
        available = set(self.available_languages())
        if not available:
            return []
        return [lang for lang in self.languages.split("+") if lang not in available]

    # ── OCR ──────────────────────────────────────────────────────────
    def _config_string(self, psm: int, whitelist: str | None) -> list[str]:
        args = [
            "--oem",
            str(self.settings.oem),
            "--psm",
            str(psm),
            "-c",
            "preserve_interword_spaces=1",
        ]
        if whitelist:
            args += ["-c", f"tessedit_char_whitelist={whitelist}"]
        return args

    def read(
        self,
        image: np.ndarray,
        *,
        psm: int = 7,
        whitelist: str | None = None,
        lang: str | None = None,
        variant: str = "",
        timeout: float = 45.0,
        use_cache: bool | None = None,
    ) -> OCRResult:
        """OCR an image (a single text line by default) and return text + geometry."""
        self.ensure_ready()
        if image is None or getattr(image, "size", 0) == 0:
            return OCRResult(variant=variant, psm=psm, empty=True)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        lang = lang or self.languages
        cache_enabled = self.settings.cache_ocr if use_cache is None else use_cache
        key = ""
        if cache_enabled:
            import hashlib

            digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
            key = f"{digest}|{psm}|{lang}|{whitelist}|{self.settings.oem}|{image.shape}"
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        started = time.perf_counter()
        payload = self._run_cli(image, lang=lang, psm=psm, whitelist=whitelist, timeout=timeout)
        lines = _parse_tsv(payload)
        # only a payload *without* tabs can be plain text; a TSV page with no words
        # must not be turned into "text"
        plain = ""
        if "\t" not in payload:
            plain = payload.strip()
        if not lines and plain:
            # some builds ignore the tsv renderer: treat stdout as one plain line
            text = normalize_persian(plain)
            if text:
                lines = [
                    OCRLine(
                        text_rtl=text,
                        text_ltr=text,
                        confidence=0.0,
                        bbox=(0, 0, image.shape[1], image.shape[0]),
                    )
                ]

        result = OCRResult(
            lines=lines,
            confidence=_weighted_confidence([token for line in lines for token in line.tokens]),
            variant=variant,
            psm=psm,
            lang=lang,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            empty=not lines,
        )
        if cache_enabled:
            if len(self._cache) > 8192:
                self._cache.clear()
            self._cache[key] = result
        return result

    def _run_cli(
        self,
        image: np.ndarray,
        *,
        lang: str,
        psm: int,
        whitelist: str | None,
        timeout: float,
    ) -> str:
        if not self.binary:
            raise EngineUnavailableError(
                "Tesseract binary not found. Install it or set NIDOCR_TESSERACT_CMD."
            )
        success, buffer = cv2.imencode(".bmp", image)
        if not success:  # pragma: no cover - defensive
            raise OCRError("Could not encode the ROI for OCR")
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as handle:
                handle.write(buffer.tobytes())
                temporary_path = handle.name
            command = [
                self.binary,
                temporary_path,
                "stdout",
                "-l",
                lang,
                *self._config_string(psm, whitelist),
                "tsv",
            ]
            completed = subprocess.run(  # noqa: S603 - fixed binary path
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OCRError(f"Tesseract timed out after {timeout:.0f}s") from exc
        except OSError as exc:  # pragma: no cover - depends on environment
            raise EngineUnavailableError(f"Could not execute tesseract: {exc}") from exc
        finally:
            if temporary_path:
                Path(temporary_path).unlink(missing_ok=True)

        if completed.returncode != 0:
            message = (completed.stderr or "").strip().splitlines()
            detail = message[-1] if message else "unknown error"
            if "tessdata" in detail.lower() or "failed loading language" in detail.lower():
                raise EngineUnavailableError(
                    f"Tesseract could not load language '{lang}': {detail}. "
                    "Run `python scripts/fetch_models.py` or set NIDOCR_TESSDATA_DIR."
                )
            raise OCRError(f"Tesseract failed: {detail}")
        return completed.stdout or ""

    def read_page(self, image: np.ndarray, *, psm: int = 6, variant: str = "") -> OCRResult:
        """OCR a whole card (used to locate the printed field labels)."""
        return self.read(image, psm=psm, whitelist=None, variant=variant)

    def version(self) -> str:
        if not self.binary:
            return "unavailable"
        try:
            completed = subprocess.run(  # noqa: S603
                [self.binary, "--version"], capture_output=True, text=True, timeout=15, check=False
            )
            return (completed.stdout or "").splitlines()[0].strip()
        except Exception:  # pragma: no cover
            return "unknown"

    def clear_cache(self) -> None:
        self._cache.clear()


_engine: TesseractEngine | None = None


def get_engine(settings: Settings | None = None) -> TesseractEngine:
    """Process-wide engine singleton."""
    global _engine
    if _engine is None or settings is not None:
        _engine = TesseractEngine(settings)
    return _engine


def find_tesseract_binary() -> str | None:
    """Best-effort lookup used by the CLI/UI diagnostics."""
    settings = get_settings()
    return settings.resolved_tesseract_cmd() or shutil.which("tesseract")
