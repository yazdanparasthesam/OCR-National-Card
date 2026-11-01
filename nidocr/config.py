"""Runtime configuration (12-factor style, pydantic-settings)."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CANDIDATE_TESSERACT_BINARIES = (
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
    str(Path.home() / ".local/ocr/bin/tesseract"),
    str(PROJECT_ROOT / "bin/tesseract"),
)


class Settings(BaseSettings):
    """All tunables of the OCR stack.

    Every field can be overridden with an environment variable prefixed with
    ``NIDOCR_`` (see ``.env.example``).
    """

    model_config = SettingsConfigDict(
        env_prefix="NIDOCR_",
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # ── OCR engine ────────────────────────────────────────────────────
    tesseract_cmd: str | None = None
    tessdata_dir: Path | None = None
    languages: str = "fas"
    oem: int = 1  # 1 = LSTM only
    passes: int = 3  # preprocessing variants OCR'd per field
    upscale: float = 3.0
    min_confidence: float = 45.0
    use_label_anchors: bool = True
    anchor_min_similarity: float = 0.66
    cache_ocr: bool = True

    # ── Vision / layout ───────────────────────────────────────────────
    canonical_width: int = 1012
    canonical_height: int = 638
    card_aspect: float = 1.5857  # 85.6mm / 53.98mm
    max_image_side: int = 2400
    template_dir: Path = PROJECT_ROOT / "nidocr" / "templates"
    default_template: str = "national_card_front"
    use_name_lexicon: bool = True
    lexicon_dir: Path | None = None

    # ── API ───────────────────────────────────────────────────────────
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = 12
    log_level: str = "INFO"
    api_url: str = "http://127.0.0.1:8000"

    # ── validators ────────────────────────────────────────────────────
    birth_year_min: int = 1200
    birth_year_max: int = 1450

    # ── helpers ───────────────────────────────────────────────────────
    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                return value
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("languages")
    @classmethod
    def _normalise_languages(cls, value: str) -> str:
        langs = [item.strip() for item in value.split(",") if item.strip()]
        return "+".join(langs) if langs else "fas"

    @field_validator("lexicon_dir", mode="before")
    @classmethod
    def _lexicon_dir(cls, value):
        if value in ("", None):
            return None
        return value

    @field_validator("tessdata_dir", mode="before")
    @classmethod
    def _empty_to_none(cls, value):
        if value in ("", None):
            return None
        return value

    def resolved_tessdata_dir(self) -> Path | None:
        """Folder holding ``*.traineddata`` files, or ``None`` when not configured."""
        if self.tessdata_dir and Path(self.tessdata_dir).is_dir():
            return Path(self.tessdata_dir)
        bundled = PROJECT_ROOT / "models" / "tessdata"
        if bundled.is_dir() and any(bundled.glob("*.traineddata")):
            return bundled
        env = os.environ.get("TESSDATA_PREFIX")
        if env:
            candidate = Path(env)
            if (candidate / "fas.traineddata").is_file():
                return candidate
            if (candidate / "tessdata" / "fas.traineddata").is_file():
                return candidate / "tessdata"
        return None

    def resolved_tesseract_cmd(self) -> str | None:
        """Locate the tesseract executable, or ``None`` when it is not installed."""
        if self.tesseract_cmd and Path(self.tesseract_cmd).exists():
            return str(self.tesseract_cmd)
        env_cmd = os.environ.get("TESSERACT_CMD")
        if env_cmd and Path(env_cmd).exists():
            return env_cmd
        found = shutil.which("tesseract")
        if found:
            return found
        for candidate in _CANDIDATE_TESSERACT_BINARIES:
            if Path(candidate).exists():
                return candidate
        return None

    def engine_ready(self) -> tuple[bool, str]:
        """``(ready, message)`` describing whether OCR can run."""
        cmd = self.resolved_tesseract_cmd()
        if not cmd:
            return False, (
                "Tesseract binary not found. Install it (`apt-get install tesseract-ocr "
                "tesseract-ocr-fas`) or set NIDOCR_TESSERACT_CMD."
            )
        tessdata = self.resolved_tessdata_dir()
        if tessdata is None:
            return False, (
                "Persian traineddata not found. Run `python scripts/fetch_models.py` or set "
                "NIDOCR_TESSDATA_DIR to a folder containing fas.traineddata."
            )
        return True, f"tesseract={cmd} tessdata={tessdata}"

    def template_path(self, name: str | None = None) -> Path:
        name = name or self.default_template
        path = Path(name)
        if path.suffix != ".json":
            path = Path(self.template_dir) / f"{name}.json"
            if not path.is_file():
                path = Path(self.template_dir) / f"{name.replace('-', '_')}.json"
        if not path.is_file() and path.name == "national_card_front.json":
            fallback = Path(name)
            if fallback.is_file():
                return fallback
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests)."""
    get_settings.cache_clear()
