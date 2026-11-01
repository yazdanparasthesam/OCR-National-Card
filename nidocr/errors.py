"""Project specific exceptions, mapped to HTTP status codes by the API."""

from __future__ import annotations


class NIDOCRError(Exception):
    """Base class for every error raised by the package."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        """Store the Persian/English message and optional structured details for the API."""
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class EngineUnavailableError(NIDOCRError):
    """Tesseract binary or tessdata missing → HTTP 503 (the service is up, the engine is not)."""
    status_code = 503
    code = "engine_unavailable"


class InvalidImageError(NIDOCRError):
    """The upload could not be decoded as an image → HTTP 422."""
    status_code = 422
    code = "invalid_image"


class CardNotFoundError(NIDOCRError):
    """No card was detected in the image → HTTP 422 (the caller should send a better photo)."""
    status_code = 422
    code = "card_not_found"


class TemplateNotFoundError(NIDOCRError):
    """Unknown template name → HTTP 404, listing the shipped templates."""
    status_code = 404
    code = "template_not_found"


class FieldNotFoundError(NIDOCRError):
    """Unknown field key → HTTP 404 with the list of valid keys."""
    status_code = 404
    code = "field_not_found"


class OCRError(NIDOCRError):
    """A generic OCR failure (engine crash, unreadable output) → HTTP 500."""
    status_code = 500
    code = "ocr_failed"
