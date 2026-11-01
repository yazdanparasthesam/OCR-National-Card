"""Project specific exceptions, mapped to HTTP status codes by the API."""

from __future__ import annotations


class NIDOCRError(Exception):
    """Base class for every error raised by the package."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class EngineUnavailableError(NIDOCRError):
    status_code = 503
    code = "engine_unavailable"


class InvalidImageError(NIDOCRError):
    status_code = 422
    code = "invalid_image"


class CardNotFoundError(NIDOCRError):
    status_code = 422
    code = "card_not_found"


class TemplateNotFoundError(NIDOCRError):
    status_code = 404
    code = "template_not_found"


class FieldNotFoundError(NIDOCRError):
    status_code = 404
    code = "field_not_found"


class OCRError(NIDOCRError):
    status_code = 500
    code = "ocr_failed"
