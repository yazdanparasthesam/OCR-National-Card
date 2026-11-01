"""nidocr — Iranian National Card OCR.

Detects the fields of an Iranian national card (کارت ملی هوشمند) with OpenCV,
reads the Persian text with Tesseract (``fas`` traineddata), validates every
field (national-code checksum, Jalali dates, serial format, ...) and exposes the
result through a FastAPI service and a Streamlit UI.

Typical use::

    from nidocr import NationalCardOCR

    ocr = NationalCardOCR()
    result = ocr.extract("card.jpg")
    print(result.fields["national_code"].value)
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "__version__",
    "NationalCardOCR",
    "Settings",
    "get_settings",
    "CardResult",
    "FieldResult",
]

from nidocr.config import Settings, get_settings


def __getattr__(name: str):  # lazy import so `import nidocr` stays light
    if name == "NationalCardOCR":
        from nidocr.pipeline import NationalCardOCR

        return NationalCardOCR
    if name in {"CardResult", "FieldResult"}:
        from nidocr import schemas

        return getattr(schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
