"""Shared FastAPI dependencies: pipeline access and upload handling."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, File, HTTPException, UploadFile

from nidocr.config import get_settings
from nidocr.errors import InvalidImageError
from nidocr.image_io import load_image
from nidocr.pipeline import NationalCardOCR, get_ocr

ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",
}


def pipeline() -> NationalCardOCR:
    """The process-wide OCR pipeline (Tesseract client is thread-safe by process)."""
    return get_ocr()


Pipeline = Annotated[NationalCardOCR, Depends(pipeline)]


async def read_upload(file: UploadFile) -> bytes:
    """Read an upload, enforcing the configured size limit."""
    settings = get_settings()
    limit = int(settings.max_upload_mb) * 1024 * 1024
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=422, detail="فایل خالی است (empty upload)")
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"حجم فایل بیشتر از {settings.max_upload_mb} مگابایت است",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"نوع فایل پشتیبانی نمی‌شود: {file.content_type}",
        )
    return payload


def decode(payload: bytes):
    """Decode uploaded bytes into a BGR/gray numpy image."""
    try:
        return load_image(payload)
    except InvalidImageError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise InvalidImageError(f"تصویر قابل خواندن نیست: {exc}") from exc


UploadFileDep = Annotated[UploadFile, File(description="JPEG/PNG/WebP/BMP/TIFF card photo")]

__all__ = ["Pipeline", "UploadFileDep", "decode", "pipeline", "read_upload"]
