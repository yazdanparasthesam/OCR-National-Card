"""ASGI entry point: ``uvicorn nidocr.api.main:app``."""

from __future__ import annotations

from nidocr.api.app import create_app

app = create_app()

__all__ = ["app"]
