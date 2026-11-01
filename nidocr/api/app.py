"""FastAPI service for the Iranian national card OCR.

The API is deliberately split per field: every field of the card has its own
route (``POST /api/v1/fields/national-code`` …), and ``POST /api/v1/card`` reads
all of them at once.  Nothing here knows about OCR internals — the routes only
translate HTTP to :class:`nidocr.pipeline.NationalCardOCR`.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nidocr import __version__
from nidocr.api.routes import card, fields, health, templates, validate
from nidocr.config import get_settings
from nidocr.errors import NIDOCRError


def create_app() -> FastAPI:
    """Build the FastAPI application (routers, CORS, error mapping, landing page).
    Factory form so tests and ``uvicorn --reload`` can create isolated apps; every
    router is mounted under ``settings.api_prefix`` (default ``/api/v1``).
    """
    settings = get_settings()
    app = FastAPI(
        title="NIDOCR — Iranian National Card OCR",
        description=(
            "Persian (Farsi) OCR for the Iranian national card (**کارت ملی هوشمند**).\n\n"
            "OpenCV locates and rectifies the card, then every printed field is read "
            "with a Persian OCR engine and validated (mod-11 checksum, Jalali calendar, "
            "name lexicon).\n\n"
            "Every field has its own route under `/api/v1/fields`, and `/api/v1/card` "
            "returns the whole card in a single request."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "NIDOCR", "url": "https://github.com/yazdanparasthesam/OCR-National-Card"},
        license_info={"name": "MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(NIDOCRError)
    async def _handle_nidocr_error(_request, exc: NIDOCRError) -> JSONResponse:  # noqa: ANN001
        """Translate a :class:`NIDOCRError` into the stable JSON error envelope
        ``{detail, code, extra}`` with the status code declared by the exception class.
        """
        return JSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content={"detail": exc.message, "code": getattr(exc, "code", "error"), "extra": exc.detail},
        )

    @app.get("/", include_in_schema=False)
    async def index() -> dict:
        """Landing page: service name, version and a map of the available endpoints.
        Kept out of the OpenAPI schema because ``/docs`` is the real documentation.
        """
        prefix = settings.api_prefix
        return {
            "name": "NIDOCR",
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "endpoints": {
                "health": f"{prefix}/health",
                "card": f"POST {prefix}/card",
                "batch": f"POST {prefix}/card/batch",
                "field": f"POST {prefix}/fields/{{field}}",
                "field_list": f"GET {prefix}/fields",
                "templates": f"GET {prefix}/templates",
                "national_code_check": f"POST {prefix}/validate/national-code",
            },
        }

    for router in (health.router, card.router, fields.router, templates.router, validate.router):
        app.include_router(router, prefix=settings.api_prefix)

    return app
