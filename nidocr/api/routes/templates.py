"""Template discovery."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from nidocr.errors import TemplateNotFoundError
from nidocr.schemas import TemplateInfo
from nidocr.vision.layout import available_templates, load_template

router = APIRouter(prefix="/templates", tags=["templates"])


def _info(name: str) -> TemplateInfo:
    """Load one template and project it to the public :class:`TemplateInfo` model."""
    try:
        layout = load_template(name)
    except TemplateNotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.message) from exc
    return TemplateInfo(
        name=layout.name,
        title_fa=layout.title_fa,
        title_en=layout.title_en,
        canvas=layout.canvas,
        fields=list(layout.field_layout),
    )


@router.get("", response_model=list[TemplateInfo], summary="قالب‌ها / templates")
def list_templates() -> list[TemplateInfo]:
    """Return every shipped template with its canvas size and field list."""
    return [_info(name) for name in available_templates()]


@router.get(
    "/{name}",
    response_model=TemplateInfo,
    summary="یک قالب / one template",
    responses={404: {"description": "template not found"}},
)
def template(name: str) -> TemplateInfo:
    """Return a single template by name (404 when it does not exist)."""
    return _info(name)
