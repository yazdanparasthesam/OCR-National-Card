"""Synthetic Iranian national cards (renderer + sample data) for demos and tests."""

from __future__ import annotations

from nidocr.synthetic.renderer import (
    DIFFICULTIES,
    CardSpec,
    FontSet,
    degrade,
    random_spec,
    render_card,
    save_dataset,
    shape,
)

__all__ = [
    "DIFFICULTIES",
    "CardSpec",
    "FontSet",
    "degrade",
    "random_spec",
    "render_card",
    "save_dataset",
    "shape",
]
