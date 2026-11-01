"""Render plain text into a simple text-based PDF (for samples, tests and evaluation)."""

from __future__ import annotations

import textwrap

import pymupdf

_PAGE_W, _PAGE_H = 595, 842  # A4 in points
_MARGIN = 50
_FONT_SIZE = 10
_LINE_HEIGHT = 14
_WRAP = 100


def render_text_pdf(text: str, title: str | None = None) -> bytes:
    doc = pymupdf.open()
    if title:
        doc.set_metadata({"title": title})
    lines: list[str] = []
    for raw in text.splitlines():
        lines.extend(textwrap.wrap(raw, _WRAP) or [""])

    page = None
    y = 0.0
    for line in lines:
        if page is None or y > _PAGE_H - _MARGIN:
            page = doc.new_page(width=_PAGE_W, height=_PAGE_H)
            y = _MARGIN
        if line:
            page.insert_text((_MARGIN, y), line, fontsize=_FONT_SIZE, fontname="helv")
        y += _LINE_HEIGHT
    if page is None:
        doc.new_page(width=_PAGE_W, height=_PAGE_H)
    data = doc.tobytes()
    doc.close()
    return data
