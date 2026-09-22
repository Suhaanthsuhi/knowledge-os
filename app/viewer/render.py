"""Turn a parsed page into an annotated image.

Pure functions over the IR: no Streamlit, no globals, no disk access beyond the
bytes handed in. That is what makes the interesting part of the viewer testable
rather than merely clickable.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Mapping

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

from app.ir.model import BBox, Document, Element, ElementType, Page
from app.viewer.theme import SYNTHETIC_BG, color_for, rgb, rgba

__all__ = [
    "RenderOptions",
    "scale_for",
    "scaled",
    "render_page",
    "page_of",
    "is_synthetic",
]

PDF_MIME = "application/pdf"
BADGE_RADIUS = 11
SELECTED_ALPHA = 46
BOX_ALPHA = 235


@dataclass(frozen=True)
class RenderOptions:
    dpi: int = 144
    show_boxes: bool = True
    show_order: bool = True
    show_connectors: bool = True
    show_synthetic_text: bool = True
    visible_types: frozenset[ElementType] | None = None
    selected_id: str | None = None

    def shows(self, element: Element) -> bool:
        return self.visible_types is None or element.type in self.visible_types


def scale_for(dpi: int) -> float:
    """Points to pixels. PDF user space is 72 points to the inch."""
    return dpi / 72.0


def scaled(box: BBox, scale: float) -> tuple[float, float, float, float]:
    return (box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale)


def page_of(doc: Document, page_number: int) -> Page:
    for page in doc.pages:
        if page.number == page_number:
            return page
    raise KeyError(f"document {doc.id} has no page {page_number}")


def is_synthetic(doc: Document) -> bool:
    """True when the parser invented the geometry rather than measuring it."""
    return doc.meta.get("parser") in {"markdown", "spreadsheet"}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _base_image(
    doc: Document,
    page: Page,
    options: RenderOptions,
    source: bytes | None,
    blobs: Mapping[str, bytes],
) -> Image.Image:
    scale = scale_for(options.dpi)
    size = (
        max(1, round(page.width * scale)),
        max(1, round(page.height * scale)),
    )

    if doc.mime == PDF_MIME and source:
        with fitz.open(stream=source, filetype="pdf") as pdf:
            pixmap = pdf[page.number - 1].get_pixmap(dpi=options.dpi)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return image

    if doc.mime.startswith("image/"):
        first = page.elements[0] if page.elements else None
        ref = first.image.blob_ref if first and first.image else None
        if ref and ref in blobs:
            return Image.open(io.BytesIO(blobs[ref])).convert("RGB").resize(size)

    # Synthetic geometry, or a source we cannot re-render: draw the layout the
    # parser invented so that invisible coordinates become visible ones.
    return Image.new("RGB", size, SYNTHETIC_BG)


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: tuple[int, int, int, int],
    *,
    dash: int = 7,
    gap: int = 5,
    width: int = 2,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    step_x, step_y = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        segment = min(dash, length - position)
        draw.line(
            [
                (x0 + step_x * position, y0 + step_y * position),
                (x0 + step_x * (position + segment), y0 + step_y * (position + segment)),
            ],
            fill=colour,
            width=width,
        )
        position += dash + gap


def _nearest_edges(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Endpoints joining the facing edges of two boxes, not their centres.

    A centre-to-centre line would cut straight through a diagram it is meant to
    annotate.
    """
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    a_mid_x, b_mid_x = (ax0 + ax1) / 2, (bx0 + bx1) / 2

    if by0 >= ay1:  # b sits below a
        return (a_mid_x, ay1), (b_mid_x, by0)
    if ay0 >= by1:  # b sits above a
        return (a_mid_x, ay0), (b_mid_x, by1)

    a_mid_y, b_mid_y = (ay0 + ay1) / 2, (by0 + by1) / 2
    if bx0 >= ax1:
        return (ax1, a_mid_y), (bx0, b_mid_y)
    return (ax0, a_mid_y), (bx1, b_mid_y)


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        while len(raw) > width:
            cut = raw.rfind(" ", 0, width)
            cut = cut if cut > 0 else width
            lines.append(raw[:cut])
            raw = raw[cut:].lstrip()
        lines.append(raw)
    return lines


def _draw_synthetic_text(
    draw: ImageDraw.ImageDraw,
    element: Element,
    box: tuple[float, float, float, float],
    colour: str,
) -> None:
    content = element.text_content().strip()
    if not content:
        return
    x0, y0, x1, y1 = box
    font = _font(12)
    char_width = max(1, int((x1 - x0) // 7))
    available = max(1, int((y1 - y0) // 15))
    lines = _wrap(content, char_width)[:available]
    for index, line in enumerate(lines):
        draw.text((x0 + 8, y0 + 6 + index * 15), line, font=font, fill=rgb(colour))


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    label: str,
    colour: str,
) -> None:
    x, y = position
    draw.ellipse(
        [x - BADGE_RADIUS, y - BADGE_RADIUS, x + BADGE_RADIUS, y + BADGE_RADIUS],
        fill=rgba(colour, 255),
    )
    font = _font(12)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    draw.text(
        (x - (right - left) / 2, y - (bottom - top) / 2 - 1),
        label,
        font=font,
        fill=(255, 255, 255),
    )


def render_page(
    doc: Document,
    page_number: int,
    options: RenderOptions | None = None,
    *,
    source: bytes | None = None,
    blobs: Mapping[str, bytes] | None = None,
) -> Image.Image:
    """Render one page with its element boxes, order badges and caption links."""
    options = options or RenderOptions()
    blobs = blobs or {}
    page = page_of(doc, page_number)
    base = _base_image(doc, page, options, source, blobs)

    scale = scale_for(options.dpi)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    visible = [e for e in page.elements if options.shows(e)]
    by_id = {e.id: e for e in page.elements}
    synthetic = is_synthetic(doc)

    if options.show_connectors:
        for element in visible:
            partner_id = (
                element.image.caption_id
                if element.image is not None
                else element.attrs.get("caption_id")
            )
            partner = by_id.get(partner_id) if partner_id else None
            if partner is None:
                continue
            _dashed_line(
                draw,
                *_nearest_edges(
                    scaled(element.bbox, scale), scaled(partner.bbox, scale)
                ),
                rgba(color_for(ElementType.CAPTION), 210),
            )

    if options.show_boxes:
        for element in visible:
            colour = color_for(element.type)
            box = scaled(element.bbox, scale)
            selected = element.id == options.selected_id
            dashed_border = bool(element.attrs.get("synthetic_bbox"))

            if selected:
                draw.rectangle(box, fill=rgba(colour, SELECTED_ALPHA))

            if dashed_border:
                x0, y0, x1, y1 = box
                edges = [
                    ((x0, y0), (x1, y0)),
                    ((x1, y0), (x1, y1)),
                    ((x1, y1), (x0, y1)),
                    ((x0, y1), (x0, y0)),
                ]
                for start, end in edges:
                    _dashed_line(
                        draw, start, end, rgba(colour, BOX_ALPHA),
                        dash=6, gap=4, width=3 if selected else 2,
                    )
            else:
                draw.rectangle(
                    box, outline=rgba(colour, BOX_ALPHA), width=4 if selected else 2
                )

            if synthetic and options.show_synthetic_text:
                _draw_synthetic_text(draw, element, box, colour)

    if options.show_order:
        for element in visible:
            x0, y0, _x1, _y1 = scaled(element.bbox, scale)
            # Sit the badge just outside the corner so it does not cover the
            # very text you are trying to read, clamped to stay on the canvas.
            centre = (
                max(BADGE_RADIUS, x0 - BADGE_RADIUS * 0.45),
                max(BADGE_RADIUS, y0 - BADGE_RADIUS * 0.45),
            )
            _draw_badge(draw, centre, str(element.order), color_for(element.type))

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
