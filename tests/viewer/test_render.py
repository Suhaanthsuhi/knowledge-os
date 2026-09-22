from __future__ import annotations

import pytest

from app.ir.model import BBox, Document, Element, ElementType, Page
from app.parsing.base import InMemoryBlobSink, parse_document
from app.viewer.render import (
    RenderOptions,
    is_synthetic,
    page_of,
    render_page,
    scale_for,
    scaled,
)
from app.viewer.theme import COLORS, TYPE_ORDER, color_for, rgb, rgba


def _doc() -> Document:
    return Document(
        id="d",
        source_name="s.md",
        mime="text/markdown",
        checksum="c",
        meta={"parser": "markdown"},
        pages=[
            Page(
                number=1,
                width=300,
                height=200,
                elements=[
                    Element(
                        id="p1e000",
                        type=ElementType.HEADING,
                        page_number=1,
                        bbox=BBox(x0=10, y0=10, x1=200, y1=30),
                        order=0,
                        text="Title",
                        attrs={"synthetic_bbox": True},
                    )
                ],
            )
        ],
    )


def test_scale_converts_points_to_pixels():
    assert scale_for(72) == 1.0
    assert scale_for(144) == 2.0


def test_scaled_multiplies_every_edge():
    assert scaled(BBox(x0=1, y0=2, x1=3, y1=4), 2.0) == (2.0, 4.0, 6.0, 8.0)


def test_page_of_finds_by_page_number_not_index():
    doc = _doc()
    doc.pages[0].number = 7
    assert page_of(doc, 7).number == 7
    with pytest.raises(KeyError):
        page_of(doc, 1)


def test_is_synthetic_tracks_the_parser():
    assert is_synthetic(_doc())
    pdf = _doc()
    pdf.meta["parser"] = "pdf"
    assert not is_synthetic(pdf)


def test_every_element_type_has_a_distinct_colour():
    assert set(COLORS) == set(TYPE_ORDER)
    body_colours = {
        color_for(t)
        for t in TYPE_ORDER
        if t not in {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_NUMBER}
    }
    assert len(body_colours) == 7  # margin furniture deliberately shares one grey


def test_colour_helpers_parse_hex():
    assert rgb("#4F46E5") == (79, 70, 229)
    assert rgba("#4F46E5", 128) == (79, 70, 229, 128)


def test_synthetic_page_is_rendered_at_the_scaled_page_size():
    image = render_page(_doc(), 1, RenderOptions(dpi=144))
    assert image.size == (600, 400)


def test_render_is_pure_and_repeatable():
    first = render_page(_doc(), 1, RenderOptions(dpi=96))
    second = render_page(_doc(), 1, RenderOptions(dpi=96))
    assert first.tobytes() == second.tobytes()


def test_hiding_every_type_leaves_the_page_untouched():
    plain = render_page(_doc(), 1, RenderOptions(visible_types=frozenset()))
    boxed = render_page(_doc(), 1, RenderOptions())
    assert plain.tobytes() != boxed.tobytes()


def test_selection_changes_the_rendering():
    unselected = render_page(_doc(), 1, RenderOptions())
    selected = render_page(_doc(), 1, RenderOptions(selected_id="p1e000"))
    assert unselected.tobytes() != selected.tobytes()


@pytest.mark.parametrize(
    "key", ["simple", "two_column", "table_figure", "diagram_png", "sheet"]
)
def test_every_fixture_renders_every_page(key, fixtures):
    path = fixtures[key]
    sink = InMemoryBlobSink()
    doc = parse_document(path, blobs=sink)

    for page in doc.pages:
        image = render_page(
            doc,
            page.number,
            RenderOptions(dpi=96),
            source=path.read_bytes(),
            blobs=sink.blobs,
        )
        assert image.size[0] > 0 and image.size[1] > 0


def test_a_pdf_page_renders_at_the_requested_dpi(fixtures):
    path = fixtures["simple"]
    sink = InMemoryBlobSink()
    doc = parse_document(path, blobs=sink)

    low = render_page(doc, 1, RenderOptions(dpi=72), source=path.read_bytes(), blobs=sink.blobs)
    high = render_page(doc, 1, RenderOptions(dpi=144), source=path.read_bytes(), blobs=sink.blobs)

    assert high.size[0] == pytest.approx(low.size[0] * 2, abs=2)
