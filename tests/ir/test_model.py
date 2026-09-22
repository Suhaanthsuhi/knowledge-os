from __future__ import annotations

import pytest

from app.ir.model import (
    BBox,
    Document,
    Element,
    ElementType,
    ImageRef,
    Page,
    Provenance,
    TableData,
)


def test_bbox_rounds_floats_to_two_places():
    box = BBox(x0=1.23456, y0=2.0, x1=3.98765, y1=4.0)
    assert box.x0 == 1.23
    assert box.x1 == 3.99


def test_bbox_geometry_helpers():
    box = BBox(x0=10, y0=20, x1=40, y1=60)
    assert box.width == 30
    assert box.height == 40
    assert box.contains(BBox(x0=15, y0=25, x1=20, y1=30))
    assert not box.contains(BBox(x0=15, y0=25, x1=200, y1=30))


def test_bbox_horizontal_overlap_ratio():
    a = BBox(x0=0, y0=0, x1=100, y1=10)
    b = BBox(x0=50, y0=20, x1=150, y1=30)
    assert b.horizontal_overlap_ratio(a) == pytest.approx(0.5)


def test_element_provenance_carries_location():
    element = Element(
        id="p1e000",
        type=ElementType.PARAGRAPH,
        page_number=1,
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
        order=0,
        text="hello",
    )
    prov = element.provenance("doc123")
    assert prov == Provenance(
        doc_id="doc123",
        page_number=1,
        element_id="p1e000",
        bbox=BBox(x0=0, y0=0, x1=10, y1=10),
    )


def test_table_data_dimensions_are_validated():
    with pytest.raises(ValueError):
        TableData(headers=["a", "b"], rows=[["1"]], n_rows=1, n_cols=2)


def test_document_defaults_are_empty_not_shared():
    doc_a = Document(id="a", source_name="a.md", mime="text/markdown", checksum="x")
    doc_b = Document(id="b", source_name="b.md", mime="text/markdown", checksum="y")
    doc_a.meta["k"] = 1
    assert doc_b.meta == {}
    assert doc_a.pages == []


def test_document_page_count_and_warning_helper():
    doc = Document(
        id="a",
        source_name="a.pdf",
        mime="application/pdf",
        checksum="x",
        pages=[Page(number=1, width=612, height=792)],
    )
    doc.add_warning("p1e000", "table_fallback", "structure extraction failed")
    assert doc.page_count == 1
    assert doc.meta["warnings"] == [
        {
            "element_id": "p1e000",
            "code": "table_fallback",
            "detail": "structure extraction failed",
        }
    ]


def test_image_ref_allows_missing_blob():
    ref = ImageRef(blob_ref=None, width=None, height=None, format=None)
    assert ref.blob_ref is None
