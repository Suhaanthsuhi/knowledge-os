from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, ParseError
from app.parsing.pdf import PdfParser


@pytest.fixture(scope="module")
def simple(fixtures):
    sink = InMemoryBlobSink()
    return PdfParser().parse(fixtures["simple"], blobs=sink), sink


@pytest.fixture(scope="module")
def two_column(fixtures):
    sink = InMemoryBlobSink()
    return PdfParser().parse(fixtures["two_column"], blobs=sink), sink


@pytest.fixture(scope="module")
def table_figure(fixtures):
    sink = InMemoryBlobSink()
    return PdfParser().parse(fixtures["table_figure"], blobs=sink), sink


def _texts(doc, element_type):
    return [e.text for e in doc.iter_elements() if e.type is element_type]


def test_simple_pdf_has_one_page_and_correct_metadata(simple):
    doc, _ = simple
    assert doc.page_count == 1
    assert doc.mime == "application/pdf"
    assert doc.meta["parser"] == "pdf"
    assert doc.meta["page_count"] == 1


def test_font_size_drives_heading_levels(simple):
    doc, _ = simple
    headings = [
        (e.text, e.level) for e in doc.iter_elements() if e.type is ElementType.HEADING
    ]
    assert ("Payment Architecture", 1) in headings
    assert ("Dependencies", 2) in headings


def test_title_is_the_first_level_one_heading(simple):
    doc, _ = simple
    assert doc.title == "Payment Architecture"


def test_body_text_is_captured_as_paragraphs(simple):
    doc, _ = simple
    body = " ".join(_texts(doc, ElementType.PARAGRAPH))
    assert "Payment API is responsible for processing payment requests" in body
    assert "depends on Redis for caching" in body


def test_paragraphs_are_parented_to_the_preceding_heading(simple):
    doc, _ = simple
    elements = list(doc.iter_elements())
    index = {e.id: e for e in elements}
    body = [
        e
        for e in elements
        if e.type is ElementType.PARAGRAPH and "Payment API is responsible" in (e.text or "")
    ]
    assert body
    parent = index[body[0].parent_id]
    assert parent.text == "Payment Architecture"


def test_two_column_pages_read_down_each_column(two_column):
    doc, _ = two_column
    page_one = [e for e in doc.pages[0].elements if e.type is ElementType.PARAGRAPH]
    ordered = sorted(page_one, key=lambda e: e.order)

    left = [e for e in ordered if e.bbox.x0 < 300]
    right = [e for e in ordered if e.bbox.x0 >= 300]
    assert left and right
    assert max(e.order for e in left) < min(e.order for e in right)


def test_running_header_is_typed_as_header(two_column):
    doc, _ = two_column
    headers = _texts(doc, ElementType.HEADER)
    assert headers.count("Payment Platform Handbook") == 2


def test_page_numbers_are_typed_as_page_numbers(two_column):
    doc, _ = two_column
    assert sorted(_texts(doc, ElementType.PAGE_NUMBER)) == ["1", "2"]


def test_table_structure_is_extracted(table_figure):
    doc, _ = table_figure
    tables = [e for e in doc.iter_elements() if e.type is ElementType.TABLE]
    assert len(tables) == 1

    table = tables[0].table
    assert table.headers == ["Service", "Datastore", "Owner"]
    assert ["Payment API", "PostgreSQL", "Payments Team"] in table.rows
    assert table.n_cols == 3
    assert table.n_rows == 3


def test_table_text_is_not_also_emitted_as_paragraphs(table_figure):
    doc, _ = table_figure
    paragraphs = " ".join(_texts(doc, ElementType.PARAGRAPH))
    assert "PostgreSQL" not in paragraphs


def test_image_is_extracted_to_a_blob(table_figure):
    doc, sink = table_figure
    images = [e for e in doc.iter_elements() if e.type is ElementType.IMAGE]
    assert len(images) == 1

    ref = images[0].image.blob_ref
    assert ref in sink.blobs
    assert len(sink.blobs[ref]) > 0
    assert images[0].image.width and images[0].image.height


def test_figure_caption_is_paired_both_ways(table_figure):
    doc, _ = table_figure
    images = [e for e in doc.iter_elements() if e.type is ElementType.IMAGE]
    captions = [e for e in doc.iter_elements() if e.type is ElementType.CAPTION]

    assert len(captions) == 1
    assert captions[0].text.startswith("Figure 1:")
    assert images[0].image.caption_id == captions[0].id
    assert captions[0].attrs["captions"] == images[0].id


def test_every_bbox_sits_inside_its_page(table_figure):
    doc, _ = table_figure
    for page in doc.pages:
        for element in page.elements:
            assert 0 <= element.bbox.x0 <= page.width + 1
            assert 0 <= element.bbox.y0 <= page.height + 1
            assert element.bbox.x1 <= page.width + 1
            assert element.bbox.y1 <= page.height + 1


def test_element_ids_are_unique_and_order_is_strictly_increasing(two_column):
    doc, _ = two_column
    elements = list(doc.iter_elements())
    assert len({e.id for e in elements}) == len(elements)
    orders = [e.order for e in elements]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_parsing_is_deterministic(fixtures):
    from app.storage.base import serialize_document

    first = PdfParser().parse(fixtures["table_figure"], blobs=InMemoryBlobSink())
    second = PdfParser().parse(fixtures["table_figure"], blobs=InMemoryBlobSink())
    assert serialize_document(first) == serialize_document(second)


def test_a_non_pdf_file_raises_parse_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.7 truncated garbage")
    with pytest.raises(ParseError):
        PdfParser().parse(path, blobs=InMemoryBlobSink())
