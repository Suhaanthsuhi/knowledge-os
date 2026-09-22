from __future__ import annotations

from collections import Counter

import fitz
import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, parse_document
from app.storage.base import serialize_document
from app.storage.filestore import FileDocumentStore

SOURCES = ["simple", "two_column", "table_figure", "diagram_png", "sheet"]


@pytest.fixture(params=SOURCES)
def parsed(request, fixtures):
    path = fixtures[request.param]
    sink = InMemoryBlobSink()
    return parse_document(path, blobs=sink), sink, path


def _non_whitespace(text: str) -> Counter:
    return Counter(character for character in text if not character.isspace())


def test_pdf_text_is_lossless(fixtures):
    """Every non-whitespace character PyMuPDF reports survives into some element."""
    for name in ("simple", "two_column", "table_figure"):
        doc = parse_document(fixtures[name], blobs=InMemoryBlobSink())
        with fitz.open(str(fixtures[name])) as pdf:
            for page_number, pdf_page in enumerate(pdf, start=1):
                expected = _non_whitespace(pdf_page.get_text())
                page = next(p for p in doc.pages if p.number == page_number)
                actual = _non_whitespace(
                    " ".join(element.text_content() for element in page.elements)
                )
                missing = expected - actual
                assert not missing, f"{name} page {page_number} lost {dict(missing)}"


def test_parsing_is_deterministic(parsed):
    doc, _, path = parsed
    again = parse_document(path, blobs=InMemoryBlobSink())
    assert serialize_document(doc) == serialize_document(again)


def test_element_ids_are_unique(parsed):
    doc, _, _path = parsed
    ids = [element.id for element in doc.iter_elements()]
    assert len(ids) == len(set(ids))


def test_order_is_strictly_increasing_and_contiguous(parsed):
    doc, _, _path = parsed
    orders = [element.order for element in doc.iter_elements()]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)


def test_every_reference_resolves(parsed):
    doc, _, _path = parsed
    index = {element.id: element for element in doc.iter_elements()}
    for element in index.values():
        if element.parent_id is not None:
            assert element.parent_id in index
            assert index[element.parent_id].type is ElementType.HEADING
        if element.image is not None and element.image.caption_id is not None:
            assert element.image.caption_id in index
        if "captions" in element.attrs:
            assert element.attrs["captions"] in index


def test_every_bbox_lies_within_its_page(parsed):
    doc, _, _path = parsed
    tolerance = 1.0
    for page in doc.pages:
        for element in page.elements:
            assert element.bbox.x0 >= -tolerance
            assert element.bbox.y0 >= -tolerance
            assert element.bbox.x1 <= page.width + tolerance
            assert element.bbox.y1 <= page.height + tolerance


def test_table_text_is_not_double_counted(parsed):
    doc, _, _path = parsed
    tables = [e for e in doc.iter_elements() if e.type is ElementType.TABLE]
    others = " ".join(
        e.text or "" for e in doc.iter_elements() if e.type is not ElementType.TABLE
    )
    for table in tables:
        for row in table.table.rows:
            for cell in row:
                if len(cell) > 6:
                    assert cell not in others, f"{cell!r} appears both in and out of a table"


def test_every_element_yields_provenance(parsed):
    doc, _, _path = parsed
    for element in doc.iter_elements():
        provenance = element.provenance(doc.id)
        assert provenance.doc_id == doc.id
        assert provenance.element_id == element.id
        assert provenance.page_number == element.page_number


def test_every_referenced_blob_was_emitted(parsed):
    doc, sink, _path = parsed
    for element in doc.iter_elements():
        if element.image is not None and element.image.blob_ref is not None:
            assert element.image.blob_ref in sink.blobs


def test_documents_round_trip_through_the_store(parsed, tmp_path):
    doc, sink, _path = parsed
    store = FileDocumentStore(tmp_path)
    store.put(doc, sink.blobs)
    assert store.get(doc.id) == doc


def test_the_project_asset_round_trips(tmp_path):
    sink = InMemoryBlobSink()
    doc = parse_document("app/assets/payment_system.md", blobs=sink)
    store = FileDocumentStore(tmp_path)
    store.put(doc, sink.blobs)

    assert store.get(doc.id) == doc
    assert doc.title == "Payment System"
