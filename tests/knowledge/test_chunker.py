from __future__ import annotations

from app.ir.model import BBox, Document, Element, ElementType, Page
from app.ir.tree import assign_parents
from app.knowledge.chunker import MAX_CHUNK_CHARS, chunk_document
from app.parsing.base import InMemoryBlobSink, parse_document


def _element(eid, etype, order, text, *, level=None, page=1):
    return Element(
        id=eid,
        type=etype,
        page_number=page,
        bbox=BBox(x0=0, y0=order * 10, x1=100, y1=order * 10 + 8),
        order=order,
        level=level,
        text=text,
    )


def _doc(elements, *, doc_id="d") -> Document:
    assign_parents(elements)
    return Document(
        id=doc_id,
        source_name="s.md",
        mime="text/markdown",
        checksum="c",
        pages=[Page(number=1, width=612, height=792, elements=elements)],
    )


def test_a_short_section_becomes_one_chunk():
    doc = _doc(
        [
            _element("p1e000", ElementType.HEADING, 0, "Payments", level=1),
            _element("p1e001", ElementType.PARAGRAPH, 1, "The Payment API runs."),
            _element("p1e002", ElementType.PARAGRAPH, 2, "It uses Redis."),
        ]
    )
    chunks = chunk_document(doc)

    assert len(chunks) == 1
    assert chunks[0].heading == "Payments"
    assert chunks[0].element_ids == ("p1e001", "p1e002")
    assert "Payment API" in chunks[0].text
    assert "Redis" in chunks[0].text


def test_separate_sections_become_separate_chunks():
    doc = _doc(
        [
            _element("p1e000", ElementType.HEADING, 0, "Alpha", level=1),
            _element("p1e001", ElementType.PARAGRAPH, 1, "first body"),
            _element("p1e002", ElementType.HEADING, 2, "Beta", level=1),
            _element("p1e003", ElementType.PARAGRAPH, 3, "second body"),
        ]
    )
    chunks = chunk_document(doc)

    assert [chunk.heading for chunk in chunks] == ["Alpha", "Beta"]
    assert chunks[0].element_ids == ("p1e001",)
    assert chunks[1].element_ids == ("p1e003",)


def test_a_long_section_splits_without_breaking_an_element():
    body = [
        _element(f"p1e{index + 1:03d}", ElementType.PARAGRAPH, index + 1, "x" * 600)
        for index in range(5)
    ]
    doc = _doc([_element("p1e000", ElementType.HEADING, 0, "Long", level=1), *body])

    chunks = chunk_document(doc)

    assert len(chunks) > 1
    every_id = [eid for chunk in chunks for eid in chunk.element_ids]
    assert every_id == [element.id for element in body]
    for chunk in chunks:
        assert chunk.heading == "Long"


def test_an_oversized_single_element_becomes_its_own_chunk():
    doc = _doc(
        [
            _element("p1e000", ElementType.HEADING, 0, "Big", level=1),
            _element("p1e001", ElementType.PARAGRAPH, 1, "y" * (MAX_CHUNK_CHARS * 2)),
            _element("p1e002", ElementType.PARAGRAPH, 2, "small"),
        ]
    )
    chunks = chunk_document(doc)

    assert chunks[0].element_ids == ("p1e001",)
    assert chunks[1].element_ids == ("p1e002",)


def test_margin_furniture_is_excluded():
    doc = _doc(
        [
            _element("p1e000", ElementType.HEADER, 0, "Company Confidential"),
            _element("p1e001", ElementType.HEADING, 1, "Body", level=1),
            _element("p1e002", ElementType.PARAGRAPH, 2, "real content"),
            _element("p1e003", ElementType.FOOTER, 3, "page footer"),
            _element("p1e004", ElementType.PAGE_NUMBER, 4, "7"),
        ]
    )
    chunks = chunk_document(doc)

    every_id = [eid for chunk in chunks for eid in chunk.element_ids]
    assert every_id == ["p1e002"]


def test_table_text_is_chunked_as_text():
    from app.ir.model import TableData

    table = _element("p1e001", ElementType.TABLE, 1, None)
    table.table = TableData(
        headers=["Service", "Owner"],
        rows=[["Payment API", "Payments Team"]],
        n_rows=1,
        n_cols=2,
    )
    doc = _doc([_element("p1e000", ElementType.HEADING, 0, "Owners", level=1), table])

    chunks = chunk_document(doc)
    assert "Payments Team" in chunks[0].text


def test_content_before_any_heading_still_chunks():
    doc = _doc(
        [
            _element("p1e000", ElementType.PARAGRAPH, 0, "preamble text"),
            _element("p1e001", ElementType.HEADING, 1, "Later", level=1),
            _element("p1e002", ElementType.PARAGRAPH, 2, "body"),
        ]
    )
    chunks = chunk_document(doc)

    assert chunks[0].heading is None
    assert chunks[0].element_ids == ("p1e000",)


def test_chunk_ids_are_stable_and_document_scoped():
    doc = _doc([_element("p1e000", ElementType.PARAGRAPH, 0, "body")], doc_id="abc")
    assert chunk_document(doc)[0].id == "abc:c000"


def test_an_empty_document_yields_no_chunks():
    assert chunk_document(_doc([])) == []


def test_the_project_asset_chunks_with_every_paragraph_covered():
    doc = parse_document("app/assets/payment_system.md", blobs=InMemoryBlobSink())
    chunks = chunk_document(doc)

    covered = {eid for chunk in chunks for eid in chunk.element_ids}
    expected = {
        element.id
        for element in doc.iter_elements()
        if element.type is ElementType.PARAGRAPH
    }
    assert expected <= covered
