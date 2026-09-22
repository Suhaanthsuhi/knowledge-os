from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, ParseError
from app.parsing.markdown import MarkdownParser


def _parse(tmp_path, text: str):
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    return MarkdownParser().parse(path, blobs=InMemoryBlobSink())


def test_headings_get_levels_and_paragraphs_get_parents(tmp_path):
    doc = _parse(
        tmp_path,
        "# Payment System\n\nThe Payment API processes requests.\n\n"
        "## Dependencies\n\nIt depends on Redis.\n",
    )
    elements = list(doc.iter_elements())
    kinds = [(e.type, e.level, e.text) for e in elements]

    assert kinds == [
        (ElementType.HEADING, 1, "Payment System"),
        (ElementType.PARAGRAPH, None, "The Payment API processes requests."),
        (ElementType.HEADING, 2, "Dependencies"),
        (ElementType.PARAGRAPH, None, "It depends on Redis."),
    ]
    assert elements[1].parent_id == elements[0].id
    assert elements[3].parent_id == elements[2].id


def test_single_page_with_monotonic_synthetic_bboxes(tmp_path):
    doc = _parse(tmp_path, "# A\n\npara one\n\npara two\n")
    assert doc.page_count == 1

    elements = list(doc.iter_elements())
    ys = [e.bbox.y0 for e in elements]
    assert ys == sorted(ys)
    assert all(e.attrs["synthetic_bbox"] is True for e in elements)
    assert all(e.bbox.y1 <= doc.pages[0].height for e in elements)


def test_list_items_become_list_item_elements(tmp_path):
    doc = _parse(tmp_path, "# Deps\n\n- Redis\n- PostgreSQL\n")
    items = [e for e in doc.iter_elements() if e.type is ElementType.LIST_ITEM]
    assert [e.text for e in items] == ["Redis", "PostgreSQL"]


def test_fenced_code_becomes_code_element(tmp_path):
    doc = _parse(tmp_path, "# T\n\n```python\nx = 1\n```\n")
    code = [e for e in doc.iter_elements() if e.type is ElementType.CODE]
    assert len(code) == 1
    assert code[0].text == "x = 1\n"
    assert code[0].attrs["language"] == "python"


def test_gfm_table_becomes_structured_table_element(tmp_path):
    doc = _parse(
        tmp_path,
        "# Services\n\n| Service | Store |\n| --- | --- |\n"
        "| Payment API | PostgreSQL |\n| Ledger | Redis |\n",
    )
    tables = [e for e in doc.iter_elements() if e.type is ElementType.TABLE]
    assert len(tables) == 1

    table = tables[0].table
    assert table.headers == ["Service", "Store"]
    assert table.rows == [["Payment API", "PostgreSQL"], ["Ledger", "Redis"]]
    assert (table.n_rows, table.n_cols) == (2, 2)


def test_title_is_the_first_level_one_heading(tmp_path):
    doc = _parse(tmp_path, "# Payment System\n\nbody\n")
    assert doc.title == "Payment System"


def test_document_identity_is_content_addressed(tmp_path):
    doc = _parse(tmp_path, "# A\n\nbody\n")
    other = _parse(tmp_path, "# A\n\nbody\n")
    assert doc.id == other.id
    assert doc.mime == "text/markdown"
    assert doc.meta["parser"] == "markdown"


def test_parsing_is_deterministic(tmp_path):
    from app.storage.base import serialize_document

    text = "# A\n\none\n\n## B\n\n- x\n- y\n"
    assert serialize_document(_parse(tmp_path, text)) == serialize_document(
        _parse(tmp_path, text)
    )


def test_every_source_line_survives_into_some_element(tmp_path):
    text = (
        "# Payment System\n\nThe Payment API depends on Redis.\n\n"
        "## Incidents\n\n- INC-2391 affected the Payment API\n"
    )
    doc = _parse(tmp_path, text)
    combined = " ".join(e.text_content() for e in doc.iter_elements())
    for fragment in ["Payment System", "depends on Redis", "Incidents", "INC-2391"]:
        assert fragment in combined


def test_empty_document_parses_to_an_empty_page(tmp_path):
    doc = _parse(tmp_path, "")
    assert doc.page_count == 1
    assert doc.pages[0].elements == []


def test_non_utf8_file_raises_parse_error(tmp_path):
    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ParseError, match="not valid UTF-8"):
        MarkdownParser().parse(path, blobs=InMemoryBlobSink())
