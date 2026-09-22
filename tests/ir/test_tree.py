from __future__ import annotations

import pytest

from app.ir.model import BBox, Document, Element, ElementType, Page
from app.ir.tree import assign_parents, element_index, iter_elements, section_tree


def _element(
    eid: str,
    etype: ElementType,
    order: int,
    *,
    level: int | None = None,
    page: int = 1,
    text: str = "",
) -> Element:
    return Element(
        id=eid,
        type=etype,
        page_number=page,
        bbox=BBox(x0=0, y0=order * 10, x1=100, y1=order * 10 + 8),
        order=order,
        level=level,
        text=text,
    )


def _doc(*pages: Page) -> Document:
    return Document(
        id="d", source_name="s.pdf", mime="application/pdf", checksum="c", pages=list(pages)
    )


def test_element_index_is_keyed_by_id_across_pages():
    doc = _doc(
        Page(number=1, width=612, height=792, elements=[_element("p1e000", ElementType.PARAGRAPH, 0)]),
        Page(number=2, width=612, height=792, elements=[_element("p2e000", ElementType.PARAGRAPH, 1, page=2)]),
    )
    index = element_index(doc)
    assert set(index) == {"p1e000", "p2e000"}
    assert index["p2e000"].page_number == 2


def test_element_index_rejects_duplicate_ids():
    doc = _doc(
        Page(
            number=1,
            width=612,
            height=792,
            elements=[
                _element("dup", ElementType.PARAGRAPH, 0),
                _element("dup", ElementType.PARAGRAPH, 1),
            ],
        )
    )
    with pytest.raises(ValueError, match="duplicate element id"):
        element_index(doc)


def test_iter_elements_follows_global_order_not_page_order():
    doc = _doc(
        Page(number=1, width=612, height=792, elements=[_element("a", ElementType.PARAGRAPH, 3)]),
        Page(number=2, width=612, height=792, elements=[_element("b", ElementType.PARAGRAPH, 1, page=2)]),
    )
    assert [e.id for e in iter_elements(doc)] == ["b", "a"]


def test_assign_parents_links_to_nearest_enclosing_heading():
    elements = [
        _element("h1", ElementType.HEADING, 0, level=1),
        _element("p1", ElementType.PARAGRAPH, 1),
        _element("h2", ElementType.HEADING, 2, level=2),
        _element("p2", ElementType.PARAGRAPH, 3),
        _element("h1b", ElementType.HEADING, 4, level=1),
        _element("p3", ElementType.PARAGRAPH, 5),
    ]
    assign_parents(elements)
    parents = {e.id: e.parent_id for e in elements}
    assert parents == {
        "h1": None,
        "p1": "h1",
        "h2": "h1",
        "p2": "h2",
        "h1b": None,
        "p3": "h1b",
    }


def test_section_tree_nests_by_heading_level():
    elements = [
        _element("h1", ElementType.HEADING, 0, level=1, text="Payments"),
        _element("p1", ElementType.PARAGRAPH, 1, text="intro"),
        _element("h2", ElementType.HEADING, 2, level=2, text="Redis"),
        _element("p2", ElementType.PARAGRAPH, 3, text="cache"),
    ]
    assign_parents(elements)
    doc = _doc(Page(number=1, width=612, height=792, elements=elements))
    tree = section_tree(doc)

    assert len(tree) == 1
    root = tree[0]
    assert root.heading.text == "Payments"
    assert [e.id for e in root.elements] == ["p1"]
    assert len(root.children) == 1
    assert root.children[0].heading.text == "Redis"
    assert [e.id for e in root.children[0].elements] == ["p2"]


def test_section_tree_handles_content_before_any_heading():
    elements = [
        _element("p0", ElementType.PARAGRAPH, 0, text="preamble"),
        _element("h1", ElementType.HEADING, 1, level=1, text="Body"),
    ]
    assign_parents(elements)
    doc = _doc(Page(number=1, width=612, height=792, elements=elements))
    tree = section_tree(doc)

    assert tree[0].heading is None
    assert [e.id for e in tree[0].elements] == ["p0"]
    assert tree[1].heading.id == "h1"


def test_section_tree_handles_level_jumps():
    elements = [
        _element("h1", ElementType.HEADING, 0, level=1),
        _element("h3", ElementType.HEADING, 1, level=3),
        _element("p", ElementType.PARAGRAPH, 2),
    ]
    assign_parents(elements)
    doc = _doc(Page(number=1, width=612, height=792, elements=elements))
    tree = section_tree(doc)

    assert tree[0].children[0].heading.id == "h3"
    assert [e.id for e in tree[0].children[0].elements] == ["p"]
