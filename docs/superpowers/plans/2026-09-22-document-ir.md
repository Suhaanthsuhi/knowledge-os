# Document IR + Parsing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert PDF, Markdown, image and spreadsheet files into a lossless, deterministic, coordinate-preserving Document IR that every later sub-project builds on.

**Architecture:** Parsers are pure functions from a file path to a `Document` — no model calls, no network, no filesystem writes. A `Document` holds pages, each holding a flat reading-ordered list of `Element`s carrying type, bbox, parent heading and stable id. Hierarchy is derived on demand rather than stored. Persistence sits behind a `DocumentStore` protocol with a JSON-plus-blobs file implementation.

**Tech Stack:** Python 3.13, Pydantic v2, PyMuPDF, pdfplumber, markdown-it-py, openpyxl, Pillow, pytest, reportlab.

**Spec:** `docs/superpowers/specs/2026-09-22-document-ir-design.md`

## Global Constraints

- **Do not run `git commit`.** The user commits and pushes. Every task ends by staging with `git add` only.
- Python `>=3.13`. Project uses `uv`; add dependencies with `uv add` / `uv add --dev`.
- Package root is `app/`. Tests live in `tests/`, mirroring the package layout.
- **Zero network calls and zero model calls in `app/ir/`, `app/parsing/`, `app/storage/`.** Parsing must be reproducible offline.
- **Determinism:** parsing the same bytes twice must yield byte-identical `document.json`. No timestamps, no `uuid`, no unordered set iteration in output.
- All coordinates are PDF points, origin top-left, y increasing downward.
- All `BBox` floats are rounded to 2 decimal places at model-validation time.
- Do not modify `app/config.py`, `app/graph/`, `app/extraction/`, or `playground/`.
- Type hints on every public function. `from __future__ import annotations` at the top of every module.

---

## File Structure

| File | Responsibility |
|---|---|
| `app/ir/model.py` | Pydantic models: `BBox`, `ElementType`, `TableData`, `ImageRef`, `Provenance`, `Element`, `Page`, `Document` |
| `app/ir/ids.py` | Deterministic id generation: document id, element id, blob ref |
| `app/ir/tree.py` | Element index, global reading-order iteration, derived section tree |
| `app/storage/base.py` | `DocumentStore` protocol, `serialize_document` |
| `app/storage/filestore.py` | `FileDocumentStore` — JSON + blob directory |
| `app/parsing/base.py` | `ParseError`, `BlobSink`, `InMemoryBlobSink`, `DocumentParser` protocol, `parse_document` dispatch |
| `app/parsing/markdown.py` | Markdown token stream → elements, synthetic bboxes |
| `app/parsing/layout.py` | Pure layout functions: `Block`, `LayoutConfig`, heading detection, gutters, reading order, margin blocks, caption pairing |
| `app/parsing/pdf.py` | PyMuPDF spans/images + pdfplumber tables, wired through `layout.py` |
| `app/parsing/image.py` | Standalone PNG/JPEG → one-page one-element document |
| `app/parsing/spreadsheet.py` | openpyxl → one `TABLE` element per sheet |
| `tests/fixtures/generate.py` | reportlab synthetic PDF builder |
| `tests/ir/`, `tests/storage/`, `tests/parsing/` | Unit, golden-file and invariant tests |

---

## Task 1: IR data model and ids

**Files:**
- Create: `app/ir/__init__.py`, `app/ir/model.py`, `app/ir/ids.py`
- Create: `tests/__init__.py`, `tests/ir/__init__.py`, `tests/ir/test_model.py`, `tests/ir/test_ids.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: `BBox(x0,y0,x1,y1)`, `ElementType` (StrEnum), `TableData`, `ImageRef`, `Provenance`, `Element`, `Page`, `Document`, `document_id(data: bytes) -> tuple[str, str]`, `element_id(page_number: int, index: int) -> str`, `blob_ref(page_number: int, index: int, ext: str) -> str`.

- [ ] **Step 1: Add dependencies**

```bash
uv add pymupdf pdfplumber markdown-it-py openpyxl pillow
uv add --dev pytest reportlab
```

Expected: `pyproject.toml` gains the five runtime deps and two dev deps; `uv.lock` updates.

- [ ] **Step 2: Create package directories**

```bash
mkdir -p app/ir app/parsing app/storage tests/ir tests/parsing tests/storage tests/fixtures
touch app/ir/__init__.py app/parsing/__init__.py app/storage/__init__.py
touch tests/__init__.py tests/ir/__init__.py tests/parsing/__init__.py tests/storage/__init__.py
```

- [ ] **Step 3: Write the failing tests**

Create `tests/ir/test_model.py`:

```python
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
```

Create `tests/ir/test_ids.py`:

```python
from __future__ import annotations

from app.ir.ids import blob_ref, document_id, element_id


def test_document_id_is_content_addressed_and_stable():
    short_a, full_a = document_id(b"hello world")
    short_b, full_b = document_id(b"hello world")
    assert short_a == short_b
    assert full_a == full_b
    assert len(short_a) == 16
    assert len(full_a) == 64
    assert full_a.startswith(short_a)


def test_document_id_differs_for_different_bytes():
    assert document_id(b"a")[0] != document_id(b"b")[0]


def test_element_id_is_zero_padded_and_page_scoped():
    assert element_id(12, 7) == "p12e007"
    assert element_id(1, 0) == "p1e000"


def test_blob_ref_includes_page_index_and_extension():
    assert blob_ref(3, 1, "png") == "p3_i001.png"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/ir -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ir.model'`

- [ ] **Step 5: Implement `app/ir/ids.py`**

```python
from __future__ import annotations

import hashlib

__all__ = ["document_id", "element_id", "blob_ref"]


def document_id(data: bytes) -> tuple[str, str]:
    """Return (short_id, full_checksum) derived from the file's bytes.

    Content addressing makes re-ingesting an identical file a no-op.
    """
    digest = hashlib.sha256(data).hexdigest()
    return digest[:16], digest


def element_id(page_number: int, index: int) -> str:
    """Stable element id, unique within a document."""
    return f"p{page_number}e{index:03d}"


def blob_ref(page_number: int, index: int, ext: str) -> str:
    """Stable filename for an extracted binary blob."""
    return f"p{page_number}_i{index:03d}.{ext.lstrip('.')}"
```

- [ ] **Step 6: Implement `app/ir/model.py`**

```python
from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterator

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "BBox",
    "ElementType",
    "TableData",
    "ImageRef",
    "Provenance",
    "Element",
    "Page",
    "Document",
]


class BBox(BaseModel):
    """Axis-aligned rectangle in PDF points, origin top-left, y increasing downward."""

    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x0", "y0", "x1", "y1")
    @classmethod
    def _round(cls, value: float) -> float:
        # Fixed precision keeps serialized output byte-identical across runs.
        return round(float(value), 2)

    @property
    def width(self) -> float:
        return round(self.x1 - self.x0, 2)

    @property
    def height(self) -> float:
        return round(self.y1 - self.y0, 2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, other: BBox) -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def horizontal_overlap_ratio(self, other: BBox) -> float:
        """Overlap of the shared x-range as a fraction of *self*'s width."""
        if self.width <= 0:
            return 0.0
        overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        return max(0.0, overlap) / self.width


class ElementType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    cell_bboxes: list[list[BBox | None]] | None = None

    @model_validator(mode="after")
    def _check_dimensions(self) -> TableData:
        if self.n_rows != len(self.rows):
            raise ValueError(f"n_rows={self.n_rows} but got {len(self.rows)} rows")
        for row in self.rows:
            if len(row) != self.n_cols:
                raise ValueError(f"row has {len(row)} cells, expected {self.n_cols}")
        if self.headers and len(self.headers) != self.n_cols:
            raise ValueError(
                f"{len(self.headers)} headers, expected {self.n_cols}"
            )
        return self

    def as_text(self) -> str:
        """Flat text rendering, used for lossless-text checks and fallbacks."""
        lines = []
        if self.headers:
            lines.append(" | ".join(self.headers))
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)


class ImageRef(BaseModel):
    blob_ref: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    caption_id: str | None = None


class Provenance(BaseModel):
    """The single value every downstream chunk, fact and graph edge carries."""

    doc_id: str
    page_number: int
    element_id: str
    bbox: BBox


class Element(BaseModel):
    id: str
    type: ElementType
    page_number: int
    bbox: BBox
    order: int
    parent_id: str | None = None
    text: str | None = None
    level: int | None = None
    table: TableData | None = None
    image: ImageRef | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    def provenance(self, doc_id: str) -> Provenance:
        return Provenance(
            doc_id=doc_id,
            page_number=self.page_number,
            element_id=self.id,
            bbox=self.bbox,
        )

    def text_content(self) -> str:
        """All text this element carries, including table cells."""
        if self.table is not None:
            return self.table.as_text()
        return self.text or ""


class Page(BaseModel):
    number: int
    width: float
    height: float
    elements: list[Element] = Field(default_factory=list)


class Document(BaseModel):
    id: str
    source_name: str
    mime: str
    checksum: str
    title: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    pages: list[Page] = Field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def add_warning(self, element_id: str | None, code: str, detail: str) -> None:
        """Record a per-element degradation. Content is never silently dropped."""
        self.meta.setdefault("warnings", []).append(
            {"element_id": element_id, "code": code, "detail": detail}
        )

    def element(self, element_id: str) -> Element:
        from app.ir.tree import element_index

        return element_index(self)[element_id]

    def iter_elements(self) -> Iterator[Element]:
        from app.ir.tree import iter_elements

        return iter_elements(self)

    def section_tree(self):
        from app.ir.tree import section_tree

        return section_tree(self)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/ir -v`
Expected: PASS, 11 tests.

- [ ] **Step 8: Stage the changes (do not commit)**

```bash
git add pyproject.toml uv.lock app/ir tests/__init__.py tests/ir
git status --short
```

---

## Task 2: Element index, reading order and derived section tree

**Files:**
- Create: `app/ir/tree.py`
- Create: `tests/ir/test_tree.py`

**Interfaces:**
- Consumes: `Document`, `Page`, `Element`, `ElementType`, `BBox` from Task 1.
- Produces: `element_index(doc: Document) -> dict[str, Element]`, `iter_elements(doc: Document) -> Iterator[Element]`, `SectionNode(heading: Element | None, elements: list[Element], children: list[SectionNode])`, `section_tree(doc: Document) -> list[SectionNode]`, `assign_parents(elements: list[Element]) -> None`.

- [ ] **Step 1: Write the failing test**

Create `tests/ir/test_tree.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ir/test_tree.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ir.tree'`

- [ ] **Step 3: Implement `app/ir/tree.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from app.ir.model import Document, Element, ElementType

__all__ = ["element_index", "iter_elements", "SectionNode", "section_tree", "assign_parents"]


def element_index(doc: Document) -> dict[str, Element]:
    """O(1) lookup of every element in the document by id."""
    index: dict[str, Element] = {}
    for page in doc.pages:
        for element in page.elements:
            if element.id in index:
                raise ValueError(f"duplicate element id: {element.id}")
            index[element.id] = element
    return index


def iter_elements(doc: Document) -> Iterator[Element]:
    """Walk every element in global reading order, across page boundaries."""
    everything = [element for page in doc.pages for element in page.elements]
    yield from sorted(everything, key=lambda e: e.order)


def assign_parents(elements: list[Element]) -> None:
    """Set `parent_id` on each element to its nearest enclosing heading.

    Mutates in place. Elements must already be in reading order. A heading's
    parent is the nearest preceding heading of a strictly smaller level.
    """
    stack: list[Element] = []
    for element in sorted(elements, key=lambda e: e.order):
        if element.type is ElementType.HEADING:
            level = element.level or 1
            while stack and (stack[-1].level or 1) >= level:
                stack.pop()
            element.parent_id = stack[-1].id if stack else None
            stack.append(element)
        else:
            element.parent_id = stack[-1].id if stack else None


@dataclass
class SectionNode:
    """A heading and everything beneath it. `heading` is None for preamble content."""

    heading: Element | None
    elements: list[Element] = field(default_factory=list)
    children: list[SectionNode] = field(default_factory=list)

    @property
    def level(self) -> int:
        if self.heading is None:
            return 0
        return self.heading.level or 1

    def title(self) -> str:
        return (self.heading.text or "") if self.heading else ""


def section_tree(doc: Document) -> list[SectionNode]:
    """Derive the heading hierarchy on demand. Storage stays flat."""
    roots: list[SectionNode] = []
    stack: list[SectionNode] = []

    for element in iter_elements(doc):
        if element.type is ElementType.HEADING:
            node = SectionNode(heading=element)
            level = element.level or 1
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        else:
            if not stack:
                preamble = SectionNode(heading=None)
                roots.append(preamble)
                stack.append(preamble)
            stack[-1].elements.append(element)

    return roots
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ir -v`
Expected: PASS, 18 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/ir/tree.py tests/ir/test_tree.py
git status --short
```

---

## Task 3: Document storage

**Files:**
- Create: `app/storage/base.py`, `app/storage/filestore.py`
- Create: `tests/storage/test_filestore.py`

**Interfaces:**
- Consumes: `Document` from Task 1.
- Produces: `serialize_document(doc: Document) -> str`, `DocumentStore` protocol, `FileDocumentStore(root: Path)` with `put(doc, blobs) -> str`, `get(doc_id) -> Document`, `get_blob(doc_id, ref) -> bytes`, `exists(doc_id) -> bool`, `list_ids() -> Iterator[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/storage/test_filestore.py`:

```python
from __future__ import annotations

import json

import pytest

from app.ir.model import BBox, Document, Element, ElementType, ImageRef, Page
from app.storage.filestore import FileDocumentStore


def _doc(doc_id: str = "abc123") -> Document:
    return Document(
        id=doc_id,
        source_name="payment.pdf",
        mime="application/pdf",
        checksum="f" * 64,
        title="Payment Architecture",
        meta={"page_count": 1, "parser": "pdf", "parser_version": "1"},
        pages=[
            Page(
                number=1,
                width=612,
                height=792,
                elements=[
                    Element(
                        id="p1e000",
                        type=ElementType.HEADING,
                        page_number=1,
                        bbox=BBox(x0=72, y0=100, x1=300, y1=120),
                        order=0,
                        level=1,
                        text="Payment Architecture",
                    ),
                    Element(
                        id="p1e001",
                        type=ElementType.IMAGE,
                        page_number=1,
                        bbox=BBox(x0=72, y0=140, x1=400, y1=340),
                        order=1,
                        parent_id="p1e000",
                        image=ImageRef(blob_ref="p1_i000.png", width=640, height=400, format="png"),
                    ),
                ],
            )
        ],
    )


def test_put_then_get_round_trips(tmp_path):
    store = FileDocumentStore(tmp_path)
    doc = _doc()

    doc_id = store.put(doc, {"p1_i000.png": b"\x89PNG-bytes"})

    assert doc_id == "abc123"
    loaded = store.get("abc123")
    assert loaded == doc


def test_put_writes_expected_layout(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {"p1_i000.png": b"bytes"})

    assert (tmp_path / "abc123" / "document.json").is_file()
    assert (tmp_path / "abc123" / "blobs" / "p1_i000.png").read_bytes() == b"bytes"


def test_serialized_json_is_sorted_and_deterministic(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {})
    first = (tmp_path / "abc123" / "document.json").read_text()

    store.put(_doc(), {})
    second = (tmp_path / "abc123" / "document.json").read_text()

    assert first == second
    top_level = list(json.loads(first).keys())
    assert top_level == sorted(top_level)


def test_get_blob_returns_bytes(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {"p1_i000.png": b"\x01\x02"})
    assert store.get_blob("abc123", "p1_i000.png") == b"\x01\x02"


def test_exists_and_list_ids(tmp_path):
    store = FileDocumentStore(tmp_path)
    assert not store.exists("abc123")
    store.put(_doc("abc123"), {})
    store.put(_doc("def456"), {})
    assert store.exists("abc123")
    assert sorted(store.list_ids()) == ["abc123", "def456"]


def test_get_missing_document_raises(tmp_path):
    store = FileDocumentStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("nope")


def test_blob_ref_cannot_escape_the_document_directory(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {})
    with pytest.raises(ValueError, match="invalid blob ref"):
        store.get_blob("abc123", "../../etc/passwd")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/storage -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.storage.filestore'`

- [ ] **Step 3: Implement `app/storage/base.py`**

```python
from __future__ import annotations

import json
from typing import Iterator, Mapping, Protocol, runtime_checkable

from app.ir.model import Document

__all__ = ["serialize_document", "deserialize_document", "DocumentStore"]


def serialize_document(doc: Document) -> str:
    """Canonical JSON: sorted keys, stable indentation, no ASCII escaping.

    BBox floats are already rounded at validation time, so the same document
    always produces the same bytes.
    """
    payload = doc.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def deserialize_document(raw: str) -> Document:
    return Document.model_validate_json(raw)


@runtime_checkable
class DocumentStore(Protocol):
    def put(self, doc: Document, blobs: Mapping[str, bytes]) -> str: ...
    def get(self, doc_id: str) -> Document: ...
    def get_blob(self, doc_id: str, ref: str) -> bytes: ...
    def exists(self, doc_id: str) -> bool: ...
    def list_ids(self) -> Iterator[str]: ...
```

- [ ] **Step 4: Implement `app/storage/filestore.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping

from app.ir.model import Document
from app.storage.base import deserialize_document, serialize_document

__all__ = ["FileDocumentStore"]

DOCUMENT_FILENAME = "document.json"
BLOB_DIRNAME = "blobs"


class FileDocumentStore:
    """Stores each document as `<root>/<doc_id>/document.json` plus a blob directory.

    Because `doc_id` is a content hash, re-ingesting an identical file simply
    overwrites identical bytes — idempotency needs no dedup pass.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    @staticmethod
    def _check_ref(ref: str) -> None:
        if "/" in ref or "\\" in ref or ref in {".", ".."} or ref.startswith("."):
            raise ValueError(f"invalid blob ref: {ref!r}")

    def put(self, doc: Document, blobs: Mapping[str, bytes]) -> str:
        target = self._dir(doc.id)
        target.mkdir(parents=True, exist_ok=True)
        if blobs:
            blob_dir = target / BLOB_DIRNAME
            blob_dir.mkdir(exist_ok=True)
            for ref, data in sorted(blobs.items()):
                self._check_ref(ref)
                (blob_dir / ref).write_bytes(data)
        (target / DOCUMENT_FILENAME).write_text(
            serialize_document(doc), encoding="utf-8"
        )
        return doc.id

    def get(self, doc_id: str) -> Document:
        path = self._dir(doc_id) / DOCUMENT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"no stored document with id {doc_id!r}")
        return deserialize_document(path.read_text(encoding="utf-8"))

    def get_blob(self, doc_id: str, ref: str) -> bytes:
        self._check_ref(ref)
        path = self._dir(doc_id) / BLOB_DIRNAME / ref
        if not path.is_file():
            raise FileNotFoundError(f"no blob {ref!r} for document {doc_id!r}")
        return path.read_bytes()

    def exists(self, doc_id: str) -> bool:
        return (self._dir(doc_id) / DOCUMENT_FILENAME).is_file()

    def list_ids(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for child in sorted(self.root.iterdir()):
            if (child / DOCUMENT_FILENAME).is_file():
                yield child.name
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/storage -v`
Expected: PASS, 7 tests.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add app/storage tests/storage
git status --short
```

---

## Task 4: Parser protocol, blob sink and dispatch

**Files:**
- Create: `app/parsing/base.py`
- Create: `tests/parsing/test_base.py`

**Interfaces:**
- Consumes: `Document` (Task 1), `document_id` (Task 1).
- Produces: `ParseError`, `BlobSink` protocol with `add(ref: str, data: bytes) -> None`, `InMemoryBlobSink` with a `.blobs: dict[str, bytes]` attribute, `DocumentParser` protocol with `extensions: ClassVar[frozenset[str]]` / `version: ClassVar[str]` / `parse(path, *, blobs) -> Document`, `register_parser(extension: str, parser: DocumentParser) -> None`, `detect_extension(path: Path) -> str`, `parse_document(path: Path | str, *, blobs: BlobSink | None = None) -> Document`, `source_identity(path: Path) -> tuple[str, str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/parsing/test_base.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from app.ir.model import Document
from app.parsing import base
from app.parsing.base import (
    InMemoryBlobSink,
    ParseError,
    detect_extension,
    parse_document,
    register_parser,
    source_identity,
)


@pytest.fixture
def clean_registry():
    saved = dict(base._REGISTRY)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(saved)


class FakeParser:
    extensions = frozenset({".fake"})
    version = "1"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def parse(self, path: Path, *, blobs) -> Document:
        self.calls.append(path)
        blobs.add("p1_i000.png", b"x")
        short_id, checksum = source_identity(path)
        return Document(
            id=short_id, source_name=path.name, mime="application/fake", checksum=checksum
        )


def test_in_memory_blob_sink_collects_bytes():
    sink = InMemoryBlobSink()
    sink.add("a.png", b"1")
    sink.add("b.png", b"2")
    assert sink.blobs == {"a.png": b"1", "b.png": b"2"}


def test_in_memory_blob_sink_rejects_duplicate_refs():
    sink = InMemoryBlobSink()
    sink.add("a.png", b"1")
    with pytest.raises(ValueError, match="duplicate blob ref"):
        sink.add("a.png", b"2")


def test_source_identity_is_content_addressed(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    short_id, checksum = source_identity(path)
    assert len(short_id) == 16
    assert checksum.startswith(short_id)


def test_parse_document_dispatches_on_extension(tmp_path, clean_registry):
    parser = FakeParser()
    register_parser(".fake", parser)
    path = tmp_path / "doc.fake"
    path.write_bytes(b"content")

    doc = parse_document(path)

    assert parser.calls == [path]
    assert doc.source_name == "doc.fake"


def test_parse_document_collects_blobs_when_sink_supplied(tmp_path, clean_registry):
    register_parser(".fake", FakeParser())
    path = tmp_path / "doc.fake"
    path.write_bytes(b"content")

    sink = InMemoryBlobSink()
    parse_document(path, blobs=sink)

    assert sink.blobs == {"p1_i000.png": b"x"}


def test_parse_document_rejects_unknown_extension(tmp_path):
    path = tmp_path / "doc.unknown"
    path.write_bytes(b"not a known format")
    with pytest.raises(ParseError, match="no parser"):
        parse_document(path)


def test_parse_document_rejects_missing_file(tmp_path):
    with pytest.raises(ParseError, match="not a readable file"):
        parse_document(tmp_path / "absent.pdf")


@pytest.mark.parametrize(
    ("magic", "expected"),
    [
        (b"%PDF-1.7\n", ".pdf"),
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff\xe0", ".jpg"),
        (b"PK\x03\x04", ".xlsx"),
    ],
)
def test_detect_extension_sniffs_magic_bytes(tmp_path, magic, expected):
    path = tmp_path / "mystery"
    path.write_bytes(magic + b"padding" * 10)
    assert detect_extension(path) == expected


def test_detect_extension_prefers_the_suffix(tmp_path):
    path = tmp_path / "doc.md"
    path.write_bytes(b"# heading")
    assert detect_extension(path) == ".md"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.parsing.base'`

- [ ] **Step 3: Implement `app/parsing/base.py`**

```python
from __future__ import annotations

import importlib
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from app.ir.ids import document_id
from app.ir.model import Document

__all__ = [
    "ParseError",
    "BlobSink",
    "InMemoryBlobSink",
    "DocumentParser",
    "register_parser",
    "detect_extension",
    "parse_document",
    "source_identity",
]


class ParseError(Exception):
    """Raised when a document cannot be parsed at all. Never a half-document."""


@runtime_checkable
class BlobSink(Protocol):
    def add(self, ref: str, data: bytes) -> None: ...


class InMemoryBlobSink:
    """Collects extracted binaries so parsers never touch the filesystem."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def add(self, ref: str, data: bytes) -> None:
        if ref in self.blobs:
            raise ValueError(f"duplicate blob ref: {ref!r}")
        self.blobs[ref] = data


@runtime_checkable
class DocumentParser(Protocol):
    extensions: ClassVar[frozenset[str]]
    version: ClassVar[str]

    def parse(self, path: Path, *, blobs: BlobSink) -> Document: ...


# Extension -> dotted module path and class name. Imported lazily so that
# importing app.parsing.base does not pull in PyMuPDF, pdfplumber or openpyxl.
_BUILTIN: dict[str, tuple[str, str]] = {
    ".pdf": ("app.parsing.pdf", "PdfParser"),
    ".md": ("app.parsing.markdown", "MarkdownParser"),
    ".markdown": ("app.parsing.markdown", "MarkdownParser"),
    ".png": ("app.parsing.image", "ImageParser"),
    ".jpg": ("app.parsing.image", "ImageParser"),
    ".jpeg": ("app.parsing.image", "ImageParser"),
    ".xlsx": ("app.parsing.spreadsheet", "SpreadsheetParser"),
}

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"PK\x03\x04", ".xlsx"),
)

_REGISTRY: dict[str, DocumentParser] = {}


def register_parser(extension: str, parser: DocumentParser) -> None:
    """Register (or override) the parser used for an extension."""
    _REGISTRY[extension.lower()] = parser


def _get_parser(extension: str) -> DocumentParser:
    extension = extension.lower()
    if extension in _REGISTRY:
        return _REGISTRY[extension]
    if extension not in _BUILTIN:
        raise ParseError(f"no parser registered for extension {extension!r}")
    module_name, class_name = _BUILTIN[extension]
    module = importlib.import_module(module_name)
    parser = getattr(module, class_name)()
    _REGISTRY[extension] = parser
    return parser


def detect_extension(path: Path) -> str:
    """Extension from the suffix, falling back to magic-byte sniffing."""
    suffix = path.suffix.lower()
    if suffix and (suffix in _REGISTRY or suffix in _BUILTIN):
        return suffix
    head = path.read_bytes()[:16]
    for magic, extension in _MAGIC:
        if head.startswith(magic):
            return extension
    return suffix


def source_identity(path: Path) -> tuple[str, str]:
    """(short document id, full sha256) for the file's bytes."""
    return document_id(path.read_bytes())


def parse_document(path: Path | str, *, blobs: BlobSink | None = None) -> Document:
    """Parse any supported file into the IR. No network, no model calls."""
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"not a readable file: {path}")
    sink = blobs if blobs is not None else InMemoryBlobSink()
    parser = _get_parser(detect_extension(path))
    return parser.parse(path, blobs=sink)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/parsing/test_base.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/parsing/base.py tests/parsing/test_base.py
git status --short
```

---

## Task 5: Markdown parser

**Files:**
- Create: `app/parsing/markdown.py`
- Create: `tests/parsing/test_markdown.py`

**Interfaces:**
- Consumes: `Document`, `Page`, `Element`, `ElementType`, `BBox`, `TableData` (Task 1); `assign_parents` (Task 2); `ParseError`, `BlobSink`, `source_identity` (Task 4).
- Produces: `MarkdownParser` with `extensions = frozenset({".md", ".markdown"})`, `version = "1"`, `parse(path, *, blobs) -> Document`; module constants `LINE_HEIGHT = 14.0`, `PAGE_WIDTH = 612.0`, `MARGIN = 72.0`.

Element text is the **raw markdown source** of each block (markdown-it's
`inline.content`), not rendered plain text. That keeps the markdown path
trivially lossless and preserves emphasis and link syntax for later
extraction.

- [ ] **Step 1: Write the failing test**

Create `tests/parsing/test_markdown.py`:

```python
from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink
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
    from app.parsing.base import ParseError

    path = tmp_path / "bad.md"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ParseError, match="not valid UTF-8"):
        MarkdownParser().parse(path, blobs=InMemoryBlobSink())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_markdown.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.parsing.markdown'`

- [ ] **Step 3: Implement `app/parsing/markdown.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.ir.ids import element_id
from app.ir.model import BBox, Document, Element, ElementType, Page, TableData
from app.ir.tree import assign_parents
from app.parsing.base import BlobSink, ParseError, source_identity

__all__ = ["MarkdownParser", "LINE_HEIGHT", "PAGE_WIDTH", "MARGIN"]

LINE_HEIGHT = 14.0
PAGE_WIDTH = 612.0
MARGIN = 72.0


class MarkdownParser:
    """Markdown token stream to IR.

    Markdown carries no geometry, so bboxes are synthetic: one continuous
    page whose y coordinates derive from source line numbers. Every element
    is tagged `attrs["synthetic_bbox"] = True` so downstream code can tell
    "position unknown" from a real rectangle.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".md", ".markdown"})
    version: ClassVar[str] = "1"

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(f"{path} is not valid UTF-8: {exc}") from exc

        short_id, checksum = source_identity(path)
        # gfm-like brings tables and strikethrough. linkify is switched off: it
        # needs an extra dependency and autolinking adds nothing to the IR.
        tokens = MarkdownIt("gfm-like", {"linkify": False}).parse(text)

        builder = _ElementBuilder()
        builder.walk(tokens)
        elements = builder.elements

        assign_parents(elements)

        height = max(
            (e.bbox.y1 for e in elements),
            default=MARGIN,
        ) + MARGIN
        page = Page(number=1, width=PAGE_WIDTH, height=round(height, 2), elements=elements)

        title = next(
            (
                e.text
                for e in elements
                if e.type is ElementType.HEADING and e.level == 1 and e.text
            ),
            None,
        )

        return Document(
            id=short_id,
            source_name=path.name,
            mime="text/markdown",
            checksum=checksum,
            title=title,
            meta={
                "page_count": 1,
                "parser": "markdown",
                "parser_version": self.version,
                "line_count": text.count("\n") + 1,
            },
            pages=[page],
        )


def _bbox_for(token_map: list[int] | None, fallback_line: int) -> BBox:
    """Synthetic geometry: y derives from source line numbers, x spans the text column."""
    if token_map:
        start, end = token_map[0], max(token_map[1], token_map[0] + 1)
    else:
        start, end = fallback_line, fallback_line + 1
    return BBox(
        x0=MARGIN,
        y0=MARGIN + start * LINE_HEIGHT,
        x1=PAGE_WIDTH - MARGIN,
        y1=MARGIN + end * LINE_HEIGHT,
    )


class _ElementBuilder:
    """Turns a markdown-it token stream into a flat, ordered element list."""

    def __init__(self) -> None:
        self.elements: list[Element] = []
        self._order = 0
        self._last_line = 0
        self._list_depth = 0

    def _emit(
        self,
        etype: ElementType,
        token_map: list[int] | None,
        *,
        text: str | None = None,
        level: int | None = None,
        table: TableData | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        bbox = _bbox_for(token_map, self._last_line)
        if token_map:
            self._last_line = max(token_map[1], token_map[0] + 1)
        else:
            self._last_line += 1

        element_attrs: dict[str, Any] = {"synthetic_bbox": True}
        if attrs:
            element_attrs.update(attrs)

        self.elements.append(
            Element(
                id=element_id(1, self._order),
                type=etype,
                page_number=1,
                bbox=bbox,
                order=self._order,
                text=text,
                level=level,
                table=table,
                attrs=element_attrs,
            )
        )
        self._order += 1

    def walk(self, tokens: list[Token]) -> None:
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                inline = tokens[i + 1]
                self._emit(
                    ElementType.HEADING,
                    token.map,
                    text=inline.content.strip(),
                    level=int(token.tag[1:]),
                )
                i += 3
                continue

            if token.type in {"bullet_list_open", "ordered_list_open"}:
                self._list_depth += 1
                i += 1
                continue

            if token.type in {"bullet_list_close", "ordered_list_close"}:
                self._list_depth -= 1
                i += 1
                continue

            if token.type == "paragraph_open":
                inline = tokens[i + 1]
                etype = (
                    ElementType.LIST_ITEM if self._list_depth > 0 else ElementType.PARAGRAPH
                )
                self._emit(etype, token.map, text=inline.content.strip())
                i += 3
                continue

            if token.type in {"fence", "code_block"}:
                self._emit(
                    ElementType.CODE,
                    token.map,
                    text=token.content,
                    attrs={"language": (token.info or "").strip() or None},
                )
                i += 1
                continue

            if token.type == "table_open":
                end = _matching_close(tokens, i, "table_open", "table_close")
                table = _build_table(tokens[i : end + 1])
                self._emit(ElementType.TABLE, token.map, table=table)
                i = end + 1
                continue

            i += 1


def _matching_close(tokens: list[Token], start: int, open_type: str, close_type: str) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].type == open_type:
            depth += 1
        elif tokens[index].type == close_type:
            depth -= 1
            if depth == 0:
                return index
    raise ParseError(f"unbalanced {open_type} in markdown token stream")


def _build_table(tokens: list[Token]) -> TableData:
    headers: list[str] = []
    rows: list[list[str]] = []
    current: list[str] = []
    in_header = False

    for token in tokens:
        if token.type == "thead_open":
            in_header = True
        elif token.type == "thead_close":
            in_header = False
        elif token.type == "tr_open":
            current = []
        elif token.type == "tr_close":
            if in_header:
                headers = current
            else:
                rows.append(current)
        elif token.type == "inline":
            current.append(token.content.strip())

    n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    normalized = [row + [""] * (n_cols - len(row)) for row in rows]
    return TableData(
        headers=headers,
        rows=normalized,
        n_rows=len(normalized),
        n_cols=n_cols,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/parsing/test_markdown.py -v`
Expected: PASS, 11 tests.

- [ ] **Step 5: Verify the project's own asset parses**

Run:

```bash
uv run python -c "
from app.parsing.base import parse_document
doc = parse_document('app/assets/payment_system.md')
print(doc.id, doc.title, doc.page_count)
for e in doc.iter_elements():
    print(f'  {e.id} {e.type} {(e.text or \"\")[:50]!r}')
"
```

Expected: a document id, title `Payment System`, one page, and elements
covering every heading and paragraph of the asset.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add app/parsing/markdown.py tests/parsing/test_markdown.py
git status --short
```

---

## Task 6: Layout analysis

**Files:**
- Create: `app/parsing/layout.py`
- Create: `tests/parsing/test_layout.py`

**Interfaces:**
- Consumes: `BBox`, `ElementType` (Task 1).
- Produces: `LayoutConfig` (frozen dataclass), `Block` (dataclass with `index, page_number, bbox, text, max_size, is_bold, line_count, gap_below, kind`), `body_font_size(blocks, cfg) -> float`, `heading_levels(blocks, body_size, cfg) -> dict[int, int]`, `find_gutters(blocks, cfg) -> list[float]`, `column_index(bbox, gutters) -> int`, `reading_order(blocks, gutters) -> list[Block]`, `margin_blocks(blocks, page_height, page_count, cfg) -> dict[int, ElementType]`, `reading_order(blocks, gutters, zones=None) -> list[Block]`, `pair_captions(figures, text_blocks, body_size, cfg) -> dict[int, int]`.

Every function here is pure and operates on `Block` values, never on
PyMuPDF objects. That is what makes them unit-testable without a PDF.

- [ ] **Step 1: Write the failing test**

Create `tests/parsing/test_layout.py`:

```python
from __future__ import annotations

import pytest

from app.ir.model import BBox, ElementType
from app.parsing.layout import (
    Block,
    LayoutConfig,
    body_font_size,
    find_gutters,
    heading_levels,
    margin_blocks,
    pair_captions,
    reading_order,
)

CFG = LayoutConfig()


def block(
    index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "body text here",
    *,
    size: float = 11.0,
    bold: bool = False,
    lines: int = 1,
    gap_below: float = 0.0,
    page: int = 1,
    kind: str = "text",
) -> Block:
    return Block(
        index=index,
        page_number=page,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        text=text,
        max_size=size,
        is_bold=bold,
        line_count=lines,
        gap_below=gap_below,
        kind=kind,
    )


def test_body_font_size_is_the_char_weighted_median():
    blocks = [
        block(0, 72, 100, 300, 120, "Title", size=18.0, bold=True),
        block(1, 72, 130, 500, 200, "x" * 400, size=11.0),
        block(2, 72, 210, 500, 280, "y" * 400, size=11.0),
    ]
    assert body_font_size(blocks, CFG) == 11.0


def test_body_font_size_ignores_non_text_blocks():
    blocks = [
        block(0, 72, 100, 500, 200, "x" * 100, size=10.0),
        block(1, 72, 210, 500, 400, "", size=0.0, kind="image"),
    ]
    assert body_font_size(blocks, CFG) == 10.0


def test_larger_text_becomes_headings_ranked_by_size():
    blocks = [
        block(0, 72, 100, 300, 120, "Payment Architecture", size=18.0, bold=True),
        block(1, 72, 130, 500, 200, "x" * 200, size=11.0),
        block(2, 72, 210, 300, 228, "Dependencies", size=14.0, bold=True),
        block(3, 72, 240, 500, 320, "y" * 200, size=11.0),
    ]
    levels = heading_levels(blocks, 11.0, CFG)
    assert levels == {0: 1, 2: 2}


def test_bold_short_line_with_a_gap_is_a_heading_at_body_size():
    blocks = [
        block(0, 72, 100, 200, 114, "Overview", size=11.0, bold=True, lines=1, gap_below=12.0),
        block(1, 72, 130, 500, 300, "x" * 400, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {0: 1}


def test_bold_but_long_paragraph_is_not_a_heading():
    blocks = [
        block(0, 72, 100, 500, 200, "b" * 300, size=11.0, bold=True, lines=6, gap_below=12.0),
        block(1, 72, 210, 500, 300, "x" * 400, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {}


def test_heading_sizes_are_clustered_to_half_points():
    blocks = [
        block(0, 72, 100, 300, 120, "A", size=18.02, bold=True),
        block(1, 72, 130, 300, 150, "B", size=17.98, bold=True),
        block(2, 72, 160, 500, 400, "x" * 400, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {0: 1, 1: 1}


def test_find_gutters_detects_a_two_column_layout():
    left = [block(i, 72, 100 + i * 40, 272, 130 + i * 40, "x" * 80) for i in range(8)]
    right = [block(8 + i, 340, 100 + i * 40, 540, 130 + i * 40, "y" * 80) for i in range(8)]
    gutters = find_gutters(left + right, CFG)

    assert len(gutters) == 1
    assert 272 < gutters[0] < 340


def test_find_gutters_returns_nothing_for_a_single_column():
    blocks = [block(i, 72, 100 + i * 40, 540, 130 + i * 40, "x" * 80) for i in range(8)]
    assert find_gutters(blocks, CFG) == []


def test_column_index_counts_gutters_to_the_left():
    assert column_index(BBox(x0=72, y0=0, x1=272, y1=10), [306.0]) == 0
    assert column_index(BBox(x0=340, y0=0, x1=540, y1=10), [306.0]) == 1


def test_reading_order_reads_down_each_column_in_turn():
    blocks = [
        block(0, 72, 100, 272, 130, "left top"),
        block(1, 340, 100, 540, 130, "right top"),
        block(2, 72, 200, 272, 230, "left bottom"),
        block(3, 340, 200, 540, 230, "right bottom"),
    ]
    ordered = reading_order(blocks, [306.0])
    assert [b.text for b in ordered] == [
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    ]


def test_reading_order_falls_back_to_top_to_bottom_without_gutters():
    blocks = [
        block(0, 300, 200, 500, 230, "second"),
        block(1, 72, 100, 272, 130, "first"),
        block(2, 72, 200, 272, 230, "middle"),
    ]
    assert [b.text for b in reading_order(blocks, [])] == ["first", "middle", "second"]


def test_reading_order_puts_headers_first_and_footers_last():
    blocks = [
        block(0, 72, 300, 540, 340, "body"),
        block(1, 72, 30, 540, 50, "running header"),
        block(2, 72, 750, 540, 765, "running footer"),
    ]
    zones = {1: ElementType.HEADER, 2: ElementType.FOOTER}
    ordered = reading_order(blocks, [], zones)
    assert [b.text for b in ordered] == ["running header", "body", "running footer"]


def test_repeated_bottom_margin_text_becomes_a_footer():
    blocks = [
        block(i, 72, 740, 540, 760, "Knowledge OS Internal", size=8.0, page=i + 1)
        for i in range(3)
    ]
    blocks.append(block(99, 72, 300, 540, 400, "real body text", size=11.0))

    result = margin_blocks(blocks, page_height=792, page_count=3, cfg=CFG)
    assert result == {0: ElementType.FOOTER, 1: ElementType.FOOTER, 2: ElementType.FOOTER}


def test_repeated_top_margin_text_becomes_a_header():
    blocks = [
        block(i, 72, 30, 540, 50, "Payment Platform Handbook", size=9.0, page=i + 1)
        for i in range(4)
    ]
    result = margin_blocks(blocks, page_height=792, page_count=4, cfg=CFG)
    assert set(result.values()) == {ElementType.HEADER}


def test_bare_numbers_in_the_margin_become_page_numbers():
    blocks = [
        block(0, 300, 750, 320, 765, "1", size=9.0, page=1),
        block(1, 300, 750, 320, 765, "2", size=9.0, page=2),
    ]
    result = margin_blocks(blocks, page_height=792, page_count=2, cfg=CFG)
    assert result == {0: ElementType.PAGE_NUMBER, 1: ElementType.PAGE_NUMBER}


def test_body_text_in_the_margin_band_is_left_alone_when_it_does_not_repeat():
    blocks = [
        block(0, 72, 740, 540, 760, "a unique closing sentence", size=11.0, page=1),
        block(1, 72, 740, 540, 760, "a different closing sentence", size=11.0, page=2),
    ]
    assert margin_blocks(blocks, page_height=792, page_count=2, cfg=CFG) == {}


def test_caption_below_a_figure_is_paired():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    caption = block(1, 100, 410, 400, 424, "Figure 1: Payment architecture", size=9.0)
    body = block(2, 72, 500, 540, 600, "x" * 200, size=11.0)

    assert pair_captions([figure], [caption, body], 11.0, CFG) == {0: 1}


def test_caption_above_is_used_when_nothing_sits_below():
    figure = block(0, 100, 300, 400, 500, "", size=0.0, kind="image")
    caption = block(1, 100, 275, 400, 292, "Table 2: Service owners", size=9.0)

    assert pair_captions([figure], [caption], 11.0, CFG) == {0: 1}


def test_a_distant_block_is_not_treated_as_a_caption():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    far = block(1, 100, 600, 400, 620, "Figure 9: unrelated", size=9.0)

    assert pair_captions([figure], [far], 11.0, CFG) == {}


def test_a_block_that_barely_overlaps_the_figure_is_not_a_caption():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    aside = block(1, 420, 410, 540, 424, "Figure 4: sidebar", size=9.0)

    assert pair_captions([figure], [aside], 11.0, CFG) == {}


def test_a_caption_is_claimed_by_only_one_figure():
    upper = block(0, 100, 100, 400, 300, "", size=0.0, kind="image")
    lower = block(1, 100, 340, 400, 500, "", size=0.0, kind="image")
    caption = block(2, 100, 310, 400, 326, "Figure 1: shared", size=9.0)

    paired = pair_captions([upper, lower], [caption], 11.0, CFG)
    assert list(paired.values()) == [2]
    assert len(paired) == 1


def test_pattern_match_outranks_a_merely_smaller_block():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    small = block(1, 100, 405, 400, 415, "some small note", size=9.0)
    labelled = block(2, 100, 420, 400, 434, "Figure 7: the real caption", size=9.0)

    assert pair_captions([figure], [small, labelled], 11.0, CFG) == {0: 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.parsing.layout'`

- [ ] **Step 3: Implement `app/parsing/layout.py`**

```python
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.ir.model import BBox, ElementType

__all__ = [
    "LayoutConfig",
    "Block",
    "body_font_size",
    "heading_levels",
    "find_gutters",
    "column_index",
    "reading_order",
    "margin_blocks",
    "pair_captions",
]

CAPTION_PATTERN = re.compile(r"^(figure|fig\.|table|chart)\s*\d+", re.IGNORECASE)
PAGE_NUMBER_PATTERN = re.compile(
    r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$|^[ivxlcdm]+$", re.IGNORECASE
)
MAX_HEADING_LEVEL = 6


@dataclass(frozen=True)
class LayoutConfig:
    """Every layout threshold in one place, so they are tunable and visible."""

    heading_size_ratio: float = 1.15
    heading_max_chars: int = 80
    heading_gap_ratio: float = 0.5
    body_line_height: float = 12.0
    gutter_min_width: float = 18.0
    gutter_min_coverage: float = 0.60
    margin_band_ratio: float = 0.08
    repeat_page_ratio: float = 0.50
    caption_max_distance: float = 40.0
    caption_max_chars: int = 200
    caption_min_overlap: float = 0.50


@dataclass
class Block:
    """A positioned run of content, independent of any PDF library."""

    index: int
    page_number: int
    bbox: BBox
    text: str
    max_size: float = 0.0
    is_bold: bool = False
    line_count: int = 1
    gap_below: float = 0.0
    kind: str = "text"  # "text" | "table" | "image"
    attrs: dict = field(default_factory=dict)


def _text_blocks(blocks: list[Block]) -> list[Block]:
    return [b for b in blocks if b.kind == "text" and b.max_size > 0]


def body_font_size(blocks: list[Block], cfg: LayoutConfig) -> float:
    """Character-weighted median font size across the whole document body.

    Document-wide rather than per page: a title page's few huge glyphs would
    otherwise drag a per-page median upward and suppress real headings.
    """
    weighted: list[tuple[float, int]] = [
        (b.max_size, max(len(b.text.strip()), 1)) for b in _text_blocks(blocks)
    ]
    if not weighted:
        return 0.0

    weighted.sort(key=lambda pair: pair[0])
    total = sum(weight for _, weight in weighted)
    seen = 0
    for size, weight in weighted:
        seen += weight
        if seen * 2 >= total:
            return size
    return weighted[-1][0]


def _is_heading_candidate(block: Block, body_size: float, cfg: LayoutConfig) -> bool:
    if block.kind != "text" or not block.text.strip():
        return False
    if body_size > 0 and block.max_size >= body_size * cfg.heading_size_ratio:
        return True
    return (
        block.is_bold
        and block.line_count == 1
        and len(block.text.strip()) <= cfg.heading_max_chars
        and block.gap_below >= cfg.body_line_height * cfg.heading_gap_ratio
    )


def heading_levels(
    blocks: list[Block], body_size: float, cfg: LayoutConfig
) -> dict[int, int]:
    """Map block index -> heading level, consistent across the whole document."""
    candidates = [b for b in blocks if _is_heading_candidate(b, body_size, cfg)]
    if not candidates:
        return {}

    def bucket(size: float) -> float:
        return round(size * 2) / 2

    ranked = sorted({bucket(b.max_size) for b in candidates}, reverse=True)
    level_of = {size: min(rank + 1, MAX_HEADING_LEVEL) for rank, size in enumerate(ranked)}
    return {b.index: level_of[bucket(b.max_size)] for b in candidates}


def _y_coverage(blocks: list[Block]) -> float:
    """Length of the union of the blocks' vertical extents."""
    intervals = sorted((b.bbox.y0, b.bbox.y1) for b in blocks)
    covered = 0.0
    current_start, current_end = None, None
    for start, end in intervals:
        if current_end is None or start > current_end:
            if current_end is not None:
                covered += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        covered += current_end - current_start
    return covered


def find_gutters(blocks: list[Block], cfg: LayoutConfig) -> list[float]:
    """Centres of vertical whitespace bands that separate text columns."""
    positioned = [b for b in blocks if b.bbox.width > 0]
    if len(positioned) < 4:
        return []

    text_height = max(b.bbox.y1 for b in positioned) - min(b.bbox.y0 for b in positioned)
    if text_height <= 0:
        return []

    x_min = min(b.bbox.x0 for b in positioned)
    x_max = max(b.bbox.x1 for b in positioned)
    bin_count = int(math.ceil(x_max - x_min)) + 1
    occupied = [False] * bin_count
    for b in positioned:
        start = max(0, int(b.bbox.x0 - x_min))
        end = min(bin_count, int(math.ceil(b.bbox.x1 - x_min)))
        for i in range(start, end):
            occupied[i] = True

    gutters: list[float] = []
    i = 0
    while i < bin_count:
        if occupied[i]:
            i += 1
            continue
        j = i
        while j < bin_count and not occupied[j]:
            j += 1
        if (j - i) >= cfg.gutter_min_width and i > 0 and j < bin_count:
            centre = x_min + (i + j) / 2
            left = [b for b in positioned if b.bbox.x1 <= centre]
            right = [b for b in positioned if b.bbox.x0 >= centre]
            minimum = cfg.gutter_min_coverage * text_height
            if (
                left
                and right
                and _y_coverage(left) >= minimum
                and _y_coverage(right) >= minimum
            ):
                gutters.append(round(centre, 2))
        i = j

    return gutters


def column_index(bbox: BBox, gutters: list[float]) -> int:
    return sum(1 for gutter in gutters if bbox.x0 >= gutter)


ZONE_RANK: dict[ElementType, int] = {
    ElementType.HEADER: 0,
    ElementType.FOOTER: 2,
    ElementType.PAGE_NUMBER: 2,
}


def reading_order(
    blocks: list[Block],
    gutters: list[float],
    zones: dict[int, ElementType] | None = None,
) -> list[Block]:
    """Column-aware ordering: down each column in turn, else top to bottom.

    `zones` maps block index -> margin element type. Running headers sort
    ahead of the body and footers behind it, so margin furniture keeps a
    natural position without polluting the body's column order.
    """
    zones = zones or {}
    return sorted(
        blocks,
        key=lambda b: (
            b.page_number,
            ZONE_RANK.get(zones.get(b.index), 1),
            column_index(b.bbox, gutters),
            b.bbox.y0,
            b.bbox.x0,
        ),
    )


def _normalize(text: str) -> str:
    collapsed = " ".join(text.split()).lower()
    return re.sub(r"\d+", "#", collapsed)


def margin_blocks(
    blocks: list[Block],
    page_height: float,
    page_count: int,
    cfg: LayoutConfig,
) -> dict[int, ElementType]:
    """Identify running headers, footers and page numbers.

    A block qualifies by sitting wholly inside the top or bottom margin band
    AND either matching a page-number pattern or repeating across pages.
    """
    band = page_height * cfg.margin_band_ratio
    threshold = max(2, math.ceil(page_count * cfg.repeat_page_ratio))

    in_band: list[tuple[Block, ElementType]] = []
    for block in blocks:
        if block.kind != "text" or not block.text.strip():
            continue
        if block.bbox.y1 <= band:
            in_band.append((block, ElementType.HEADER))
        elif block.bbox.y0 >= page_height - band:
            in_band.append((block, ElementType.FOOTER))

    counts = Counter(_normalize(block.text) for block, _ in in_band)

    result: dict[int, ElementType] = {}
    for block, zone in in_band:
        stripped = block.text.strip()
        if PAGE_NUMBER_PATTERN.fullmatch(stripped):
            result[block.index] = ElementType.PAGE_NUMBER
        elif counts[_normalize(block.text)] >= threshold:
            result[block.index] = zone
    return result


def pair_captions(
    figures: list[Block],
    text_blocks: list[Block],
    body_size: float,
    cfg: LayoutConfig,
) -> dict[int, int]:
    """Map figure block index -> caption block index.

    Searched below first, then above. A `Figure 3: ...` style label always
    outranks a merely-smaller block; ties break toward the closer block.
    Each caption is claimed by at most one figure.
    """
    claimed: set[int] = set()
    paired: dict[int, int] = {}

    for figure in sorted(figures, key=lambda f: (f.page_number, f.bbox.y0)):
        best_key: tuple[int, float, int] | None = None
        best_index: int | None = None

        for below in (True, False):
            for candidate in text_blocks:
                if candidate.page_number != figure.page_number:
                    continue
                if candidate.index in claimed or candidate.kind != "text":
                    continue
                stripped = candidate.text.strip()
                if not stripped or len(stripped) > cfg.caption_max_chars:
                    continue
                if (
                    candidate.bbox.horizontal_overlap_ratio(figure.bbox)
                    < cfg.caption_min_overlap
                ):
                    continue

                if below:
                    distance = candidate.bbox.y0 - figure.bbox.y1
                else:
                    distance = figure.bbox.y0 - candidate.bbox.y1
                if distance < 0 or distance > cfg.caption_max_distance:
                    continue

                is_labelled = bool(CAPTION_PATTERN.match(stripped))
                is_smaller = 0 < candidate.max_size < body_size
                if not (is_labelled or is_smaller):
                    continue

                key = (0 if is_labelled else 1, distance, candidate.index)
                if best_key is None or key < best_key:
                    best_key, best_index = key, candidate.index

            if best_index is not None:
                break

        if best_index is not None:
            paired[figure.index] = best_index
            claimed.add(best_index)

    return paired
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/parsing/test_layout.py -v`
Expected: PASS, 22 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/parsing/layout.py tests/parsing/test_layout.py
git status --short
```

---

## Task 7: Synthetic PDF fixtures

**Files:**
- Create: `tests/fixtures/__init__.py`, `tests/fixtures/generate.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore` (nothing to ignore — fixtures are committed; verify no rule excludes `*.pdf`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build_all(out_dir: Path | None = None) -> dict[str, Path]` returning keys `simple`, `two_column`, `table_figure`, `diagram_png`, `sheet`; pytest session fixture `fixtures` yielding that dict.

Fixtures are authored in code so ground truth for element type, reading
order and text is exact. reportlab's `invariant` flag suppresses the
creation timestamp and randomised object ids, so fixture bytes — and
therefore content-addressed document ids — are stable across runs.

- [ ] **Step 1: Write the failing test**

Create `tests/fixtures/__init__.py` (empty file), then `tests/parsing/test_fixtures.py`:

```python
from __future__ import annotations

import hashlib


def test_all_fixtures_are_generated(fixtures):
    assert set(fixtures) == {"simple", "two_column", "table_figure", "diagram_png", "sheet"}
    for path in fixtures.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_generated_pdfs_are_byte_stable(tmp_path):
    from tests.fixtures.generate import build_all

    first = build_all(tmp_path / "a")
    second = build_all(tmp_path / "b")

    for key in first:
        digest_a = hashlib.sha256(first[key].read_bytes()).hexdigest()
        digest_b = hashlib.sha256(second[key].read_bytes()).hexdigest()
        assert digest_a == digest_b, f"{key} is not deterministic"


def test_pdf_fixtures_start_with_the_pdf_magic(fixtures):
    for key in ("simple", "two_column", "table_figure"):
        assert fixtures[key].read_bytes().startswith(b"%PDF-")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_fixtures.py -v`
Expected: FAIL — `fixture 'fixtures' not found`

- [ ] **Step 3: Implement `tests/fixtures/generate.py`**

```python
"""Builds the synthetic fixture files used by the parser tests.

Authored in code so ground truth is exact, and deterministic so that
content-addressed document ids stay stable across runs.

Run directly to regenerate: `uv run python -m tests.fixtures.generate`
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from PIL import Image, ImageDraw
from reportlab import rl_config
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas

__all__ = ["build_all", "FIXTURE_DIR", "PAGE_WIDTH", "PAGE_HEIGHT"]

FIXTURE_DIR = Path(__file__).parent / "files"
PAGE_WIDTH, PAGE_HEIGHT = LETTER  # 612 x 792 points

BODY_LINES = [
    "The Payment API is responsible for processing payment requests.",
    "It validates each request, reserves funds, and writes an entry to",
    "the ledger before acknowledging the caller.",
]

DEPENDENCY_LINES = [
    "The Payment API depends on Redis for caching and on PostgreSQL",
    "for persistent storage. The Payments Team owns the service.",
]

COLUMN_LINES = [
    "Incident INC-2391 affected the",
    "Payment API when Redis connection",
    "timeouts exceeded the configured",
    "threshold during peak traffic.",
    "The Payments Team resolved it by",
    "raising the connection pool size.",
]


def _canvas(path: Path) -> rl_canvas.Canvas:
    rl_config.invariant = 1
    return rl_canvas.Canvas(str(path), pagesize=LETTER, invariant=1)


def build_diagram_png(path: Path) -> Path:
    """A deterministic architecture diagram: three boxes joined by arrows."""
    image = Image.new("RGB", (480, 300), "white")
    draw = ImageDraw.Draw(image)

    boxes = {
        "API Gateway": (160, 20, 320, 70),
        "Payment Service": (160, 115, 320, 165),
        "Redis": (30, 220, 180, 270),
        "PostgreSQL": (300, 220, 450, 270),
    }
    for label, (x0, y0, x1, y1) in boxes.items():
        draw.rectangle((x0, y0, x1, y1), outline="black", width=2)
        draw.text((x0 + 12, y0 + 18), label, fill="black")

    draw.line((240, 70, 240, 115), fill="black", width=2)
    draw.line((200, 165, 105, 220), fill="black", width=2)
    draw.line((280, 165, 375, 220), fill="black", width=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return path


def build_simple(path: Path) -> Path:
    """One page: a title, a body paragraph, a subheading, and a running footer."""
    canvas = _canvas(path)

    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(72, 720, "Payment Architecture")

    canvas.setFont("Helvetica", 11)
    y = 686
    for line in BODY_LINES:
        canvas.drawString(72, y, line)
        y -= 14

    canvas.setFont("Helvetica-Bold", 14)
    canvas.drawString(72, y - 18, "Dependencies")

    canvas.setFont("Helvetica", 11)
    y -= 46
    for line in DEPENDENCY_LINES:
        canvas.drawString(72, y, line)
        y -= 14

    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(PAGE_WIDTH / 2, 40, "Knowledge OS Internal")

    canvas.showPage()
    canvas.save()
    return path


def build_two_column(path: Path) -> Path:
    """Two pages of two-column body text with a running header and page numbers."""
    canvas = _canvas(path)

    for page_number in (1, 2):
        canvas.setFont("Helvetica", 9)
        canvas.drawString(72, 750, "Payment Platform Handbook")

        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(72, 706, f"Section {page_number}")

        canvas.setFont("Helvetica", 11)
        for column_x in (72, 340):
            y = 670
            for line in COLUMN_LINES:
                canvas.drawString(column_x, y, line)
                y -= 14

        canvas.setFont("Helvetica", 9)
        canvas.drawCentredString(PAGE_WIDTH / 2, 40, str(page_number))
        canvas.showPage()

    canvas.save()
    return path


def build_table_figure(path: Path, diagram: Path) -> Path:
    """One page: heading, a ruled table, and a captioned diagram."""
    canvas = _canvas(path)

    canvas.setFont("Helvetica-Bold", 18)
    canvas.drawString(72, 720, "Service Inventory")

    rows = [
        ["Service", "Datastore", "Owner"],
        ["Payment API", "PostgreSQL", "Payments Team"],
        ["Ledger", "Redis", "Payments Team"],
        ["Gateway", "None", "Platform Team"],
    ]
    col_x = [72, 232, 372, 540]
    top = 690
    row_height = 22

    for row_index, row in enumerate(rows):
        y = top - row_index * row_height
        canvas.setFont("Helvetica-Bold" if row_index == 0 else "Helvetica", 10)
        for col_index, cell in enumerate(row):
            canvas.drawString(col_x[col_index] + 4, y - 15, cell)
        canvas.line(col_x[0], y, col_x[-1], y)

    bottom = top - len(rows) * row_height
    canvas.line(col_x[0], bottom, col_x[-1], bottom)
    for x in col_x:
        canvas.line(x, top, x, bottom)

    canvas.drawImage(
        ImageReader(str(diagram)), 120, 300, width=360, height=225, mask=None
    )

    canvas.setFont("Helvetica", 9)
    canvas.drawString(120, 286, "Figure 1: Payment service architecture")

    canvas.showPage()
    canvas.save()
    return path


def build_sheet(path: Path) -> Path:
    """A two-sheet workbook with headers, a merged title and a formula."""
    workbook = Workbook()

    services = workbook.active
    services.title = "Services"
    services["A1"] = "Service Inventory"
    services.merge_cells("A1:C1")
    services.append([])
    services.append(["Service", "Datastore", "Owner"])
    services.append(["Payment API", "PostgreSQL", "Payments Team"])
    services.append(["Ledger", "Redis", "Payments Team"])

    incidents = workbook.create_sheet("Incidents")
    incidents.append(["Incident", "Service", "Hours"])
    incidents.append(["INC-2391", "Payment API", 4])
    incidents.append(["INC-2402", "Ledger", 2])
    incidents["C4"] = "=SUM(C2:C3)"

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


def build_all(out_dir: Path | None = None) -> dict[str, Path]:
    target = Path(out_dir) if out_dir is not None else FIXTURE_DIR
    target.mkdir(parents=True, exist_ok=True)

    diagram = build_diagram_png(target / "diagram.png")
    return {
        "simple": build_simple(target / "simple.pdf"),
        "two_column": build_two_column(target / "two_column.pdf"),
        "table_figure": build_table_figure(target / "table_figure.pdf", diagram),
        "diagram_png": diagram,
        "sheet": build_sheet(target / "inventory.xlsx"),
    }


if __name__ == "__main__":
    for name, built in build_all().items():
        print(f"{name}: {built}")
```

- [ ] **Step 4: Implement `tests/conftest.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.generate import build_all


@pytest.fixture(scope="session")
def fixtures() -> dict[str, Path]:
    """Regenerate the synthetic fixture files once per test session."""
    return build_all()
```

- [ ] **Step 5: Generate the fixtures and run the tests**

```bash
uv run python -m tests.fixtures.generate
uv run pytest tests/parsing/test_fixtures.py -v
```

Expected: five paths printed, then PASS, 3 tests.

- [ ] **Step 6: Confirm the fixtures are not gitignored**

```bash
git check-ignore -v tests/fixtures/files/simple.pdf || echo "not ignored - good"
```

Expected: `not ignored - good`. If a rule matches, add
`!tests/fixtures/files/` to `.gitignore`.

- [ ] **Step 7: Stage the changes (do not commit)**

```bash
git add tests/fixtures tests/conftest.py tests/parsing/test_fixtures.py .gitignore
git status --short
```

---

## Task 8: PDF parser

**Files:**
- Create: `app/parsing/pdf.py`
- Create: `tests/parsing/test_pdf.py`

**Interfaces:**
- Consumes: everything from Tasks 1, 2, 4, 6; the `fixtures` session fixture from Task 7.
- Produces: `PdfParser(cfg: LayoutConfig | None = None)` with `extensions = frozenset({".pdf"})`, `version = "1"`, `parse(path, *, blobs) -> Document`.

Pipeline order matters. Tables are located first so their bboxes can mask
the text spans inside them; margin blocks are identified before gutter
detection, because a centred page number sitting in the gutter would
otherwise hide the column split.

- [ ] **Step 1: Write the failing test**

Create `tests/parsing/test_pdf.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_pdf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.parsing.pdf'`

- [ ] **Step 3: Implement `app/parsing/pdf.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable

import fitz  # PyMuPDF
import pdfplumber

from app.ir.ids import blob_ref, element_id
from app.ir.model import (
    BBox,
    Document,
    Element,
    ElementType,
    ImageRef,
    Page,
    TableData,
)
from app.ir.tree import assign_parents
from app.parsing.base import BlobSink, ParseError, source_identity
from app.parsing.layout import (
    Block,
    LayoutConfig,
    body_font_size,
    column_index,
    find_gutters,
    heading_levels,
    margin_blocks,
    pair_captions,
    reading_order,
)

__all__ = ["PdfParser"]

BOLD_FLAG = 1 << 4


@dataclass
class _TableRegion:
    bbox: BBox
    table: TableData


class _Counter:
    """Hands out document-wide unique block indices."""

    def __init__(self) -> None:
        self._value = -1

    def next(self) -> int:
        self._value += 1
        return self._value


def _to_table_data(rows: list[list[Any]]) -> TableData | None:
    cleaned = [
        [("" if cell is None else str(cell)).strip() for cell in row] for row in rows
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return None

    n_cols = max(len(row) for row in cleaned)
    padded = [row + [""] * (n_cols - len(row)) for row in cleaned]

    headers: list[str] = []
    if len(padded) >= 2 and all(cell for cell in padded[0]):
        headers, padded = padded[0], padded[1:]

    return TableData(
        headers=headers,
        rows=padded,
        n_rows=len(padded),
        n_cols=n_cols,
    )


class PdfParser:
    """PyMuPDF for text, geometry and images; pdfplumber for table structure.

    Both libraries report coordinates with a top-left origin and y increasing
    downward, so their boxes are directly comparable for unrotated pages. A
    rotated page is parsed anyway but records a warning.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})
    version: ClassVar[str] = "1"

    def __init__(self, cfg: LayoutConfig | None = None) -> None:
        self.cfg = cfg or LayoutConfig()

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        short_id, checksum = source_identity(path)
        document = Document(
            id=short_id,
            source_name=path.name,
            mime="application/pdf",
            checksum=checksum,
            meta={"parser": "pdf", "parser_version": self.version},
        )

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:
            raise ParseError(f"cannot open {path} as a PDF: {exc}") from exc

        regions = self._find_tables(path, document)
        counter = _Counter()
        blocks_by_page: dict[int, list[Block]] = {}
        geometry: dict[int, tuple[float, float]] = {}

        try:
            with pdf:
                if pdf.page_count == 0:
                    raise ParseError(f"{path} contains no pages")
                for number, pdf_page in enumerate(pdf, start=1):
                    if pdf_page.rotation:
                        document.add_warning(
                            None, "rotated_page", f"page {number} rotation={pdf_page.rotation}"
                        )
                    geometry[number] = (
                        float(pdf_page.rect.width),
                        float(pdf_page.rect.height),
                    )
                    page_regions = regions.get(number, [])
                    page_blocks = self._text_blocks(pdf_page, number, page_regions, counter)
                    page_blocks += self._table_blocks(number, page_regions, counter)
                    page_blocks += self._image_blocks(
                        pdf, pdf_page, number, counter, blobs, document
                    )
                    blocks_by_page[number] = page_blocks
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"failed to parse {path}: {exc}") from exc

        self._assemble(document, blocks_by_page, geometry)
        return document

    # ---------- extraction ----------

    def _find_tables(self, path: Path, document: Document) -> dict[int, list[_TableRegion]]:
        regions: dict[int, list[_TableRegion]] = {}
        try:
            plumber = pdfplumber.open(str(path))
        except Exception as exc:
            document.add_warning(None, "table_scan_failed", str(exc))
            return regions

        with plumber:
            for number, page in enumerate(plumber.pages, start=1):
                found: list[_TableRegion] = []
                try:
                    candidates = page.find_tables()
                except Exception as exc:
                    document.add_warning(
                        None, "table_scan_failed", f"page {number}: {exc}"
                    )
                    continue

                for candidate in candidates:
                    try:
                        rows = candidate.extract()
                    except Exception as exc:
                        # Not masked, so the text survives as ordinary paragraphs.
                        document.add_warning(
                            None, "table_fallback", f"page {number}: {exc}"
                        )
                        continue
                    table = _to_table_data(rows)
                    if table is None:
                        document.add_warning(
                            None, "table_fallback", f"page {number}: no extractable cells"
                        )
                        continue
                    x0, top, x1, bottom = candidate.bbox
                    found.append(
                        _TableRegion(
                            bbox=BBox(x0=x0, y0=top, x1=x1, y1=bottom), table=table
                        )
                    )
                if found:
                    regions[number] = found
        return regions

    def _text_blocks(
        self,
        pdf_page: Any,
        page_number: int,
        regions: list[_TableRegion],
        counter: _Counter,
    ) -> list[Block]:
        collected: list[Block] = []
        for raw in pdf_page.get_text("dict").get("blocks", []):
            if raw.get("type") != 0:
                continue
            lines = raw.get("lines") or []
            spans = [span for line in lines for span in line.get("spans", [])]
            if not spans:
                continue
            text = "\n".join(
                "".join(span["text"] for span in line.get("spans", [])) for line in lines
            ).strip()
            if not text:
                continue

            x0, y0, x1, y1 = raw["bbox"]
            bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
            centre_x, centre_y = bbox.center
            # Masked: this text belongs to a table that was extracted structurally.
            if any(region.bbox.contains_point(centre_x, centre_y) for region in regions):
                continue

            collected.append(
                Block(
                    index=counter.next(),
                    page_number=page_number,
                    bbox=bbox,
                    text=text,
                    max_size=max(float(span["size"]) for span in spans),
                    is_bold=any(int(span.get("flags", 0)) & BOLD_FLAG for span in spans),
                    line_count=len(lines),
                    kind="text",
                )
            )

        collected.sort(key=lambda block: (block.bbox.y0, block.bbox.x0))
        for current, following in zip(collected, collected[1:]):
            current.gap_below = round(max(0.0, following.bbox.y0 - current.bbox.y1), 2)
        return collected

    def _table_blocks(
        self, page_number: int, regions: list[_TableRegion], counter: _Counter
    ) -> list[Block]:
        return [
            Block(
                index=counter.next(),
                page_number=page_number,
                bbox=region.bbox,
                text=region.table.as_text(),
                kind="table",
                attrs={"table": region.table},
            )
            for region in regions
        ]

    def _image_blocks(
        self,
        pdf: Any,
        pdf_page: Any,
        page_number: int,
        counter: _Counter,
        blobs: BlobSink,
        document: Document,
    ) -> list[Block]:
        collected: list[Block] = []
        for position, info in enumerate(pdf_page.get_images(full=True)):
            xref = info[0]
            rects = pdf_page.get_image_rects(xref)
            if not rects:
                document.add_warning(
                    None, "image_not_placed", f"xref {xref} on page {page_number}"
                )
                continue
            rect = rects[0]
            bbox = BBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

            try:
                extracted = pdf.extract_image(xref)
                ext = str(extracted.get("ext") or "png")
                ref = blob_ref(page_number, position, ext)
                blobs.add(ref, extracted["image"])
                image = ImageRef(
                    blob_ref=ref,
                    width=extracted.get("width"),
                    height=extracted.get("height"),
                    format=ext,
                )
            except Exception as exc:
                # Degrade: keep the element and its geometry, lose only the bytes.
                document.add_warning(
                    None, "image_decode_failed", f"xref {xref} on page {page_number}: {exc}"
                )
                image = ImageRef()

            collected.append(
                Block(
                    index=counter.next(),
                    page_number=page_number,
                    bbox=bbox,
                    text="",
                    kind="image",
                    attrs={"image": image},
                )
            )
        return collected

    # ---------- assembly ----------

    def _assemble(
        self,
        document: Document,
        blocks_by_page: dict[int, list[Block]],
        geometry: dict[int, tuple[float, float]],
    ) -> None:
        all_blocks = [block for blocks in blocks_by_page.values() for block in blocks]
        body_size = body_font_size(all_blocks, self.cfg)
        heading_map = heading_levels(all_blocks, body_size, self.cfg)

        page_height = geometry[min(geometry)][1] if geometry else 792.0
        margin_map = margin_blocks(
            all_blocks, page_height, len(blocks_by_page), self.cfg
        )

        figures = [block for block in all_blocks if block.kind in {"image", "table"}]
        caption_candidates = [
            block
            for block in all_blocks
            if block.kind == "text" and block.index not in margin_map
        ]
        caption_map = pair_captions(figures, caption_candidates, body_size, self.cfg)
        caption_owner = {caption: figure for figure, caption in caption_map.items()}

        order = 0
        element_by_block: dict[int, Element] = {}

        for page_number in sorted(blocks_by_page):
            width, height = geometry[page_number]
            page_blocks = blocks_by_page[page_number]
            body_blocks = [
                block
                for block in page_blocks
                if block.index not in margin_map and block.kind == "text"
            ]
            gutters = find_gutters(body_blocks, self.cfg)

            ordered = reading_order(page_blocks, gutters, margin_map)

            elements: list[Element] = []
            for position, block in enumerate(ordered):
                element = self._to_element(
                    block,
                    page_number,
                    position,
                    order,
                    heading_map,
                    margin_map,
                    caption_owner,
                )
                elements.append(element)
                element_by_block[block.index] = element
                order += 1

            document.pages.append(
                Page(number=page_number, width=width, height=height, elements=elements)
            )

        for figure_index, caption_index in caption_map.items():
            figure = element_by_block[figure_index]
            caption = element_by_block[caption_index]
            if figure.image is not None:
                figure.image.caption_id = caption.id
            else:
                figure.attrs["caption_id"] = caption.id
            caption.attrs["captions"] = figure.id

        assign_parents([e for page in document.pages for e in page.elements])

        document.meta["page_count"] = len(document.pages)
        document.title = next(
            (
                element.text
                for element in document.iter_elements()
                if element.type is ElementType.HEADING and element.level == 1 and element.text
            ),
            None,
        )

    def _to_element(
        self,
        block: Block,
        page_number: int,
        position: int,
        order: int,
        heading_map: dict[int, int],
        margin_map: dict[int, ElementType],
        caption_owner: dict[int, int],
    ) -> Element:
        level: int | None = None
        attrs: dict[str, Any] = {}

        if block.index in margin_map:
            element_type = margin_map[block.index]
        elif block.kind == "table":
            element_type = ElementType.TABLE
        elif block.kind == "image":
            element_type = ElementType.IMAGE
        elif block.index in caption_owner:
            element_type = ElementType.CAPTION
        elif block.index in heading_map:
            element_type = ElementType.HEADING
            level = heading_map[block.index]
        else:
            element_type = ElementType.PARAGRAPH

        if block.max_size:
            attrs["font_size"] = round(block.max_size, 2)
        if block.is_bold:
            attrs["bold"] = True

        return Element(
            id=element_id(page_number, position),
            type=element_type,
            page_number=page_number,
            bbox=block.bbox,
            order=order,
            level=level,
            text=None if block.kind in {"table", "image"} else block.text,
            table=block.attrs.get("table"),
            image=block.attrs.get("image"),
            attrs=attrs,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/parsing/test_pdf.py -v`
Expected: PASS, 16 tests.

If `test_two_column_pages_read_down_each_column` fails, print the detected
gutters to debug:

```bash
uv run python -c "
from app.parsing.pdf import PdfParser
from app.parsing.base import InMemoryBlobSink
from tests.fixtures.generate import build_all
doc = PdfParser().parse(build_all()['two_column'], blobs=InMemoryBlobSink())
for e in doc.pages[0].elements:
    print(e.order, e.type, round(e.bbox.x0), round(e.bbox.y0), repr((e.text or '')[:40]))
"
```

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/parsing/pdf.py tests/parsing/test_pdf.py
git status --short
```

---

## Task 9: Image and spreadsheet parsers

**Files:**
- Create: `app/parsing/image.py`, `app/parsing/spreadsheet.py`
- Create: `tests/parsing/test_image.py`, `tests/parsing/test_spreadsheet.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 4; the `fixtures` session fixture from Task 7.
- Produces: `ImageParser` with `extensions = frozenset({".png", ".jpg", ".jpeg"})`, `version = "1"`; `SpreadsheetParser` with `extensions = frozenset({".xlsx"})`, `version = "1"`; module constants `CELL_WIDTH = 80.0`, `ROW_HEIGHT = 18.0` in `spreadsheet.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/parsing/test_image.py`:

```python
from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, ParseError
from app.parsing.image import ImageParser


def test_a_standalone_image_becomes_one_page_one_element(fixtures):
    sink = InMemoryBlobSink()
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=sink)

    assert doc.page_count == 1
    assert len(doc.pages[0].elements) == 1

    element = doc.pages[0].elements[0]
    assert element.type is ElementType.IMAGE
    assert element.order == 0
    assert element.image.format == "png"
    assert (element.image.width, element.image.height) == (480, 300)


def test_the_bbox_covers_the_whole_page(fixtures):
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    page = doc.pages[0]
    element = page.elements[0]

    assert (element.bbox.x0, element.bbox.y0) == (0.0, 0.0)
    assert (element.bbox.x1, element.bbox.y1) == (page.width, page.height)


def test_the_original_bytes_are_stored_as_a_blob(fixtures):
    sink = InMemoryBlobSink()
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=sink)
    ref = doc.pages[0].elements[0].image.blob_ref

    assert sink.blobs[ref] == fixtures["diagram_png"].read_bytes()


def test_metadata_records_the_parser_and_mime(fixtures):
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    assert doc.mime == "image/png"
    assert doc.meta["parser"] == "image"
    assert doc.title == fixtures["diagram_png"].name


def test_parsing_is_deterministic(fixtures):
    from app.storage.base import serialize_document

    first = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    second = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    assert serialize_document(first) == serialize_document(second)


def test_an_undecodable_image_raises_parse_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    with pytest.raises(ParseError, match="cannot decode"):
        ImageParser().parse(path, blobs=InMemoryBlobSink())
```

Create `tests/parsing/test_spreadsheet.py`:

```python
from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, ParseError
from app.parsing.spreadsheet import SpreadsheetParser


@pytest.fixture(scope="module")
def workbook(fixtures):
    return SpreadsheetParser().parse(fixtures["sheet"], blobs=InMemoryBlobSink())


def test_each_sheet_becomes_a_page(workbook):
    assert workbook.page_count == 2
    assert [page.number for page in workbook.pages] == [1, 2]


def test_each_sheet_contributes_a_heading_and_a_table(workbook):
    first = workbook.pages[0].elements
    assert first[0].type is ElementType.HEADING
    assert first[0].text == "Services"
    assert first[1].type is ElementType.TABLE


def test_table_rows_and_headers_are_extracted(workbook):
    table = workbook.pages[1].elements[1].table
    assert table.headers == ["Incident", "Service", "Hours"]
    assert ["INC-2391", "Payment API", "4"] in table.rows


def test_merged_cells_repeat_their_value_across_the_range(workbook):
    table = workbook.pages[0].elements[1].table
    first_row = table.rows[0] if not table.headers else table.headers
    assert "Service Inventory" in (first_row + [cell for row in table.rows for cell in row])


def test_formulas_are_recorded_alongside_their_cached_value(workbook):
    element = workbook.pages[1].elements[1]
    assert element.attrs["formulas"] == {"C4": "=SUM(C2:C3)"}


def test_sheet_names_are_recorded_in_metadata(workbook):
    assert workbook.meta["sheet_names"] == ["Services", "Incidents"]
    assert workbook.meta["parser"] == "spreadsheet"


def test_cells_carry_synthetic_grid_geometry(workbook):
    table_element = workbook.pages[0].elements[1]
    assert table_element.attrs["synthetic_bbox"] is True
    assert table_element.bbox.x1 > table_element.bbox.x0
    assert table_element.bbox.y1 > table_element.bbox.y0


def test_parsing_is_deterministic(fixtures):
    from app.storage.base import serialize_document

    first = SpreadsheetParser().parse(fixtures["sheet"], blobs=InMemoryBlobSink())
    second = SpreadsheetParser().parse(fixtures["sheet"], blobs=InMemoryBlobSink())
    assert serialize_document(first) == serialize_document(second)


def test_a_non_workbook_raises_parse_error(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"PK\x03\x04 not a workbook")
    with pytest.raises(ParseError):
        SpreadsheetParser().parse(path, blobs=InMemoryBlobSink())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/parsing/test_image.py tests/parsing/test_spreadsheet.py -v`
Expected: FAIL — `ModuleNotFoundError` for both modules.

- [ ] **Step 3: Implement `app/parsing/image.py`**

```python
from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from app.ir.ids import blob_ref, element_id
from app.ir.model import BBox, Document, Element, ElementType, ImageRef, Page
from app.parsing.base import BlobSink, ParseError, source_identity

__all__ = ["ImageParser"]

_MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg"}


class ImageParser:
    """A standalone image becomes a one-page, one-element document.

    The uniform shape means a screenshot flows through the same retrieval and
    extraction pipeline as a PDF page, with no special-casing downstream.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg"})
    version: ClassVar[str] = "1"

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        data = path.read_bytes()
        short_id, checksum = source_identity(path)

        try:
            with PILImage.open(io.BytesIO(data)) as image:
                width, height = image.size
                image_format = (image.format or "PNG").upper()
        except (UnidentifiedImageError, OSError) as exc:
            raise ParseError(f"cannot decode {path} as an image: {exc}") from exc

        extension = "jpg" if image_format == "JPEG" else image_format.lower()
        ref = blob_ref(1, 0, extension)
        blobs.add(ref, data)

        element = Element(
            id=element_id(1, 0),
            type=ElementType.IMAGE,
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=float(width), y1=float(height)),
            order=0,
            image=ImageRef(
                blob_ref=ref, width=width, height=height, format=extension
            ),
            attrs={"standalone": True},
        )

        return Document(
            id=short_id,
            source_name=path.name,
            mime=_MIME_BY_FORMAT.get(image_format, "application/octet-stream"),
            checksum=checksum,
            title=path.name,
            meta={
                "page_count": 1,
                "parser": "image",
                "parser_version": self.version,
                "image_format": image_format,
            },
            pages=[
                Page(
                    number=1,
                    width=float(width),
                    height=float(height),
                    elements=[element],
                )
            ],
        )
```

- [ ] **Step 4: Implement `app/parsing/spreadsheet.py`**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

from app.ir.ids import element_id
from app.ir.model import BBox, Document, Element, ElementType, Page, TableData
from app.ir.tree import assign_parents
from app.parsing.base import BlobSink, ParseError, source_identity

__all__ = ["SpreadsheetParser", "CELL_WIDTH", "ROW_HEIGHT"]

CELL_WIDTH = 80.0
ROW_HEIGHT = 18.0
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


class SpreadsheetParser:
    """One page per worksheet: a HEADING for the sheet name, then one TABLE.

    Spreadsheets have no page geometry, so bboxes are synthetic grid
    coordinates and every element is tagged `synthetic_bbox`.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".xlsx"})
    version: ClassVar[str] = "1"

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        short_id, checksum = source_identity(path)

        try:
            values_book = load_workbook(path, data_only=True, read_only=False)
            formula_book = load_workbook(path, data_only=False, read_only=False)
        except Exception as exc:
            raise ParseError(f"cannot open {path} as a workbook: {exc}") from exc

        document = Document(
            id=short_id,
            source_name=path.name,
            mime=XLSX_MIME,
            checksum=checksum,
            title=path.stem,
            meta={
                "parser": "spreadsheet",
                "parser_version": self.version,
                "sheet_names": list(values_book.sheetnames),
            },
        )

        order = 0
        for page_number, sheet_name in enumerate(values_book.sheetnames, start=1):
            values_sheet = values_book[sheet_name]
            formula_sheet = formula_book[sheet_name]

            grid = self._grid(values_sheet)
            formulas = self._formulas(formula_sheet)
            table = self._table(grid)

            n_cols = max(table.n_cols, len(table.headers), 1)
            n_rows = max(table.n_rows, 1)

            heading = Element(
                id=element_id(page_number, 0),
                type=ElementType.HEADING,
                page_number=page_number,
                bbox=BBox(x0=0, y0=0, x1=CELL_WIDTH * n_cols, y1=ROW_HEIGHT),
                order=order,
                level=1,
                text=sheet_name,
                attrs={"synthetic_bbox": True},
            )
            order += 1

            table_element = Element(
                id=element_id(page_number, 1),
                type=ElementType.TABLE,
                page_number=page_number,
                bbox=BBox(
                    x0=0,
                    y0=ROW_HEIGHT,
                    x1=CELL_WIDTH * n_cols,
                    y1=ROW_HEIGHT * (n_rows + 2),
                ),
                order=order,
                table=table,
                attrs={"synthetic_bbox": True, "sheet": sheet_name, "formulas": formulas},
            )
            order += 1

            document.pages.append(
                Page(
                    number=page_number,
                    width=CELL_WIDTH * n_cols,
                    height=ROW_HEIGHT * (n_rows + 3),
                    elements=[heading, table_element],
                )
            )

        assign_parents([e for page in document.pages for e in page.elements])
        document.meta["page_count"] = len(document.pages)
        return document

    def _grid(self, sheet: Any) -> list[list[str]]:
        """Rows of stringified values, with merged ranges filled across."""
        rows = [[_stringify(cell) for cell in row] for row in sheet.iter_rows(values_only=True)]

        for merged in sorted(str(r) for r in sheet.merged_cells.ranges):
            bounds = sheet[merged]
            first = bounds[0][0]
            value = _stringify(first.value)
            if not value:
                continue
            for row in bounds:
                for cell in row:
                    r, c = cell.row - 1, cell.column - 1
                    if r < len(rows) and c < len(rows[r]):
                        rows[r][c] = value

        while rows and not any(cell for cell in rows[-1]):
            rows.pop()
        return rows

    def _formulas(self, sheet: Any) -> dict[str, str]:
        found: dict[str, str] = {}
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    found[f"{get_column_letter(cell.column)}{cell.row}"] = cell.value
        return dict(sorted(found.items()))

    def _table(self, grid: list[list[str]]) -> TableData:
        rows = [row for row in grid if any(cell for cell in row)]
        if not rows:
            return TableData()

        n_cols = max(len(row) for row in rows)
        padded = [row + [""] * (n_cols - len(row)) for row in rows]

        headers: list[str] = []
        for index, row in enumerate(padded):
            if all(cell for cell in row):
                headers = row
                padded = padded[index + 1 :]
                break

        return TableData(
            headers=headers,
            rows=padded,
            n_rows=len(padded),
            n_cols=n_cols,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/parsing/test_image.py tests/parsing/test_spreadsheet.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add app/parsing/image.py app/parsing/spreadsheet.py tests/parsing/test_image.py tests/parsing/test_spreadsheet.py
git status --short
```

---

## Task 10: Cross-cutting invariants and golden files

**Files:**
- Create: `tests/test_invariants.py`
- Create: `tests/test_golden.py`
- Create: `tests/golden/` (generated `.json` files, committed)
- Create: `scripts/regenerate_golden.py`

**Interfaces:**
- Consumes: every parser, `parse_document` (Task 4), `serialize_document` (Task 3), the `fixtures` fixture (Task 7).
- Produces: `scripts/regenerate_golden.py` writing `tests/golden/<name>.json`.

These are the tests that prove the design's central claim. The lossless
check compares a multiset of non-whitespace characters, so reflow,
re-ordering and whitespace normalization are allowed but dropping content
is not.

- [ ] **Step 1: Write the failing invariant tests**

Create `tests/test_invariants.py`:

```python
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
```

- [ ] **Step 2: Run the invariant tests to verify they fail**

Run: `uv run pytest tests/test_invariants.py -v`
Expected: FAIL — at minimum on missing `tests/golden` helpers is not involved
here, so any failure is a real invariant violation. Fix the parser, not the
test, unless the test encodes a genuinely wrong expectation.

- [ ] **Step 3: Implement `scripts/regenerate_golden.py`**

```python
"""Regenerate the committed golden IR files.

Intentional parser changes become a reviewable diff:
    uv run python -m scripts.regenerate_golden
    git diff tests/golden
"""

from __future__ import annotations

from pathlib import Path

from app.parsing.base import InMemoryBlobSink, parse_document
from app.storage.base import serialize_document
from tests.fixtures.generate import build_all

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    sources = build_all()
    sources["asset"] = Path("app/assets/payment_system.md")

    for name, path in sorted(sources.items()):
        document = parse_document(path, blobs=InMemoryBlobSink())
        target = GOLDEN_DIR / f"{name}.json"
        target.write_text(serialize_document(document), encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the golden test**

Create `tests/test_golden.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.base import InMemoryBlobSink, parse_document
from app.storage.base import serialize_document

GOLDEN_DIR = Path(__file__).parent / "golden"
NAMES = ["simple", "two_column", "table_figure", "diagram_png", "sheet", "asset"]


@pytest.mark.parametrize("name", NAMES)
def test_parsed_output_matches_the_golden_file(name, fixtures):
    golden = GOLDEN_DIR / f"{name}.json"
    assert golden.is_file(), (
        f"missing {golden}. Run: uv run python -m scripts.regenerate_golden"
    )

    source = (
        Path("app/assets/payment_system.md") if name == "asset" else fixtures[name]
    )
    actual = serialize_document(parse_document(source, blobs=InMemoryBlobSink()))

    assert actual == golden.read_text(encoding="utf-8"), (
        f"{name} output changed. If intended, run "
        "`uv run python -m scripts.regenerate_golden` and review the diff."
    )
```

- [ ] **Step 5: Generate the goldens and run the whole suite**

```bash
uv run python -m scripts.regenerate_golden
uv run pytest -v
```

Expected: six golden files written, then the full suite passes.

- [ ] **Step 6: Review the goldens by eye**

```bash
uv run python -c "
import json
data = json.load(open('tests/golden/table_figure.json'))
for page in data['pages']:
    for e in page['elements']:
        print(e['order'], e['type'], e['id'], repr((e['text'] or '')[:45]))
"
```

Expected: a heading, a table, an image and its caption, in reading order.
If anything is mistyped or out of order, that is a parser bug — fix it and
regenerate.

- [ ] **Step 7: Stage the changes (do not commit)**

```bash
git add tests/test_invariants.py tests/test_golden.py tests/golden scripts/regenerate_golden.py
git status --short
```

---

## Task 11: Ingestion entry point and README

**Files:**
- Create: `app/ingest.py`
- Create: `tests/test_ingest.py`
- Modify: `README.md` (append a "Document IR" section)

**Interfaces:**
- Consumes: `parse_document`, `InMemoryBlobSink` (Task 4); `FileDocumentStore` (Task 3).
- Produces: `ingest(path: Path | str, store: DocumentStore) -> Document`, `ingest_directory(directory: Path | str, store: DocumentStore) -> list[Document]`, and a `python -m app.ingest <path> [--store DIR]` CLI.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ingest.py`:

```python
from __future__ import annotations

import shutil

import pytest

from app.ingest import ingest, ingest_directory
from app.parsing.base import ParseError
from app.storage.filestore import FileDocumentStore


def test_ingest_parses_and_stores_in_one_call(tmp_path, fixtures):
    store = FileDocumentStore(tmp_path / "store")
    doc = ingest(fixtures["table_figure"], store)

    assert store.exists(doc.id)
    assert store.get(doc.id) == doc

    image = next(e for e in doc.iter_elements() if e.image and e.image.blob_ref)
    assert store.get_blob(doc.id, image.image.blob_ref)


def test_ingesting_the_same_file_twice_is_idempotent(tmp_path, fixtures):
    store = FileDocumentStore(tmp_path / "store")
    first = ingest(fixtures["simple"], store)
    second = ingest(fixtures["simple"], store)

    assert first.id == second.id
    assert list(store.list_ids()) == [first.id]


def test_ingest_directory_handles_mixed_formats(tmp_path, fixtures):
    source = tmp_path / "incoming"
    source.mkdir()
    for key in ("simple", "diagram_png", "sheet"):
        shutil.copy(fixtures[key], source / fixtures[key].name)
    shutil.copy("app/assets/payment_system.md", source / "payment_system.md")

    store = FileDocumentStore(tmp_path / "store")
    documents = ingest_directory(source, store)

    assert len(documents) == 4
    assert len(list(store.list_ids())) == 4
    assert {doc.mime for doc in documents} == {
        "application/pdf",
        "image/png",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/markdown",
    }


def test_ingest_directory_skips_unsupported_files_without_failing(tmp_path, fixtures):
    source = tmp_path / "incoming"
    source.mkdir()
    shutil.copy(fixtures["simple"], source / "doc.pdf")
    (source / "notes.rtf").write_text("unsupported")

    store = FileDocumentStore(tmp_path / "store")
    documents = ingest_directory(source, store)

    assert len(documents) == 1


def test_ingest_propagates_parse_errors_for_a_single_file(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 nonsense")
    store = FileDocumentStore(tmp_path / "store")

    with pytest.raises(ParseError):
        ingest(broken, store)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 3: Implement `app/ingest.py`**

```python
"""Parse files into the IR and persist them.

    uv run python -m app.ingest app/assets/payment_system.md
    uv run python -m app.ingest ./corpus --store ./store
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ir.model import Document
from app.parsing.base import InMemoryBlobSink, ParseError, parse_document
from app.storage.base import DocumentStore
from app.storage.filestore import FileDocumentStore

__all__ = ["ingest", "ingest_directory", "DEFAULT_STORE_DIR"]

DEFAULT_STORE_DIR = Path("store")


def ingest(path: Path | str, store: DocumentStore) -> Document:
    """Parse one file and persist it. Re-ingesting identical bytes is a no-op."""
    sink = InMemoryBlobSink()
    document = parse_document(path, blobs=sink)
    store.put(document, sink.blobs)
    return document


def ingest_directory(directory: Path | str, store: DocumentStore) -> list[Document]:
    """Ingest every supported file in a directory tree, skipping the rest."""
    documents: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        try:
            documents.append(ingest(path, store))
        except ParseError:
            continue
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse documents into the IR.")
    parser.add_argument("path", type=Path, help="file or directory to ingest")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE_DIR)
    args = parser.parse_args(argv)

    store = FileDocumentStore(args.store)
    targets = (
        ingest_directory(args.path, store)
        if args.path.is_dir()
        else [ingest(args.path, store)]
    )

    for document in targets:
        print(
            f"{document.id}  {document.source_name}  "
            f"{document.page_count} page(s)  "
            f"{sum(len(p.elements) for p in document.pages)} elements"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Exercise the CLI end to end**

```bash
uv run python -m app.ingest app/assets/payment_system.md --store /tmp/kos-store
ls /tmp/kos-store
```

Expected: one line of output with a document id, `1 page(s)` and an element
count; the store directory contains a single content-addressed folder.

- [ ] **Step 6: Document it in the README**

Append to `README.md`:

````markdown
## Document IR

The first layer of the platform converts source files into a lossless,
structured representation — text, hierarchy, tables, images, captions, page
locations and reading order — before any LLM extraction runs.

```
PDF / Markdown / PNG / XLSX
            ↓
      parse_document()        deterministic, no model calls
            ↓
        Document
          Page
          Element(type, bbox, order, parent_id, text|table|image)
            ↓
     FileDocumentStore        content-addressed JSON + blobs
```

Every element is addressable as `Provenance(doc_id, page_number,
element_id, bbox)`, so downstream chunks, extracted facts and graph edges
can cite a rectangle on a page rather than a filename.

### Usage

```bash
uv run python -m app.ingest app/assets/payment_system.md --store ./store
uv run python -m app.ingest ./corpus --store ./store
```

```python
from app.ingest import ingest
from app.storage.filestore import FileDocumentStore

store = FileDocumentStore("./store")
doc = ingest("Architecture.pdf", store)

for element in doc.iter_elements():
    print(element.order, element.type, element.provenance(doc.id))
```

### Tests

```bash
uv run pytest                                      # full suite
uv run python -m scripts.regenerate_golden         # after intended parser changes
```
````

- [ ] **Step 7: Run the complete suite one final time**

Run: `uv run pytest -q`
Expected: all tests pass, no warnings about missing fixtures.

- [ ] **Step 8: Stage the changes (do not commit)**

```bash
git add app/ingest.py tests/test_ingest.py README.md
git status --short
```

---

## Done

Sub-project 1 is complete when `uv run pytest -q` passes and
`uv run python -m app.ingest ./some-folder --store ./store` produces a
populated store. Sub-project 2 (evidence-grounded extraction) then refactors
`app/extraction/` to consume `Document` and attach `Provenance` to every
entity and relationship, replacing the current raw-string interface.

---

## Implementation notes: where the as-built code differs from this plan

Five changes were made during execution. Each was driven by a failing test or
an observed defect, not by preference.

**1. `section_tree` — preamble nodes must not adopt sections.**
The planned pop condition (`stack[-1].level >= level`) never pops a preamble
node, whose level is 0, so the document's first heading became a *child* of the
preamble instead of its sibling. Condition is now
`stack[-1].heading is None or stack[-1].level >= level`.
Caught by `test_section_tree_handles_content_before_any_heading`.

**2. `find_gutters` — coverage is measured between the columns, not against
the page.** The planned rule compared each side's covered height against the
full text height. A section heading sitting *above* the columns belongs to one
side only, and inflates the text height for both, so a genuine two-column page
with a heading failed the check. Replaced by `_columns_run_alongside`: the
sides' vertical spans must overlap by at least `gutter_min_coverage` of the
*shorter* side, and each side needs at least two blocks (which rejects a lone
stray element in the margin). `_y_coverage` was dropped; `_span` replaces it.
Regression tests: `test_a_heading_above_the_columns_does_not_defeat_gutter_detection`,
`test_one_stray_block_off_to_the_side_is_not_a_column`.

**3. The two-column fixture drew six consecutive lines per column**, which
PyMuPDF merges into a single text block — leaving three body blocks on the
page, below the four-block floor in `find_gutters`. The integration test was
therefore passing through the top-to-bottom fallback, not through column
detection. The fixture now draws three spaced paragraphs per column, and the
columns carry different text so ordering is actually observable.

**4. `SpreadsheetParser._table` lifts a merged full-width title row out.**
Left in place, `Service Inventory` (merged across A1:C1) satisfied the
"all cells non-empty" header rule and was taken as the header row, burying the
real column names in the data. It now returns `(TableData, banner)` and the
banner is stored as `attrs["table_title"]`.

**5. The `.xlsx` fixture was not byte-stable across sessions.** openpyxl
honours `properties.created` but overwrites `dcterms:modified` with the wall
clock at save time, and zip entries carry the current time — so the workbook's
content-addressed id drifted between runs and broke the golden test
intermittently. `_normalize_ooxml` now rewrites the archive with sorted
entries, a fixed zip timestamp, and a pinned `dcterms:modified`. Verified
identical across ten separate processes. `test_generated_fixtures_are_byte_stable`
was strengthened to compare against the *committed* fixtures, not only two
builds from the same run, since same-run builds share a wall clock and hide
exactly this class of bug.

**Other notes.**
- `MarkdownIt("gfm-like")` enables linkify, which needs `linkify-it-py`;
  disabled via `{"linkify": False}` rather than adding a dependency, since
  autolinking contributes nothing to the IR.
- `scripts/regenerate_golden.py` must be run as `-m scripts.regenerate_golden`.
  Invoked by path, `sys.path[0]` is `scripts/` and `import app` fails.
