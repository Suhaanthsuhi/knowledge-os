# Document IR + Parsing — Design Spec

Date: 2026-09-22
Status: Approved
Sub-project: 1 of 6

## Context

`knowledge-os` aims to be an enterprise multimodal knowledge infrastructure:
PDFs, markdown, images and spreadsheets become a connected, searchable,
citable knowledge layer.

The full vision spans six independent subsystems. Specing them together
would produce something too vague to implement, so it is decomposed:

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | Document IR + parsing | — |
| 2 | Evidence-grounded extraction (entities, relations, provenance, Neo4j) | 1 |
| 3 | Indexing + hybrid retrieval (chunking, embeddings, vector/keyword/graph, rerank) | 1, 2 |
| 4 | Multimodal RAG (orchestration, context assembly, cited answers) | 3 |
| 5 | Evaluation harness (benchmark, retrieval + generation metrics) | 3, 4 |
| 6 | Serving layer (API, async ingestion, tenancy, observability) | all |

**This spec covers sub-project 1 only.**

### Existing code

- `app/config.py` — Pydantic settings (Groq key, Neo4j credentials)
- `app/graph/neo4j.py` — entity/relationship upserts, candidate lookup
- `app/extraction/` — LLM `GraphExtractor`, LLM `EntityResolver`, flat schema
- `app/assets/payment_system.md` — a small domain fixture
- `playground/scratchpad.ipynb` — research notebook

None of it is modified by this sub-project. `app/extraction/` and
`app/graph/` are refactored in sub-project 2, when they start consuming IR
provenance instead of raw strings.

## Goal

Convert a source file into a lossless, structured representation of
everything meaningful in it — text, hierarchy, tables, images, captions,
page locations and reading order — **before** any LLM extraction happens.

Nothing downstream may re-parse the source file. The IR is the single
canonical representation.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Build order | Foundation-first | Everything downstream binds to the IR; retrofitting provenance means rewriting extraction and chunking |
| PDF stack | PyMuPDF + pdfplumber | ~50MB deps, sub-second per page, full coordinate fidelity, no model downloads |
| Model calls during parsing | None | Determinism, testability, zero cost on reparse |
| Persistence | JSON + blobs on disk behind a `DocumentStore` protocol | Inspectable, diffable, zero infra; Postgres/S3 slot in at sub-project 6 |
| IR shape | Flat elements + `parent_id`, tree derived on demand | Sections span pages; flat serializes cleanly and slices trivially for chunking |
| Fixtures | Synthetic PDFs generated with reportlab | Exact ground truth enables real assertions instead of smoke tests |

PyMuPDF is AGPL. Acceptable for this project; a permissive swap
(pypdfium2) would be needed for closed-source commercial distribution.
The `DocumentParser` protocol makes that swap local to `app/parsing/pdf.py`.

## Module layout

```
app/
  config.py                 # existing, untouched
  ir/
    model.py                # Document, Page, Element, BBox, TableData, ImageRef, Provenance
    ids.py                  # deterministic id generation
    tree.py                 # section-tree derivation, element index, iteration helpers
  parsing/
    base.py                 # DocumentParser protocol, BlobSink, dispatch
    layout.py               # pure functions: heading detection, reading order, caption pairing
    pdf.py                  # PyMuPDF text/images/coords + pdfplumber tables
    markdown.py             # markdown -> elements (synthetic bboxes)
    image.py                # standalone PNG/JPEG -> single-element document
    spreadsheet.py          # openpyxl -> one TABLE element per sheet
  storage/
    base.py                 # DocumentStore protocol
    filestore.py            # FileDocumentStore
tests/
  fixtures/generate.py      # reportlab synthetic PDF builder
  ir/ parsing/ storage/
```

## Data model

Coordinates: PDF points, origin top-left, y increasing downward. Page
width and height are stored so normalization to 0..1 is always possible.

```python
class BBox(BaseModel):
    x0: float; y0: float; x1: float; y1: float

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
    headers: list[str]
    rows: list[list[str]]
    n_rows: int
    n_cols: int
    cell_bboxes: list[list[BBox | None]] | None

class ImageRef(BaseModel):
    blob_ref: str | None        # None when the image failed to decode
    width: int | None
    height: int | None
    format: str | None
    caption_id: str | None

class Provenance(BaseModel):
    doc_id: str
    page_number: int
    element_id: str
    bbox: BBox

class Element(BaseModel):
    id: str                     # "p12e007", stable and scoped to the document
    type: ElementType
    page_number: int
    bbox: BBox
    order: int                  # global reading order across the document
    parent_id: str | None       # nearest enclosing HEADING element
    text: str | None
    level: int | None           # heading depth, 1-based
    table: TableData | None
    image: ImageRef | None
    attrs: dict[str, Any]       # font size/weight, column index, synthetic_bbox, parser hints

class Page(BaseModel):
    number: int; width: float; height: float
    elements: list[Element]

class Document(BaseModel):
    id: str                     # sha256(file bytes)[:16], content-addressed
    source_name: str
    mime: str
    checksum: str               # full sha256
    title: str | None
    meta: dict[str, Any]        # page_count, producer, parser, parser_version, warnings
    pages: list[Page]
```

### Properties this buys

**Provenance is a single value.** `element.provenance()` returns a
`Provenance`. Every chunk, extracted fact and KG edge in later
sub-projects carries one, so an answer cites a rectangle on a page rather
than a filename.

**Determinism.** No model calls, no timestamps, no randomness. Parsing the
same bytes twice produces a byte-identical `document.json`.

**Hierarchy without a tree.** `doc.section_tree()` derives heading
hierarchy on demand from `level` + `parent_id`; `doc.element(id)` is an
O(1) index lookup; `doc.iter_elements()` walks global reading order.
Storage stays flat.

## Parsing

### Dispatch

```python
class DocumentParser(Protocol):
    extensions: ClassVar[frozenset[str]]
    def parse(self, path: Path, *, blobs: BlobSink) -> Document: ...

def parse_document(path: Path, *, blobs: BlobSink | None = None) -> Document
```

Dispatch is by extension, falling back to magic-byte sniffing. Parsers
never write to storage; they receive a `BlobSink` for image bytes and
return a `Document`. Storage is the caller's decision, which keeps parsers
pure and testable without a filesystem.

### layout.py — shared pure functions

**Heading detection.** Compute the median font size and dominant weight
across the whole document body, not per page (per-page medians break on
title pages). A block is a heading candidate when either:

- its max span size >= `body_median * 1.15`, or
- it is bold, <= 80 characters, single-line, and followed by >= 0.5x
  body-line-height of vertical whitespace.

Candidate sizes are rounded to 0.5pt, clustered, sorted descending, and
assigned levels 1..n by rank, so `level` is consistent document-wide.
These constants live in one `LayoutConfig` dataclass rather than scattered
as literals, so they are tunable and visible in one place.

**Reading order.** Column-aware. Project block x-ranges onto the
horizontal axis; a gutter is an unoccupied x-band >= 18pt wide whose
vertical extent covers >= 60% of the page's text height. With gutters
present, sort by `(column_index, y0, x0)`; otherwise by `(y0, x0)`.
Headers, footers and page numbers are blocks lying wholly within the top
or bottom 8% of page height whose normalized text repeats on >= 50% of
pages (or matches a bare-number pattern); they are typed accordingly and
excluded from body reading order, but retained as elements.

**Caption pairing.** For each image and table, search within 40pt below,
then within 40pt above, for a text block that is <= 200 characters,
horizontally overlapping the figure by >= 50% of the block's width, and
either smaller than the body median or matching
`^(Figure|Fig\.|Table|Chart)\s*\d+`. A pattern match always outranks a
size match; ties break toward the closer block, then toward below. Each
caption pairs with at most one figure. The winner is typed `CAPTION` and
linked both ways via `image.caption_id` and the caption's
`attrs["captions"]`.

**Table masking.** Table bboxes mask PyMuPDF text spans so table text is
not also emitted as loose paragraphs. Without this, every table is indexed
twice — once as structure, once as text soup.

### Parsers

- **pdf.py** — PyMuPDF `get_text("dict")` for spans, `get_images` +
  `extract_image` for blobs, pdfplumber for table structure. Order: table
  regions, then text with those regions masked, then images, then
  `layout.py` assigns types, order and parents.
- **markdown.py** — markdown-it token stream to elements. No real
  geometry, so bboxes are synthetic: one continuous page with monotonic y
  derived from token position, and `attrs["synthetic_bbox"] = True` so
  downstream code distinguishes "position unknown" from a real rectangle.
  Lands first — simplest complete path through the model, and it makes
  `app/assets/payment_system.md` parse into real IR early.
- **image.py** — a standalone PNG/JPEG becomes a one-page, one-element
  document whose bbox is the full image, so a screenshot flows through the
  same pipeline as a PDF page.
- **spreadsheet.py** — openpyxl. One page per sheet, sheet name as a
  `HEADING`, one `TABLE` element per sheet, merged cells expanded,
  formulas stored as both cached value and formula text in `attrs`.

## Storage

```python
class DocumentStore(Protocol):
    def put(self, doc: Document, blobs: Mapping[str, bytes]) -> str
    def get(self, doc_id: str) -> Document
    def get_blob(self, doc_id: str, ref: str) -> bytes
    def exists(self, doc_id: str) -> bool
    def list_ids(self) -> Iterator[str]
```

`FileDocumentStore` writes:

```
store/<doc_id>/document.json
store/<doc_id>/blobs/<ref>
```

Because `doc_id` is the content hash, re-ingesting an identical file is a
no-op — idempotency comes free rather than needing a dedup pass. JSON is
written with sorted keys and fixed float formatting so golden files diff
cleanly.

## Failure handling

Degrade per element, fail loud per document.

- A file that cannot be opened raises `ParseError`. No half-documents.
- A table whose structure extraction fails degrades to a `PARAGRAPH` of
  its raw text plus a structured warning.
- An image that cannot be decoded keeps its `IMAGE` element and bbox with
  `blob_ref=None` plus a warning.
- Content is never silently dropped. Every degradation appends to
  `doc.meta["warnings"]` with the element id and reason, so warnings are
  inspectable and assertable in tests.

## Testing

Test-driven: tests precede implementation. Three layers.

**Unit tests on layout.py** — pure functions over hand-built span lists.
Two columns with a gutter order correctly. A bold 18pt line among 11pt
body becomes `HEADING level=1`. A repeated bottom-margin line becomes
`FOOTER`, not a paragraph. `Figure 3: ...` beneath an image pairs to it.

**Golden-file tests per parser** — parse each fixture, compare to a
committed expected `document.json`. Regenerating goldens is one command,
so intentional changes are a reviewable diff.

**Invariants, asserted on every fixture:**

- *Lossless* — concatenated whitespace-normalized element text on a page
  contains every non-whitespace character `page.get_text()` returns. This
  is the guarantee the whole design exists for.
- *Deterministic* — parsing the same bytes twice yields identical JSON.
- *Referential* — element ids unique; `order` strictly increasing; every
  `parent_id` and `caption_id` resolves; every bbox within page bounds.
- *No double-counting* — table text appears in exactly one element.

### Fixtures

`tests/fixtures/generate.py` builds fixture PDFs with reportlab: a
two-column page, a heading hierarchy, a real table, an embedded
architecture diagram with a caption, and a repeated header/footer. Because
they are authored in code, ground truth for bbox, reading order and
element types is exact. Committed, small, deterministic, regenerable.
Built around the existing payment-system domain so they stay useful for
sub-projects 2-5.

## Dependencies

Runtime: `pymupdf`, `pdfplumber`, `markdown-it-py`, `openpyxl`, `pillow`.
Dev: `pytest`, `reportlab`.

## Out of scope

Vision-model description, OCR, chunking, embeddings, retrieval, entity or
relationship extraction, Neo4j changes, the API, async workers, tenancy,
and DOCX. These belong to sub-projects 2-6.

## Success criteria

1. `parse_document()` handles PDF, markdown, PNG/JPEG and XLSX, returning
   a `Document` in every case.
2. All four invariants hold on every fixture.
3. `app/assets/payment_system.md` round-trips through
   `FileDocumentStore` unchanged.
4. Every element is addressable by `Provenance` — doc, page, element,
   bbox.
5. Parsing performs zero network calls.
