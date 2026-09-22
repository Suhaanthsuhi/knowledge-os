# Evidence-Grounded Knowledge Graph + GraphRAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Document IR into a Neo4j knowledge graph whose every fact cites a specific page rectangle, and make GraphRAG answer questions from the UI.

**Architecture:** A pure chunker splits the IR by section; an LLM extracts entities and facts plus a verbatim evidence quote per fact; the quote is matched back to a specific element to bind provenance; a two-stage resolver deduplicates entities; the result is upserted through a `GraphStore` protocol (Neo4j in production, in-memory in tests). Retrieval fuses k-hop graph traversal with pure-Python BM25 via Reciprocal Rank Fusion, and answers cite element ids that are validated against reality before rendering.

**Tech Stack:** Python 3.13, Pydantic v2, neo4j driver, langchain-groq (`openai/gpt-oss-120b`), Streamlit.

**Spec:** `docs/superpowers/specs/2026-09-22-graph-knowledge-design.md`

## Global Constraints

- **Do not run `git commit`.** The user commits and pushes. Every task ends by staging with `git add` only.
- **Do not delete Neo4j data.** Reset is a UI action the user triggers. No task performs a destructive graph write.
- **The full suite must pass offline** — no Neo4j, no Groq. Every LLM call goes through the `StructuredLLM` protocol so tests inject `FakeLLM`; every graph call goes through `GraphStore` so tests use `InMemoryGraphStore`.
- Neo4j integration tests run only when `KOS_NEO4J_TESTS=1`; they tag what they create and delete exactly that.
- **Model output never reaches Cypher unvalidated.** Relation types are checked against `RelationType` before interpolation.
- `app/ir`, `app/parsing`, `app/storage` are **not modified**. This work consumes the IR; it does not change it.
- Type hints on every public function; `from __future__ import annotations` at the top of every module.
- Python `>=3.13`, `uv` for dependencies.

---

## File Structure

| File | Responsibility |
|---|---|
| `app/knowledge/schema.py` | `EntityType`, `RelationType`, `ExtractedEntity`, `ExtractedFact`, `ChunkExtraction`, `EntityMatch`, `Chunk`, `ResolvedEntity`, `ElementRef`, `DocumentRef`, `GraphFact`, `GraphUpsert`, `entity_key` |
| `app/knowledge/llm.py` | `StructuredLLM` protocol, `GroqLLM`, `FakeLLM` |
| `app/knowledge/chunker.py` | `chunk_document(doc) -> list[Chunk]` — pure, section-aware |
| `app/knowledge/extractor.py` | `KnowledgeExtractor.extract(chunk) -> ChunkExtraction` |
| `app/knowledge/resolver.py` | `EntityResolver.resolve(candidate, known) -> ResolvedEntity` |
| `app/knowledge/pipeline.py` | evidence binding + `KnowledgePipeline.run(doc, known) -> GraphUpsert` |
| `app/graph/store.py` | `GraphStore` protocol, `GraphStats`, `FactRecord` |
| `app/graph/memory.py` | `InMemoryGraphStore` |
| `app/graph/cypher.py` | statement builders, relation-type allowlist |
| `app/graph/neo4j.py` | `Neo4jGraphStore` (replaces `Neo4jClient`) |
| `app/rag/bm25.py` | `Passage`, `tokenize`, `Bm25Index` |
| `app/rag/retriever.py` | `Evidence`, `RetrievalResult`, `Retriever`, `QuestionAnalyzer`, `GraphRetriever`, `Bm25Retriever`, `HybridRetriever`, `reciprocal_rank_fusion` |
| `app/rag/answer.py` | `Answer`, `build_context`, `answer_question` |
| `app/viewer/services.py` | cached store/LLM/index constructors shared by modes |
| `app/viewer/modes/inspect.py` | existing IR viewer, moved |
| `app/viewer/modes/ingest.py` | parse → preview → commit → reset |
| `app/viewer/modes/ask.py` | question → answer, evidence, graph path, trace |
| `app/viewer/main.py` | sidebar mode routing only |

---

## Task 1: Knowledge schema and the LLM seam

**Files:**
- Create: `app/knowledge/__init__.py`, `app/knowledge/schema.py`, `app/knowledge/llm.py`
- Create: `tests/knowledge/__init__.py`, `tests/knowledge/test_schema.py`, `tests/knowledge/test_llm.py`

**Interfaces:**
- Consumes: `Document`, `Element`, `BBox` from `app.ir.model`.
- Produces: `EntityType`, `RelationType` (StrEnums); `entity_key(name: str, entity_type: EntityType) -> str`; `ExtractedEntity(name, type, aliases)`; `ExtractedFact(source, relation, target, evidence)`; `ChunkExtraction(entities, facts)`; `EntityMatch(match, matched_key)`; `Chunk(id, doc_id, page_number, heading, text, element_ids)`; `ResolvedEntity(key, name, type, aliases)`; `ElementRef(id, doc_id, type, page_number, x0, y0, x1, y1, order, text)` with `.global_id`; `DocumentRef(id, source_name, mime, title, checksum)`; `GraphFact(source_key, target_key, relation, doc_id, element_ids, evidence, confidence)`; `GraphUpsert(document, entities, facts, elements, warnings)` with `.add_warning(code, detail)`; `StructuredLLM` protocol with `structured(prompt: str, schema: type[T]) -> T`; `GroqLLM`; `FakeLLM`.

- [ ] **Step 1: Create the package directories**

```bash
mkdir -p app/knowledge app/rag app/viewer/modes tests/knowledge tests/graph tests/rag
touch app/knowledge/__init__.py app/rag/__init__.py app/viewer/modes/__init__.py
touch tests/knowledge/__init__.py tests/graph/__init__.py tests/rag/__init__.py
```

- [ ] **Step 2: Write the failing tests**

Create `tests/knowledge/test_schema.py`:

```python
from __future__ import annotations

import pytest

from app.knowledge.schema import (
    Chunk,
    DocumentRef,
    ElementRef,
    EntityType,
    ExtractedFact,
    GraphFact,
    GraphUpsert,
    RelationType,
    ResolvedEntity,
    entity_key,
)


def test_entity_key_is_normalized_and_type_scoped():
    assert entity_key("Payment API", EntityType.SERVICE) == "service|payment api"
    assert entity_key("  payment   API ", EntityType.SERVICE) == "service|payment api"
    assert entity_key("Payment-API!", EntityType.SERVICE) == "service|payment api"


def test_entity_key_separates_different_types():
    assert entity_key("Apple", EntityType.ORGANIZATION) != entity_key(
        "Apple", EntityType.PRODUCT
    )


def test_entity_key_rejects_an_empty_name():
    with pytest.raises(ValueError, match="empty entity name"):
        entity_key("   ", EntityType.SERVICE)


def test_relation_types_are_upper_snake_case():
    for relation in RelationType:
        assert relation.value == relation.value.upper()
        assert " " not in relation.value


def test_extracted_fact_rejects_an_unknown_relation():
    with pytest.raises(ValueError):
        ExtractedFact(
            source="a", relation="NOT_A_RELATION", target="b", evidence="quote"
        )


def test_element_ref_global_id_is_document_scoped():
    ref = ElementRef(
        id="p1e003",
        doc_id="abc123",
        type="paragraph",
        page_number=1,
        x0=1, y0=2, x1=3, y1=4,
        order=3,
        text="hello",
    )
    assert ref.global_id == "abc123:p1e003"


def test_chunk_is_hashable_and_carries_element_ids():
    chunk = Chunk(
        id="d:c000",
        doc_id="d",
        page_number=1,
        heading="Payments",
        text="body",
        element_ids=("p1e000", "p1e001"),
    )
    assert chunk.element_ids == ("p1e000", "p1e001")
    assert hash(chunk)


def test_graph_upsert_records_warnings():
    upsert = GraphUpsert(
        document=DocumentRef(
            id="d", source_name="d.md", mime="text/markdown", title=None, checksum="c"
        )
    )
    upsert.add_warning("chunk_failed", "chunk 3: timeout")
    assert upsert.warnings == [{"code": "chunk_failed", "detail": "chunk 3: timeout"}]


def test_graph_fact_defaults_to_high_confidence():
    fact = GraphFact(
        source_key="service|payment api",
        target_key="database|redis",
        relation=RelationType.DEPENDS_ON,
        doc_id="d",
        element_ids=["p1e002"],
        evidence="The Payment API depends on Redis.",
    )
    assert fact.confidence == "high"


def test_resolved_entity_merges_aliases_without_duplicates():
    entity = ResolvedEntity(
        key="service|payment api",
        name="Payment API",
        type=EntityType.SERVICE,
        aliases=["payments api"],
    )
    merged = entity.with_aliases(["Payments API", "payments api", "Payment API"])
    assert sorted(merged.aliases) == ["payments api"]
    assert merged.name == "Payment API"
```

Create `tests/knowledge/test_llm.py`:

```python
from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.knowledge.llm import FakeLLM


class Shape(BaseModel):
    value: str


def test_fake_llm_returns_queued_responses_in_order():
    llm = FakeLLM(responses=[Shape(value="one"), Shape(value="two")])
    assert llm.structured("a", Shape).value == "one"
    assert llm.structured("b", Shape).value == "two"


def test_fake_llm_records_every_prompt():
    llm = FakeLLM(responses=[Shape(value="one")])
    llm.structured("the prompt", Shape)
    assert llm.prompts == ["the prompt"]


def test_fake_llm_raises_when_exhausted():
    llm = FakeLLM(responses=[])
    with pytest.raises(AssertionError, match="no queued response"):
        llm.structured("a", Shape)


def test_fake_llm_can_answer_from_a_handler():
    llm = FakeLLM(handler=lambda prompt, schema: schema(value=prompt.upper()))
    assert llm.structured("hi", Shape).value == "HI"


def test_fake_llm_can_raise_a_queued_error():
    llm = FakeLLM(responses=[RuntimeError("upstream exploded")])
    with pytest.raises(RuntimeError, match="upstream exploded"):
        llm.structured("a", Shape)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/knowledge -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.knowledge.schema'`

- [ ] **Step 4: Implement `app/knowledge/schema.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "EntityType",
    "RelationType",
    "entity_key",
    "ExtractedEntity",
    "ExtractedFact",
    "ChunkExtraction",
    "EntityMatch",
    "QuestionEntities",
    "Chunk",
    "ResolvedEntity",
    "ElementRef",
    "DocumentRef",
    "GraphFact",
    "GraphUpsert",
]

_PUNCTUATION = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")


class EntityType(StrEnum):
    PERSON = "Person"
    TEAM = "Team"
    SERVICE = "Service"
    PRODUCT = "Product"
    DATABASE = "Database"
    TECHNOLOGY = "Technology"
    ORGANIZATION = "Organization"
    INCIDENT = "Incident"
    DOCUMENT = "Document"
    POLICY = "Policy"
    LOCATION = "Location"
    CONCEPT = "Concept"
    OTHER = "Other"


class RelationType(StrEnum):
    OWNS = "OWNS"
    DEPENDS_ON = "DEPENDS_ON"
    USES = "USES"
    AFFECTS = "AFFECTS"
    DOCUMENTED_BY = "DOCUMENTED_BY"
    BELONGS_TO = "BELONGS_TO"
    CREATED_BY = "CREATED_BY"
    RESOLVES = "RESOLVES"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    lowered = name.strip().lower()
    stripped = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def entity_key(name: str, entity_type: EntityType) -> str:
    """Stable identity for an entity. Makes MERGE idempotent across ingests."""
    normalized = normalize_name(name)
    if not normalized:
        raise ValueError(f"empty entity name: {name!r}")
    return f"{entity_type.value.lower()}|{normalized}"


class ExtractedEntity(BaseModel):
    name: str = Field(description="The entity's name exactly as written in the text")
    type: EntityType = Field(description="The entity's category")
    aliases: list[str] = Field(default_factory=list)


class ExtractedFact(BaseModel):
    source: str = Field(description="Name of the source entity")
    relation: RelationType = Field(description="How source relates to target")
    target: str = Field(description="Name of the target entity")
    evidence: str = Field(
        description="A short quote copied verbatim from the text that states this fact"
    )


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


class EntityMatch(BaseModel):
    match: bool = Field(description="Whether the new entity is the same as a candidate")
    matched_key: str | None = Field(
        default=None, description="The exact key of the matching candidate, else null"
    )


class QuestionEntities(BaseModel):
    names: list[str] = Field(
        default_factory=list, description="Entity names mentioned in the question"
    )


@dataclass(frozen=True)
class Chunk:
    """A unit of text handed to the extractor, with the elements it came from."""

    id: str
    doc_id: str
    page_number: int
    heading: str | None
    text: str
    element_ids: tuple[str, ...]


class ResolvedEntity(BaseModel):
    key: str
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)

    def with_aliases(self, extra: list[str]) -> ResolvedEntity:
        """Merge aliases, dropping any that restate the canonical name."""
        canonical = normalize_name(self.name)
        merged = {normalize_name(alias) for alias in [*self.aliases, *extra]}
        merged.discard(canonical)
        merged.discard("")
        return self.model_copy(update={"aliases": sorted(merged)})


class ElementRef(BaseModel):
    """The evidence anchor written to the graph."""

    id: str
    doc_id: str
    type: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    order: int
    text: str

    @property
    def global_id(self) -> str:
        return f"{self.doc_id}:{self.id}"


class DocumentRef(BaseModel):
    id: str
    source_name: str
    mime: str
    title: str | None = None
    checksum: str = ""


class GraphFact(BaseModel):
    source_key: str
    target_key: str
    relation: RelationType
    doc_id: str
    element_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: Literal["high", "low"] = "high"


class GraphUpsert(BaseModel):
    document: DocumentRef
    entities: list[ResolvedEntity] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    elements: list[ElementRef] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    def add_warning(self, code: str, detail: str) -> None:
        self.warnings.append({"code": code, "detail": detail})
```

- [ ] **Step 5: Implement `app/knowledge/llm.py`**

```python
from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

__all__ = ["StructuredLLM", "GroqLLM", "FakeLLM", "DEFAULT_MODEL"]

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "openai/gpt-oss-120b"


@runtime_checkable
class StructuredLLM(Protocol):
    """The single seam every LLM call goes through, so tests can run offline."""

    def structured(self, prompt: str, schema: type[T]) -> T: ...


class GroqLLM:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        from langchain_groq import ChatGroq

        from app.config import settings

        self.model = model
        self._llm = ChatGroq(model=model, api_key=api_key or settings.groq_api_key)

    def structured(self, prompt: str, schema: type[T]) -> T:
        return self._llm.with_structured_output(schema).invoke(prompt)


class FakeLLM:
    """Deterministic stand-in. Queue responses, or answer from a handler.

    A queued `Exception` is raised rather than returned, so failure paths are
    testable without patching.
    """

    def __init__(
        self,
        responses: list[Any] | None = None,
        handler: Callable[[str, type[BaseModel]], Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self.prompts: list[str] = []

    def structured(self, prompt: str, schema: type[T]) -> T:
        self.prompts.append(prompt)
        if self._handler is not None:
            return self._handler(prompt, schema)
        assert self._responses, f"no queued response for {schema.__name__}"
        reply = self._responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/knowledge -v`
Expected: PASS, 14 tests.

- [ ] **Step 7: Stage the changes (do not commit)**

```bash
git add app/knowledge app/rag/__init__.py app/viewer/modes/__init__.py tests/knowledge tests/graph/__init__.py tests/rag/__init__.py
git status --short
```

---

## Task 2: Section-aware chunker

**Files:**
- Create: `app/knowledge/chunker.py`
- Create: `tests/knowledge/test_chunker.py`

**Interfaces:**
- Consumes: `Document`, `Element`, `ElementType` from `app.ir.model`; `Chunk` from Task 1.
- Produces: `MAX_CHUNK_CHARS = 1500`, `SKIPPED_TYPES: frozenset[ElementType]`, `chunk_document(doc: Document, *, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]`.

Margin furniture (header, footer, page number) is excluded: it repeats on every
page and would flood the extractor with the same non-facts.

- [ ] **Step 1: Write the failing test**

Create `tests/knowledge/test_chunker.py`:

```python
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
    assert every_id == [element.id for element in body]  # nothing lost, nothing split
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/knowledge/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.knowledge.chunker'`

- [ ] **Step 3: Implement `app/knowledge/chunker.py`**

```python
from __future__ import annotations

from app.ir.model import Document, Element, ElementType
from app.ir.tree import SectionNode
from app.knowledge.schema import Chunk

__all__ = ["chunk_document", "MAX_CHUNK_CHARS", "SKIPPED_TYPES"]

MAX_CHUNK_CHARS = 1500

# Running headers, footers and page numbers repeat on every page. Feeding them
# to the extractor would produce the same non-facts once per page.
SKIPPED_TYPES = frozenset(
    {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_NUMBER}
)


def _content(node: SectionNode) -> list[Element]:
    return [
        element
        for element in node.elements
        if element.type not in SKIPPED_TYPES and element.text_content().strip()
    ]


def _flush(
    doc: Document,
    heading: str | None,
    batch: list[Element],
    counter: list[int],
) -> Chunk:
    chunk = Chunk(
        id=f"{doc.id}:c{counter[0]:03d}",
        doc_id=doc.id,
        page_number=batch[0].page_number,
        heading=heading,
        text="\n\n".join(element.text_content().strip() for element in batch),
        element_ids=tuple(element.id for element in batch),
    )
    counter[0] += 1
    return chunk


def _walk(
    doc: Document,
    node: SectionNode,
    max_chars: int,
    counter: list[int],
    out: list[Chunk],
) -> None:
    heading = node.title() or None
    batch: list[Element] = []
    size = 0

    for element in _content(node):
        length = len(element.text_content())
        # Never split an element: an oversized one becomes its own chunk.
        if batch and size + length > max_chars:
            out.append(_flush(doc, heading, batch, counter))
            batch, size = [], 0
        batch.append(element)
        size += length
        if size >= max_chars:
            out.append(_flush(doc, heading, batch, counter))
            batch, size = [], 0

    if batch:
        out.append(_flush(doc, heading, batch, counter))

    for child in node.children:
        _walk(doc, child, max_chars, counter, out)


def chunk_document(doc: Document, *, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split a document into extraction units that remember where they came from.

    Sections are the natural boundary: a heading gives the extractor context its
    paragraphs lack on their own.
    """
    chunks: list[Chunk] = []
    counter = [0]
    for node in doc.section_tree():
        _walk(doc, node, max_chars, counter, chunks)
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/knowledge/test_chunker.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/knowledge/chunker.py tests/knowledge/test_chunker.py
git status --short
```

---

## Task 3: Extractor and entity resolver

**Files:**
- Create: `app/knowledge/extractor.py`, `app/knowledge/resolver.py`
- Create: `tests/knowledge/test_extractor.py`, `tests/knowledge/test_resolver.py`

**Interfaces:**
- Consumes: `StructuredLLM`, `FakeLLM` (Task 1); `Chunk`, `ChunkExtraction`, `ExtractedEntity`, `EntityMatch`, `EntityType`, `ResolvedEntity`, `entity_key`, `normalize_name` (Task 1).
- Produces: `KnowledgeExtractor(llm)` with `.extract(chunk: Chunk) -> ChunkExtraction`; `EntityResolver(llm: StructuredLLM | None = None)` with `.resolve(candidate: ExtractedEntity, known: dict[str, ResolvedEntity]) -> ResolvedEntity`; `EXTRACTION_PROMPT`, `RESOLUTION_PROMPT`.

- [ ] **Step 1: Write the failing tests**

Create `tests/knowledge/test_extractor.py`:

```python
from __future__ import annotations

import pytest

from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.llm import FakeLLM
from app.knowledge.schema import (
    Chunk,
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    RelationType,
)


def _chunk(text: str = "The Payment API depends on Redis.") -> Chunk:
    return Chunk(
        id="d:c000",
        doc_id="d",
        page_number=1,
        heading="Payments",
        text=text,
        element_ids=("p1e001",),
    )


def _extraction() -> ChunkExtraction:
    return ChunkExtraction(
        entities=[
            ExtractedEntity(name="Payment API", type=EntityType.SERVICE),
            ExtractedEntity(name="Redis", type=EntityType.DATABASE),
        ],
        facts=[
            ExtractedFact(
                source="Payment API",
                relation=RelationType.DEPENDS_ON,
                target="Redis",
                evidence="The Payment API depends on Redis.",
            )
        ],
    )


def test_extract_returns_the_model_output():
    llm = FakeLLM(responses=[_extraction()])
    result = KnowledgeExtractor(llm).extract(_chunk())

    assert [entity.name for entity in result.entities] == ["Payment API", "Redis"]
    assert result.facts[0].relation is RelationType.DEPENDS_ON


def test_the_prompt_carries_the_heading_and_the_text():
    llm = FakeLLM(responses=[_extraction()])
    KnowledgeExtractor(llm).extract(_chunk())

    prompt = llm.prompts[0]
    assert "Payments" in prompt
    assert "The Payment API depends on Redis." in prompt


def test_the_prompt_lists_the_allowed_vocabularies():
    llm = FakeLLM(responses=[_extraction()])
    KnowledgeExtractor(llm).extract(_chunk())

    prompt = llm.prompts[0]
    assert "DEPENDS_ON" in prompt
    assert "Service" in prompt


def test_the_prompt_demands_verbatim_evidence():
    llm = FakeLLM(responses=[_extraction()])
    KnowledgeExtractor(llm).extract(_chunk())
    assert "verbatim" in llm.prompts[0].lower()


def test_an_empty_chunk_is_not_sent_to_the_model():
    llm = FakeLLM(responses=[])
    result = KnowledgeExtractor(llm).extract(_chunk(text="   "))

    assert result == ChunkExtraction()
    assert llm.prompts == []


def test_model_failures_propagate():
    llm = FakeLLM(responses=[RuntimeError("groq down")])
    with pytest.raises(RuntimeError, match="groq down"):
        KnowledgeExtractor(llm).extract(_chunk())
```

Create `tests/knowledge/test_resolver.py`:

```python
from __future__ import annotations

from app.knowledge.llm import FakeLLM
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import (
    EntityMatch,
    EntityType,
    ExtractedEntity,
    ResolvedEntity,
    entity_key,
)


def _known(*entities: ResolvedEntity) -> dict[str, ResolvedEntity]:
    return {entity.key: entity for entity in entities}


PAYMENT = ResolvedEntity(
    key=entity_key("Payment API", EntityType.SERVICE),
    name="Payment API",
    type=EntityType.SERVICE,
)


def test_an_exact_key_match_reuses_the_existing_entity_without_the_model():
    llm = FakeLLM(responses=[])
    resolver = EntityResolver(llm)

    resolved = resolver.resolve(
        ExtractedEntity(name="payment   API", type=EntityType.SERVICE), _known(PAYMENT)
    )

    assert resolved.key == PAYMENT.key
    assert resolved.name == "Payment API"  # the canonical name wins
    assert llm.prompts == []


def test_punctuation_and_case_differences_resolve_without_the_model():
    llm = FakeLLM(responses=[])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment-API!", type=EntityType.SERVICE), _known(PAYMENT)
    )
    assert resolved.key == PAYMENT.key
    assert llm.prompts == []


def test_an_unrelated_entity_is_created_new_without_the_model():
    llm = FakeLLM(responses=[])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="PostgreSQL", type=EntityType.DATABASE), _known(PAYMENT)
    )

    assert resolved.key == entity_key("PostgreSQL", EntityType.DATABASE)
    assert llm.prompts == []


def test_the_model_adjudicates_only_a_fuzzy_candidate():
    llm = FakeLLM(responses=[EntityMatch(match=True, matched_key=PAYMENT.key)])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API service", type=EntityType.SERVICE),
        _known(PAYMENT),
    )

    assert resolved.key == PAYMENT.key
    assert len(llm.prompts) == 1


def test_a_rejected_fuzzy_match_creates_a_new_entity():
    llm = FakeLLM(responses=[EntityMatch(match=False, matched_key=None)])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API v2", type=EntityType.SERVICE), _known(PAYMENT)
    )

    assert resolved.key == entity_key("Payment API v2", EntityType.SERVICE)


def test_a_match_naming_an_unknown_key_is_ignored():
    llm = FakeLLM(responses=[EntityMatch(match=True, matched_key="service|invented")])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API gateway", type=EntityType.SERVICE),
        _known(PAYMENT),
    )
    assert resolved.key == entity_key("Payment API gateway", EntityType.SERVICE)


def test_candidates_of_a_different_type_are_not_offered_to_the_model():
    llm = FakeLLM(responses=[])
    other = ResolvedEntity(
        key=entity_key("Payment API", EntityType.DOCUMENT),
        name="Payment API",
        type=EntityType.DOCUMENT,
    )
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API docs", type=EntityType.SERVICE), _known(other)
    )

    assert llm.prompts == []
    assert resolved.type is EntityType.SERVICE


def test_an_alias_match_resolves_without_the_model():
    llm = FakeLLM(responses=[])
    aliased = PAYMENT.with_aliases(["payments service"])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payments Service", type=EntityType.SERVICE),
        _known(aliased),
    )

    assert resolved.key == PAYMENT.key
    assert llm.prompts == []


def test_resolution_works_with_no_model_configured():
    resolver = EntityResolver(None)
    resolved = resolver.resolve(
        ExtractedEntity(name="Payment API gateway", type=EntityType.SERVICE),
        _known(PAYMENT),
    )
    assert resolved.key == entity_key("Payment API gateway", EntityType.SERVICE)


def test_aliases_from_the_extraction_are_carried_onto_the_resolved_entity():
    resolved = EntityResolver(None).resolve(
        ExtractedEntity(
            name="PostgreSQL", type=EntityType.DATABASE, aliases=["Postgres"]
        ),
        {},
    )
    assert "postgres" in resolved.aliases
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/knowledge/test_extractor.py tests/knowledge/test_resolver.py -v`
Expected: FAIL — `ModuleNotFoundError` for both modules.

- [ ] **Step 3: Implement `app/knowledge/extractor.py`**

```python
from __future__ import annotations

from app.knowledge.llm import StructuredLLM
from app.knowledge.schema import Chunk, ChunkExtraction, EntityType, RelationType

__all__ = ["KnowledgeExtractor", "EXTRACTION_PROMPT"]

EXTRACTION_PROMPT = """\
Extract a knowledge graph from one section of an enterprise document.

Entity types (use exactly one of these):
{entity_types}

Relation types (use exactly one of these):
{relation_types}

Rules:
1. Extract only what the text states. Do not infer, complete or guess.
2. For every fact, quote the evidence verbatim from the text below. Copy the
   words exactly; do not paraphrase, summarise or re-punctuate.
3. Every fact's source and target must also appear in the entities list.
4. Prefer specific entity types over Other.
5. If the text states no relationships, return empty lists.

Section heading: {heading}

Text:
{text}
"""


class KnowledgeExtractor:
    """One structured model call per chunk."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def extract(self, chunk: Chunk) -> ChunkExtraction:
        if not chunk.text.strip():
            return ChunkExtraction()

        prompt = EXTRACTION_PROMPT.format(
            entity_types="\n".join(f"- {t.value}" for t in EntityType),
            relation_types="\n".join(f"- {r.value}" for r in RelationType),
            heading=chunk.heading or "(none)",
            text=chunk.text,
        )
        return self.llm.structured(prompt, ChunkExtraction)
```

- [ ] **Step 4: Implement `app/knowledge/resolver.py`**

```python
from __future__ import annotations

from app.knowledge.llm import StructuredLLM
from app.knowledge.schema import (
    EntityMatch,
    ExtractedEntity,
    ResolvedEntity,
    entity_key,
    normalize_name,
)

__all__ = ["EntityResolver", "RESOLUTION_PROMPT"]

RESOLUTION_PROMPT = """\
Decide whether a newly extracted entity refers to one that already exists.

New entity:
  name: {name}
  type: {type}

Existing candidates:
{candidates}

Rules:
1. Return match=true only if the new entity clearly denotes the same thing.
2. When matching, matched_key must be copied exactly from a candidate's key.
3. Similar names are not enough. "Payment API" and "Payment API v2" are
   different things; "Payment API" and "payments api" are the same thing.
4. Never invent a key.
"""


class EntityResolver:
    """Deterministic first, model second.

    Case, punctuation and whitespace variants are settled by the normalization
    key at no cost. The model is consulted only where a genuine judgement is
    needed, which keeps resolution mostly deterministic and cheap.
    """

    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self.llm = llm

    def resolve(
        self, candidate: ExtractedEntity, known: dict[str, ResolvedEntity]
    ) -> ResolvedEntity:
        key = entity_key(candidate.name, candidate.type)

        existing = known.get(key)
        if existing is not None:
            return existing.with_aliases([*candidate.aliases, candidate.name])

        normalized = normalize_name(candidate.name)
        for entity in known.values():
            if entity.type is not candidate.type:
                continue
            if normalized in entity.aliases:
                return entity.with_aliases([*candidate.aliases, candidate.name])

        fuzzy = self._fuzzy_candidates(normalized, candidate, known)
        if fuzzy and self.llm is not None:
            matched = self._adjudicate(candidate, fuzzy)
            if matched is not None:
                return matched.with_aliases([*candidate.aliases, candidate.name])

        fresh = ResolvedEntity(key=key, name=candidate.name.strip(), type=candidate.type)
        return fresh.with_aliases(candidate.aliases)

    def _fuzzy_candidates(
        self,
        normalized: str,
        candidate: ExtractedEntity,
        known: dict[str, ResolvedEntity],
    ) -> list[ResolvedEntity]:
        found: list[ResolvedEntity] = []
        for entity in known.values():
            if entity.type is not candidate.type:
                continue
            other = normalize_name(entity.name)
            if normalized in other or other in normalized:
                found.append(entity)
        return found

    def _adjudicate(
        self, candidate: ExtractedEntity, candidates: list[ResolvedEntity]
    ) -> ResolvedEntity | None:
        listing = "\n".join(
            f"  - key: {entity.key}\n    name: {entity.name}" for entity in candidates
        )
        prompt = RESOLUTION_PROMPT.format(
            name=candidate.name, type=candidate.type.value, candidates=listing
        )
        verdict = self.llm.structured(prompt, EntityMatch)
        if not verdict.match or not verdict.matched_key:
            return None
        # A key the model invented resolves to nothing; treat it as no match.
        return next((e for e in candidates if e.key == verdict.matched_key), None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/knowledge -v`
Expected: PASS, 30 tests.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add app/knowledge/extractor.py app/knowledge/resolver.py tests/knowledge
git status --short
```

---

## Task 4: GraphStore protocol and in-memory implementation

**Files:**
- Create: `app/graph/store.py`, `app/graph/memory.py`
- Create: `tests/graph/test_memory.py`

**Interfaces:**
- Consumes: `GraphUpsert`, `ResolvedEntity`, `GraphFact`, `ElementRef`, `DocumentRef`, `EntityType`, `RelationType`, `normalize_name` (Task 1).
- Produces: `GraphStats(documents, entities, facts, elements)` dataclass with `__add__`; `FactRecord(source_key, source_name, source_type, relation, target_key, target_name, target_type, doc_id, element_ids, evidence, hop)`; `GraphStore` protocol with `ensure_schema()`, `apply(upsert) -> GraphStats`, `known_entities() -> list[ResolvedEntity]`, `find_entities(names, limit=10) -> list[ResolvedEntity]`, `traverse(keys, hops=2, limit=200) -> list[FactRecord]`, `elements(global_ids) -> list[ElementRef]`, `stats() -> GraphStats`, `reset()`, `close()`; `InMemoryGraphStore` implementing all of it.

The element node's global identity is `gid = f"{doc_id}:{element_id}"`, because
IR element ids are only unique within their document.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_memory.py`:

```python
from __future__ import annotations

from app.graph.memory import InMemoryGraphStore
from app.graph.store import GraphStore
from app.knowledge.schema import (
    DocumentRef,
    ElementRef,
    EntityType,
    GraphFact,
    GraphUpsert,
    RelationType,
    ResolvedEntity,
    entity_key,
)

PAYMENT = entity_key("Payment API", EntityType.SERVICE)
REDIS = entity_key("Redis", EntityType.DATABASE)
TEAM = entity_key("Payments Team", EntityType.TEAM)
INCIDENT = entity_key("INC-2391", EntityType.INCIDENT)


def _element(eid: str, text: str, doc_id: str = "d") -> ElementRef:
    return ElementRef(
        id=eid,
        doc_id=doc_id,
        type="paragraph",
        page_number=1,
        x0=0, y0=0, x1=10, y1=10,
        order=0,
        text=text,
    )


def _upsert(doc_id: str = "d") -> GraphUpsert:
    return GraphUpsert(
        document=DocumentRef(
            id=doc_id, source_name="s.md", mime="text/markdown", checksum="c"
        ),
        entities=[
            ResolvedEntity(key=PAYMENT, name="Payment API", type=EntityType.SERVICE),
            ResolvedEntity(key=REDIS, name="Redis", type=EntityType.DATABASE),
            ResolvedEntity(key=TEAM, name="Payments Team", type=EntityType.TEAM),
            ResolvedEntity(key=INCIDENT, name="INC-2391", type=EntityType.INCIDENT),
        ],
        facts=[
            GraphFact(
                source_key=PAYMENT, target_key=REDIS, relation=RelationType.DEPENDS_ON,
                doc_id=doc_id, element_ids=["p1e001"], evidence="depends on Redis",
            ),
            GraphFact(
                source_key=TEAM, target_key=PAYMENT, relation=RelationType.OWNS,
                doc_id=doc_id, element_ids=["p1e002"], evidence="owns the Payment API",
            ),
            GraphFact(
                source_key=INCIDENT, target_key=PAYMENT, relation=RelationType.AFFECTS,
                doc_id=doc_id, element_ids=["p1e003"], evidence="INC-2391 affected",
            ),
        ],
        elements=[
            _element("p1e001", "The Payment API depends on Redis.", doc_id),
            _element("p1e002", "The Payments Team owns the Payment API.", doc_id),
            _element("p1e003", "INC-2391 affected the Payment API.", doc_id),
        ],
    )


def test_in_memory_store_satisfies_the_protocol():
    assert isinstance(InMemoryGraphStore(), GraphStore)


def test_apply_reports_what_it_wrote():
    stats = InMemoryGraphStore().apply(_upsert())
    assert (stats.documents, stats.entities, stats.facts, stats.elements) == (1, 4, 3, 3)


def test_applying_the_same_upsert_twice_creates_no_duplicates():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    store.apply(_upsert())

    totals = store.stats()
    assert (totals.documents, totals.entities, totals.facts) == (1, 4, 3)


def test_re_applying_merges_element_ids_onto_the_existing_fact():
    store = InMemoryGraphStore()
    store.apply(_upsert())

    second = _upsert()
    second.facts = [
        GraphFact(
            source_key=PAYMENT, target_key=REDIS, relation=RelationType.DEPENDS_ON,
            doc_id="d", element_ids=["p2e009"], evidence="also on page 2",
        )
    ]
    store.apply(second)

    records = store.traverse([PAYMENT], hops=1)
    depends = next(r for r in records if r.relation is RelationType.DEPENDS_ON)
    assert sorted(depends.element_ids) == ["p1e001", "p2e009"]


def test_find_entities_matches_on_name_case_insensitively():
    store = InMemoryGraphStore()
    store.apply(_upsert())

    found = store.find_entities(["payment api"])
    assert [entity.key for entity in found] == [PAYMENT]


def test_find_entities_matches_an_alias():
    store = InMemoryGraphStore()
    upsert = _upsert()
    upsert.entities[0] = upsert.entities[0].with_aliases(["payments service"])
    store.apply(upsert)

    assert [e.key for e in store.find_entities(["Payments Service"])] == [PAYMENT]


def test_find_entities_ignores_names_that_match_nothing():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert store.find_entities(["Kubernetes"]) == []


def test_one_hop_traversal_returns_the_adjacent_facts():
    store = InMemoryGraphStore()
    store.apply(_upsert())

    records = store.traverse([INCIDENT], hops=1)
    assert {r.relation for r in records} == {RelationType.AFFECTS}
    assert records[0].hop == 1
    assert records[0].target_name == "Payment API"


def test_two_hop_traversal_reaches_the_owning_team():
    """The multi-hop question: which team owns the service this incident hit?"""
    store = InMemoryGraphStore()
    store.apply(_upsert())

    records = store.traverse([INCIDENT], hops=2)
    owners = [r for r in records if r.relation is RelationType.OWNS]

    assert owners, "second hop did not reach the OWNS edge"
    assert owners[0].source_name == "Payments Team"
    assert owners[0].hop == 2


def test_traversal_is_bounded_by_the_hop_count():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert all(record.hop <= 1 for record in store.traverse([INCIDENT], hops=1))


def test_traversal_respects_the_limit():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert len(store.traverse([PAYMENT], hops=2, limit=1)) == 1


def test_traversal_from_an_unknown_key_returns_nothing():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert store.traverse(["service|nonexistent"], hops=2) == []


def test_elements_are_fetched_by_global_id():
    store = InMemoryGraphStore()
    store.apply(_upsert())

    found = store.elements(["d:p1e001", "d:missing"])
    assert [element.id for element in found] == ["p1e001"]


def test_known_entities_returns_everything_for_the_resolver():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert len(store.known_entities()) == 4


def test_reset_empties_the_store():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    store.reset()

    totals = store.stats()
    assert (totals.documents, totals.entities, totals.facts, totals.elements) == (
        0, 0, 0, 0,
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/graph/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.graph.store'`

- [ ] **Step 3: Implement `app/graph/store.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from app.knowledge.schema import (
    ElementRef,
    EntityType,
    GraphUpsert,
    RelationType,
    ResolvedEntity,
)

__all__ = ["GraphStats", "FactRecord", "GraphStore"]


@dataclass(frozen=True)
class GraphStats:
    documents: int = 0
    entities: int = 0
    facts: int = 0
    elements: int = 0

    def __add__(self, other: GraphStats) -> GraphStats:
        return GraphStats(
            documents=self.documents + other.documents,
            entities=self.entities + other.entities,
            facts=self.facts + other.facts,
            elements=self.elements + other.elements,
        )


class FactRecord(BaseModel):
    """One traversed edge, carrying the provenance needed to cite it."""

    source_key: str
    source_name: str
    source_type: EntityType
    relation: RelationType
    target_key: str
    target_name: str
    target_type: EntityType
    doc_id: str
    element_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    hop: int = 1

    def global_element_ids(self) -> list[str]:
        return [f"{self.doc_id}:{element_id}" for element_id in self.element_ids]

    def sentence(self) -> str:
        return f"{self.source_name} {self.relation.value} {self.target_name}"


@runtime_checkable
class GraphStore(Protocol):
    def ensure_schema(self) -> None: ...
    def apply(self, upsert: GraphUpsert) -> GraphStats: ...
    def known_entities(self) -> list[ResolvedEntity]: ...
    def find_entities(
        self, names: Sequence[str], limit: int = 10
    ) -> list[ResolvedEntity]: ...
    def traverse(
        self, keys: Sequence[str], hops: int = 2, limit: int = 200
    ) -> list[FactRecord]: ...
    def elements(self, global_ids: Sequence[str]) -> list[ElementRef]: ...
    def stats(self) -> GraphStats: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

- [ ] **Step 4: Implement `app/graph/memory.py`**

```python
from __future__ import annotations

from typing import Sequence

from app.graph.store import FactRecord, GraphStats
from app.knowledge.schema import (
    DocumentRef,
    ElementRef,
    GraphFact,
    GraphUpsert,
    ResolvedEntity,
    normalize_name,
)

__all__ = ["InMemoryGraphStore"]


def _fact_key(fact: GraphFact) -> tuple[str, str, str, str]:
    return (fact.source_key, fact.relation.value, fact.target_key, fact.doc_id)


class InMemoryGraphStore:
    """A dict-backed GraphStore.

    Exists so the pipeline, retrieval and answering can be tested without a
    database, and so traversal logic is verified independently of Cypher.
    """

    def __init__(self) -> None:
        self.documents: dict[str, DocumentRef] = {}
        self.entities: dict[str, ResolvedEntity] = {}
        self.elements_by_gid: dict[str, ElementRef] = {}
        self.facts: dict[tuple[str, str, str, str], GraphFact] = {}

    def ensure_schema(self) -> None:
        return None

    def apply(self, upsert: GraphUpsert) -> GraphStats:
        self.documents[upsert.document.id] = upsert.document

        for entity in upsert.entities:
            existing = self.entities.get(entity.key)
            self.entities[entity.key] = (
                existing.with_aliases([*entity.aliases, entity.name])
                if existing
                else entity
            )

        for element in upsert.elements:
            self.elements_by_gid[element.global_id] = element

        for fact in upsert.facts:
            key = _fact_key(fact)
            existing = self.facts.get(key)
            if existing is None:
                self.facts[key] = fact.model_copy(deep=True)
                continue
            merged = list(dict.fromkeys([*existing.element_ids, *fact.element_ids]))
            self.facts[key] = existing.model_copy(update={"element_ids": merged})

        return GraphStats(
            documents=1,
            entities=len(upsert.entities),
            facts=len(upsert.facts),
            elements=len(upsert.elements),
        )

    def known_entities(self) -> list[ResolvedEntity]:
        return list(self.entities.values())

    def find_entities(
        self, names: Sequence[str], limit: int = 10
    ) -> list[ResolvedEntity]:
        found: dict[str, ResolvedEntity] = {}
        for raw in names:
            needle = normalize_name(raw)
            if not needle:
                continue
            for entity in self.entities.values():
                name = normalize_name(entity.name)
                if needle == name or needle in entity.aliases or needle in name:
                    found[entity.key] = entity
        return list(found.values())[:limit]

    def _record(self, fact: GraphFact, hop: int) -> FactRecord:
        source = self.entities[fact.source_key]
        target = self.entities[fact.target_key]
        return FactRecord(
            source_key=source.key,
            source_name=source.name,
            source_type=source.type,
            relation=fact.relation,
            target_key=target.key,
            target_name=target.name,
            target_type=target.type,
            doc_id=fact.doc_id,
            element_ids=list(fact.element_ids),
            evidence=fact.evidence,
            hop=hop,
        )

    def traverse(
        self, keys: Sequence[str], hops: int = 2, limit: int = 200
    ) -> list[FactRecord]:
        """Breadth-first expansion, edges undirected, each edge reported once."""
        frontier = {key for key in keys if key in self.entities}
        seen_edges: set[tuple[str, str, str, str]] = set()
        visited = set(frontier)
        records: list[FactRecord] = []

        for hop in range(1, max(0, hops) + 1):
            if not frontier or len(records) >= limit:
                break
            next_frontier: set[str] = set()

            for key, fact in self.facts.items():
                if key in seen_edges:
                    continue
                if fact.source_key in frontier:
                    other = fact.target_key
                elif fact.target_key in frontier:
                    other = fact.source_key
                else:
                    continue

                seen_edges.add(key)
                records.append(self._record(fact, hop))
                if other not in visited:
                    next_frontier.add(other)
                    visited.add(other)
                if len(records) >= limit:
                    break

            frontier = next_frontier

        return records[:limit]

    def elements(self, global_ids: Sequence[str]) -> list[ElementRef]:
        return [
            self.elements_by_gid[gid]
            for gid in global_ids
            if gid in self.elements_by_gid
        ]

    def stats(self) -> GraphStats:
        return GraphStats(
            documents=len(self.documents),
            entities=len(self.entities),
            facts=len(self.facts),
            elements=len(self.elements_by_gid),
        )

    def reset(self) -> None:
        self.documents.clear()
        self.entities.clear()
        self.elements_by_gid.clear()
        self.facts.clear()

    def close(self) -> None:
        return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/graph/test_memory.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add app/graph/store.py app/graph/memory.py tests/graph
git status --short
```

---

## Task 5: Cypher builders and the Neo4j store

**Files:**
- Create: `app/graph/cypher.py`
- Modify: `app/graph/neo4j.py` (full rewrite — `Neo4jClient` becomes `Neo4jGraphStore`)
- Create: `tests/graph/test_cypher.py`

**Interfaces:**
- Consumes: `RelationType` (Task 1); `GraphStore`, `GraphStats`, `FactRecord` (Task 4).
- Produces: `relation_type(value: str) -> str`; `SCHEMA_STATEMENTS: tuple[str, ...]`; `MERGE_DOCUMENT`, `MERGE_ENTITY`, `MERGE_ELEMENT`, `LINK_ELEMENT`, `MENTION`, `EXPAND`, `FIND_ENTITIES`, `KNOWN_ENTITIES`, `FETCH_ELEMENTS`, `COUNTS`, `RESET`; `merge_fact(relation: str) -> str`; `Neo4jGraphStore(uri, username, password)`.

Only `merge_fact` interpolates anything into Cypher, and only after
`relation_type` has checked it against the enum. Everything else is
parameterised.

- [ ] **Step 1: Write the failing test**

Create `tests/graph/test_cypher.py`:

```python
from __future__ import annotations

import pytest

from app.graph import cypher
from app.knowledge.schema import RelationType


def test_relation_type_accepts_every_enum_member():
    for relation in RelationType:
        assert cypher.relation_type(relation.value) == relation.value


def test_relation_type_normalizes_case_and_whitespace():
    assert cypher.relation_type("  depends_on ") == "DEPENDS_ON"


def test_relation_type_rejects_anything_outside_the_allowlist():
    with pytest.raises(ValueError, match="unsupported relation type"):
        cypher.relation_type("EATS")


@pytest.mark.parametrize(
    "attack",
    [
        "DEPENDS_ON]->(x) DETACH DELETE x //",
        "RELATED_TO` DELETE n `",
        "DROP",
        "",
        "REL ATED",
    ],
)
def test_relation_type_rejects_injection_attempts(attack):
    with pytest.raises(ValueError):
        cypher.relation_type(attack)


def test_merge_fact_embeds_a_real_relationship_type():
    statement = cypher.merge_fact("DEPENDS_ON")
    assert "-[r:DEPENDS_ON" in statement
    assert "$source_key" in statement
    assert "$target_key" in statement


def test_merge_fact_refuses_an_unknown_relation():
    with pytest.raises(ValueError):
        cypher.merge_fact("NOT_REAL")


def test_schema_statements_are_idempotent():
    for statement in cypher.SCHEMA_STATEMENTS:
        assert "IF NOT EXISTS" in statement


def test_schema_constrains_the_three_identities():
    joined = " ".join(cypher.SCHEMA_STATEMENTS)
    assert "e.key IS UNIQUE" in joined
    assert "e.gid IS UNIQUE" in joined
    assert "d.id IS UNIQUE" in joined


def test_every_statement_uses_parameters_not_formatting():
    statements = [
        cypher.MERGE_DOCUMENT,
        cypher.MERGE_ENTITY,
        cypher.MERGE_ELEMENT,
        cypher.EXPAND,
        cypher.FIND_ENTITIES,
        cypher.FETCH_ELEMENTS,
    ]
    for statement in statements:
        assert "$" in statement
        assert "{}" not in statement
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/graph/test_cypher.py -v`
Expected: FAIL — `ImportError: cannot import name 'cypher'`

- [ ] **Step 3: Implement `app/graph/cypher.py`**

```python
from __future__ import annotations

from app.knowledge.schema import RelationType

__all__ = [
    "relation_type",
    "merge_fact",
    "SCHEMA_STATEMENTS",
    "MERGE_DOCUMENT",
    "MERGE_ENTITY",
    "MERGE_ELEMENT",
    "LINK_ELEMENT",
    "MENTION",
    "EXPAND",
    "FIND_ENTITIES",
    "KNOWN_ENTITIES",
    "FETCH_ELEMENTS",
    "COUNTS",
    "RESET",
]

_ALLOWED = frozenset(relation.value for relation in RelationType)


def relation_type(value: str) -> str:
    """Validate a relationship type against the closed vocabulary.

    Cypher cannot parameterise a relationship type, so this is the only value
    ever interpolated into a statement. Everything the model produces passes
    through here first.
    """
    candidate = value.strip().upper()
    if candidate not in _ALLOWED:
        raise ValueError(f"unsupported relation type: {value!r}")
    return candidate


SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE CONSTRAINT element_gid IF NOT EXISTS "
    "FOR (e:Element) REQUIRE e.gid IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
)

MERGE_DOCUMENT = """
MERGE (d:Document {id: $id})
SET d.source_name = $source_name,
    d.mime = $mime,
    d.title = $title,
    d.checksum = $checksum
"""

MERGE_ENTITY = """
MERGE (e:Entity {key: $key})
SET e.name = $name,
    e.type = $type,
    e.aliases = [alias IN coalesce(e.aliases, []) + $aliases
                 | alias][0..50]
"""

MERGE_ELEMENT = """
MERGE (e:Element {gid: $gid})
SET e.id = $id,
    e.doc_id = $doc_id,
    e.type = $type,
    e.page = $page,
    e.x0 = $x0, e.y0 = $y0, e.x1 = $x1, e.y1 = $y1,
    e.order = $order,
    e.text = $text
"""

LINK_ELEMENT = """
MATCH (d:Document {id: $doc_id})
MATCH (e:Element {gid: $gid})
MERGE (d)-[:HAS_ELEMENT]->(e)
"""

MENTION = """
MATCH (n:Entity {key: $key})
MATCH (e:Element {gid: $gid})
MERGE (n)-[:MENTIONED_IN]->(e)
"""


def merge_fact(relation: str) -> str:
    """Upsert one relationship, merging element ids onto an existing edge.

    The dedupe is written in plain Cypher rather than via APOC so the store
    works on any Neo4j instance, Aura Free included.
    """
    validated = relation_type(relation)
    return f"""
    MATCH (s:Entity {{key: $source_key}})
    MATCH (t:Entity {{key: $target_key}})
    MERGE (s)-[r:{validated} {{doc_id: $doc_id}}]->(t)
    WITH r, coalesce(r.element_ids, []) + $element_ids AS merged
    SET r.element_ids = [i IN range(0, size(merged) - 1)
                         WHERE NOT merged[i] IN merged[0..i] | merged[i]],
        r.evidence = $evidence,
        r.confidence = $confidence
    """


EXPAND = """
MATCH (s:Entity)-[r]-(:Entity)
WHERE s.key IN $keys AND type(r) <> 'MENTIONED_IN'
RETURN startNode(r).key   AS source_key,
       startNode(r).name  AS source_name,
       startNode(r).type  AS source_type,
       type(r)            AS relation,
       endNode(r).key     AS target_key,
       endNode(r).name    AS target_name,
       endNode(r).type    AS target_type,
       r.doc_id           AS doc_id,
       r.element_ids      AS element_ids,
       r.evidence         AS evidence
LIMIT $limit
"""

FIND_ENTITIES = """
UNWIND $names AS name
MATCH (e:Entity)
WHERE toLower(e.name) = toLower(name)
   OR toLower(e.name) CONTAINS toLower(name)
   OR any(alias IN coalesce(e.aliases, []) WHERE alias = toLower(name))
RETURN DISTINCT e.key AS key, e.name AS name, e.type AS type,
       coalesce(e.aliases, []) AS aliases
LIMIT $limit
"""

KNOWN_ENTITIES = """
MATCH (e:Entity)
RETURN e.key AS key, e.name AS name, e.type AS type,
       coalesce(e.aliases, []) AS aliases
LIMIT $limit
"""

FETCH_ELEMENTS = """
MATCH (e:Element)
WHERE e.gid IN $gids
RETURN e.id AS id, e.doc_id AS doc_id, e.type AS type, e.page AS page,
       e.x0 AS x0, e.y0 AS y0, e.x1 AS x1, e.y1 AS y1,
       e.order AS order, e.text AS text
"""

COUNTS = """
MATCH (d:Document) WITH count(d) AS documents
MATCH (e:Entity)   WITH documents, count(e) AS entities
MATCH (el:Element) WITH documents, entities, count(el) AS elements
OPTIONAL MATCH ()-[r]->() WHERE type(r) <> 'MENTIONED_IN' AND type(r) <> 'HAS_ELEMENT'
RETURN documents, entities, elements, count(r) AS facts
"""

RESET = "MATCH (n) WHERE n:Document OR n:Entity OR n:Element DETACH DELETE n"
```

- [ ] **Step 4: Rewrite `app/graph/neo4j.py`**

```python
from __future__ import annotations

from typing import Any, Sequence

from neo4j import GraphDatabase

from app.graph import cypher
from app.graph.store import FactRecord, GraphStats
from app.knowledge.schema import ElementRef, GraphUpsert, ResolvedEntity

__all__ = ["Neo4jGraphStore"]


class Neo4jGraphStore:
    """Neo4j-backed GraphStore.

    Traversal is iterative rather than a variable-length pattern: Cypher cannot
    parameterise a path length, and expanding one hop at a time yields the hop
    number for free, which the UI shows in its retrieval trace.
    """

    def __init__(self, uri: str, username: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def verify_connectivity(self) -> None:
        self.driver.verify_connectivity()

    def ensure_schema(self) -> None:
        with self.driver.session() as session:
            for statement in cypher.SCHEMA_STATEMENTS:
                session.run(statement)

    def apply(self, upsert: GraphUpsert) -> GraphStats:
        with self.driver.session() as session:
            return session.execute_write(self._apply, upsert)

    @staticmethod
    def _apply(tx: Any, upsert: GraphUpsert) -> GraphStats:
        document = upsert.document
        tx.run(
            cypher.MERGE_DOCUMENT,
            id=document.id,
            source_name=document.source_name,
            mime=document.mime,
            title=document.title,
            checksum=document.checksum,
        )

        for entity in upsert.entities:
            tx.run(
                cypher.MERGE_ENTITY,
                key=entity.key,
                name=entity.name,
                type=entity.type.value,
                aliases=entity.aliases,
            )

        for element in upsert.elements:
            tx.run(
                cypher.MERGE_ELEMENT,
                gid=element.global_id,
                id=element.id,
                doc_id=element.doc_id,
                type=element.type,
                page=element.page_number,
                x0=element.x0, y0=element.y0, x1=element.x1, y1=element.y1,
                order=element.order,
                text=element.text,
            )
            tx.run(cypher.LINK_ELEMENT, doc_id=element.doc_id, gid=element.global_id)

        for fact in upsert.facts:
            tx.run(
                cypher.merge_fact(fact.relation.value),
                source_key=fact.source_key,
                target_key=fact.target_key,
                doc_id=fact.doc_id,
                element_ids=fact.element_ids,
                evidence=fact.evidence,
                confidence=fact.confidence,
            )
            for element_id in fact.element_ids:
                gid = f"{fact.doc_id}:{element_id}"
                tx.run(cypher.MENTION, key=fact.source_key, gid=gid)
                tx.run(cypher.MENTION, key=fact.target_key, gid=gid)

        return GraphStats(
            documents=1,
            entities=len(upsert.entities),
            facts=len(upsert.facts),
            elements=len(upsert.elements),
        )

    @staticmethod
    def _entity(record: Any) -> ResolvedEntity:
        return ResolvedEntity(
            key=record["key"],
            name=record["name"],
            type=record["type"],
            aliases=list(record["aliases"] or []),
        )

    def known_entities(self, limit: int = 2000) -> list[ResolvedEntity]:
        with self.driver.session() as session:
            result = session.run(cypher.KNOWN_ENTITIES, limit=limit)
            return [self._entity(record) for record in result]

    def find_entities(
        self, names: Sequence[str], limit: int = 10
    ) -> list[ResolvedEntity]:
        cleaned = [name for name in names if name and name.strip()]
        if not cleaned:
            return []
        with self.driver.session() as session:
            result = session.run(cypher.FIND_ENTITIES, names=cleaned, limit=limit)
            return [self._entity(record) for record in result]

    def traverse(
        self, keys: Sequence[str], hops: int = 2, limit: int = 200
    ) -> list[FactRecord]:
        frontier = [key for key in keys if key]
        visited = set(frontier)
        seen: set[tuple[str, str, str, str]] = set()
        records: list[FactRecord] = []

        with self.driver.session() as session:
            for hop in range(1, max(0, hops) + 1):
                if not frontier or len(records) >= limit:
                    break
                result = session.run(
                    cypher.EXPAND, keys=frontier, limit=limit - len(records)
                )
                next_frontier: list[str] = []

                for row in result:
                    identity = (
                        row["source_key"],
                        row["relation"],
                        row["target_key"],
                        row["doc_id"] or "",
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    records.append(
                        FactRecord(
                            source_key=row["source_key"],
                            source_name=row["source_name"],
                            source_type=row["source_type"],
                            relation=row["relation"],
                            target_key=row["target_key"],
                            target_name=row["target_name"],
                            target_type=row["target_type"],
                            doc_id=row["doc_id"] or "",
                            element_ids=list(row["element_ids"] or []),
                            evidence=row["evidence"] or "",
                            hop=hop,
                        )
                    )
                    for key in (row["source_key"], row["target_key"]):
                        if key not in visited:
                            visited.add(key)
                            next_frontier.append(key)

                frontier = next_frontier

        return records[:limit]

    def elements(self, global_ids: Sequence[str]) -> list[ElementRef]:
        wanted = [gid for gid in global_ids if gid]
        if not wanted:
            return []
        with self.driver.session() as session:
            result = session.run(cypher.FETCH_ELEMENTS, gids=wanted)
            return [
                ElementRef(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    type=row["type"],
                    page_number=row["page"],
                    x0=row["x0"], y0=row["y0"], x1=row["x1"], y1=row["y1"],
                    order=row["order"],
                    text=row["text"] or "",
                )
                for row in result
            ]

    def stats(self) -> GraphStats:
        with self.driver.session() as session:
            row = session.run(cypher.COUNTS).single()
            if row is None:
                return GraphStats()
            return GraphStats(
                documents=row["documents"],
                entities=row["entities"],
                facts=row["facts"],
                elements=row["elements"],
            )

    def reset(self) -> None:
        """Delete every Document, Entity and Element. Only ever called from the UI."""
        with self.driver.session() as session:
            session.run(cypher.RESET)

    def close(self) -> None:
        self.driver.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/graph -v`
Expected: PASS, 25 tests.

- [ ] **Step 6: Confirm the store satisfies the protocol**

Run:

```bash
uv run python -c "
from app.graph.store import GraphStore
from app.graph.neo4j import Neo4jGraphStore
print('protocol satisfied:', isinstance(Neo4jGraphStore.__new__(Neo4jGraphStore), GraphStore))
"
```

Expected: `protocol satisfied: True`

- [ ] **Step 7: Stage the changes (do not commit)**

```bash
git add app/graph tests/graph
git status --short
```

---

## Task 6: Pipeline — evidence binding and orchestration

**Files:**
- Create: `app/knowledge/pipeline.py`
- Create: `tests/knowledge/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces: `bind_evidence(quote, chunk, elements) -> tuple[list[str], str]`; `element_ref(doc, element) -> ElementRef`; `KnowledgePipeline(extractor, resolver)` with `.run(doc: Document, known: Sequence[ResolvedEntity] = (), *, on_progress: Callable[[int, int], None] | None = None) -> GraphUpsert`.

- [ ] **Step 1: Write the failing test**

Create `tests/knowledge/test_pipeline.py`:

```python
from __future__ import annotations

from app.ir.model import BBox, Document, Element, ElementType, Page
from app.ir.tree import assign_parents
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.llm import FakeLLM
from app.knowledge.pipeline import KnowledgePipeline, bind_evidence
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import (
    Chunk,
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    RelationType,
    entity_key,
)


def _element(eid, etype, order, text, *, level=None):
    return Element(
        id=eid,
        type=etype,
        page_number=1,
        bbox=BBox(x0=0, y0=order * 10, x1=100, y1=order * 10 + 8),
        order=order,
        level=level,
        text=text,
    )


def _doc() -> Document:
    elements = [
        _element("p1e000", ElementType.HEADING, 0, "Payment System", level=1),
        _element("p1e001", ElementType.PARAGRAPH, 1, "The Payment API depends on Redis."),
        _element("p1e002", ElementType.PARAGRAPH, 2, "The Payments Team owns the Payment API."),
    ]
    assign_parents(elements)
    return Document(
        id="doc1",
        source_name="payments.md",
        mime="text/markdown",
        checksum="c" * 64,
        title="Payment System",
        pages=[Page(number=1, width=612, height=792, elements=elements)],
    )


def _extraction() -> ChunkExtraction:
    return ChunkExtraction(
        entities=[
            ExtractedEntity(name="Payment API", type=EntityType.SERVICE),
            ExtractedEntity(name="Redis", type=EntityType.DATABASE),
            ExtractedEntity(name="Payments Team", type=EntityType.TEAM),
        ],
        facts=[
            ExtractedFact(
                source="Payment API",
                relation=RelationType.DEPENDS_ON,
                target="Redis",
                evidence="The Payment API depends on Redis.",
            ),
            ExtractedFact(
                source="Payments Team",
                relation=RelationType.OWNS,
                target="Payment API",
                evidence="The Payments Team owns the Payment API.",
            ),
        ],
    )


def _pipeline(llm: FakeLLM) -> KnowledgePipeline:
    return KnowledgePipeline(KnowledgeExtractor(llm), EntityResolver(llm))


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="doc1:c000",
        doc_id="doc1",
        page_number=1,
        heading="Payment System",
        text=text,
        element_ids=("p1e001", "p1e002"),
    )


def _elements() -> dict[str, Element]:
    return {element.id: element for element in _doc().iter_elements()}


def test_a_quote_binds_to_the_element_it_came_from():
    ids, confidence = bind_evidence(
        "The Payment API depends on Redis.", _chunk("…"), _elements()
    )
    assert ids == ["p1e001"]
    assert confidence == "high"


def test_binding_tolerates_whitespace_and_case_differences():
    ids, confidence = bind_evidence(
        "the payment   api DEPENDS on redis", _chunk("…"), _elements()
    )
    assert ids == ["p1e001"]
    assert confidence == "high"


def test_an_unmatched_quote_falls_back_to_the_whole_chunk_at_low_confidence():
    ids, confidence = bind_evidence(
        "a sentence that appears nowhere", _chunk("…"), _elements()
    )
    assert ids == ["p1e001", "p1e002"]
    assert confidence == "low"


def test_an_empty_quote_falls_back_to_the_whole_chunk():
    ids, confidence = bind_evidence("   ", _chunk("…"), _elements())
    assert ids == ["p1e001", "p1e002"]
    assert confidence == "low"


def test_the_pipeline_produces_facts_with_element_level_provenance():
    llm = FakeLLM(responses=[_extraction()])
    upsert = _pipeline(llm).run(_doc())

    depends = next(f for f in upsert.facts if f.relation is RelationType.DEPENDS_ON)
    assert depends.element_ids == ["p1e001"]
    assert depends.doc_id == "doc1"
    assert depends.confidence == "high"


def test_every_fact_resolves_to_a_known_entity_key():
    llm = FakeLLM(responses=[_extraction()])
    upsert = _pipeline(llm).run(_doc())

    keys = {entity.key for entity in upsert.entities}
    for fact in upsert.facts:
        assert fact.source_key in keys
        assert fact.target_key in keys


def test_only_evidence_elements_are_written():
    llm = FakeLLM(responses=[_extraction()])
    upsert = _pipeline(llm).run(_doc())

    assert {element.id for element in upsert.elements} == {"p1e001", "p1e002"}


def test_the_document_reference_carries_identity():
    llm = FakeLLM(responses=[_extraction()])
    upsert = _pipeline(llm).run(_doc())

    assert upsert.document.id == "doc1"
    assert upsert.document.title == "Payment System"
    assert upsert.document.mime == "text/markdown"


def test_a_fact_naming_an_unextracted_entity_is_dropped_with_a_warning():
    extraction = ChunkExtraction(
        entities=[ExtractedEntity(name="Payment API", type=EntityType.SERVICE)],
        facts=[
            ExtractedFact(
                source="Payment API",
                relation=RelationType.USES,
                target="Kafka",
                evidence="The Payment API depends on Redis.",
            )
        ],
    )
    upsert = _pipeline(FakeLLM(responses=[extraction])).run(_doc())

    assert upsert.facts == []
    assert upsert.warnings[0]["code"] == "unknown_entity"
    assert "Kafka" in upsert.warnings[0]["detail"]


def test_a_failing_chunk_is_recorded_and_the_rest_still_commits():
    long_text = "x" * 1600
    elements = [
        _element("p1e000", ElementType.HEADING, 0, "Payment System", level=1),
        _element("p1e001", ElementType.PARAGRAPH, 1, long_text),
        _element("p1e002", ElementType.PARAGRAPH, 2, "The Payment API depends on Redis."),
    ]
    assign_parents(elements)
    doc = Document(
        id="doc2",
        source_name="p.md",
        mime="text/markdown",
        checksum="c",
        pages=[Page(number=1, width=612, height=792, elements=elements)],
    )

    llm = FakeLLM(
        responses=[
            RuntimeError("groq timeout"),
            ChunkExtraction(
                entities=[
                    ExtractedEntity(name="Payment API", type=EntityType.SERVICE),
                    ExtractedEntity(name="Redis", type=EntityType.DATABASE),
                ],
                facts=[
                    ExtractedFact(
                        source="Payment API",
                        relation=RelationType.DEPENDS_ON,
                        target="Redis",
                        evidence="The Payment API depends on Redis.",
                    )
                ],
            ),
        ]
    )
    upsert = KnowledgePipeline(KnowledgeExtractor(llm), EntityResolver(None)).run(doc)

    assert any(w["code"] == "chunk_failed" for w in upsert.warnings)
    assert len(upsert.facts) == 1


def test_known_entities_are_reused_rather_than_duplicated():
    from app.knowledge.schema import ResolvedEntity

    known = [
        ResolvedEntity(
            key=entity_key("Payment API", EntityType.SERVICE),
            name="Payment API",
            type=EntityType.SERVICE,
        )
    ]
    llm = FakeLLM(responses=[_extraction()])
    upsert = _pipeline(llm).run(_doc(), known)

    matching = [e for e in upsert.entities if e.name == "Payment API"]
    assert len(matching) == 1


def test_progress_is_reported_per_chunk():
    llm = FakeLLM(responses=[_extraction()])
    seen: list[tuple[int, int]] = []
    _pipeline(llm).run(_doc(), on_progress=lambda done, total: seen.append((done, total)))

    assert seen == [(1, 1)]


def test_a_document_with_no_content_produces_an_empty_upsert():
    doc = Document(id="empty", source_name="e.md", mime="text/markdown", checksum="c")
    upsert = _pipeline(FakeLLM(responses=[])).run(doc)

    assert upsert.entities == []
    assert upsert.facts == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/knowledge/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.knowledge.pipeline'`

- [ ] **Step 3: Implement `app/knowledge/pipeline.py`**

```python
from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from app.ir.model import Document, Element
from app.knowledge.chunker import chunk_document
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import (
    Chunk,
    DocumentRef,
    ElementRef,
    GraphFact,
    GraphUpsert,
    ResolvedEntity,
    normalize_name,
)

__all__ = ["KnowledgePipeline", "bind_evidence", "element_ref"]

_WHITESPACE = re.compile(r"\s+")


def _flatten(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


def bind_evidence(
    quote: str, chunk: Chunk, elements: Mapping[str, Element]
) -> tuple[list[str], str]:
    """Locate the element a fact's evidence quote came from.

    Chunk-level provenance would only narrow a fact to a few paragraphs. Matching
    the quote pins it to one element, which is what makes a citation a rectangle
    on a page rather than a region of a document.
    """
    needle = _flatten(quote)
    if needle:
        for element_id in chunk.element_ids:
            element = elements.get(element_id)
            if element is None:
                continue
            if needle in _flatten(element.text_content()):
                return [element_id], "high"

    # No match: keep the fact, but say honestly that it is only chunk-level.
    return list(chunk.element_ids), "low"


def element_ref(doc: Document, element: Element) -> ElementRef:
    return ElementRef(
        id=element.id,
        doc_id=doc.id,
        type=element.type.value,
        page_number=element.page_number,
        x0=element.bbox.x0,
        y0=element.bbox.y0,
        x1=element.bbox.x1,
        y1=element.bbox.y1,
        order=element.order,
        text=element.text_content(),
    )


class KnowledgePipeline:
    """Document in, GraphUpsert out.

    Extraction is isolated per chunk: a model failure on one chunk is recorded
    and the remaining chunks still contribute, so a single bad call does not
    cost the whole document.
    """

    def __init__(self, extractor: KnowledgeExtractor, resolver: EntityResolver) -> None:
        self.extractor = extractor
        self.resolver = resolver

    def run(
        self,
        doc: Document,
        known: Sequence[ResolvedEntity] = (),
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> GraphUpsert:
        upsert = GraphUpsert(
            document=DocumentRef(
                id=doc.id,
                source_name=doc.source_name,
                mime=doc.mime,
                title=doc.title,
                checksum=doc.checksum,
            )
        )

        elements = {element.id: element for element in doc.iter_elements()}
        registry: dict[str, ResolvedEntity] = {entity.key: entity for entity in known}
        touched_entities: dict[str, ResolvedEntity] = {}
        evidence_elements: dict[str, ElementRef] = {}

        chunks = chunk_document(doc)
        for index, chunk in enumerate(chunks, start=1):
            try:
                extraction = self.extractor.extract(chunk)
            except Exception as error:  # one bad chunk must not cost the document
                upsert.add_warning("chunk_failed", f"{chunk.id}: {error}")
                if on_progress:
                    on_progress(index, len(chunks))
                continue

            by_name: dict[str, ResolvedEntity] = {}
            for candidate in extraction.entities:
                if not candidate.name.strip():
                    continue
                resolved = self.resolver.resolve(candidate, registry)
                registry[resolved.key] = resolved
                touched_entities[resolved.key] = resolved
                by_name[normalize_name(candidate.name)] = resolved
                for alias in candidate.aliases:
                    by_name.setdefault(normalize_name(alias), resolved)

            for fact in extraction.facts:
                source = by_name.get(normalize_name(fact.source))
                target = by_name.get(normalize_name(fact.target))
                if source is None or target is None:
                    missing = fact.source if source is None else fact.target
                    upsert.add_warning(
                        "unknown_entity",
                        f"{chunk.id}: fact names {missing!r}, which was not extracted",
                    )
                    continue
                if source.key == target.key:
                    upsert.add_warning(
                        "self_reference", f"{chunk.id}: {source.name} relates to itself"
                    )
                    continue

                element_ids, confidence = bind_evidence(fact.evidence, chunk, elements)
                for element_id in element_ids:
                    element = elements.get(element_id)
                    if element is not None:
                        evidence_elements[element_id] = element_ref(doc, element)

                upsert.facts.append(
                    GraphFact(
                        source_key=source.key,
                        target_key=target.key,
                        relation=fact.relation,
                        doc_id=doc.id,
                        element_ids=element_ids,
                        evidence=fact.evidence.strip(),
                        confidence=confidence,
                    )
                )

            if on_progress:
                on_progress(index, len(chunks))

        upsert.entities = list(touched_entities.values())
        upsert.elements = list(evidence_elements.values())
        return upsert
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/knowledge -v`
Expected: PASS, 43 tests.

- [ ] **Step 5: Verify a real document end to end against the in-memory store**

Run:

```bash
uv run python -c "
from app.graph.memory import InMemoryGraphStore
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.llm import GroqLLM
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.resolver import EntityResolver
from app.parsing.base import InMemoryBlobSink, parse_document

doc = parse_document('app/assets/payment_system.md', blobs=InMemoryBlobSink())
llm = GroqLLM()
upsert = KnowledgePipeline(KnowledgeExtractor(llm), EntityResolver(llm)).run(doc)
for fact in upsert.facts:
    print(f'{fact.source_key} -{fact.relation}-> {fact.target_key}  {fact.element_ids} {fact.confidence}')
store = InMemoryGraphStore(); store.apply(upsert)
print('stats:', store.stats())
"
```

Expected: facts such as `service|payment api -DEPENDS_ON-> database|redis ['p1e002'] high`,
each naming a real element id from the parsed asset.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add app/knowledge/pipeline.py tests/knowledge/test_pipeline.py
git status --short
```

---

## Task 7: BM25 over IR elements

**Files:**
- Create: `app/rag/bm25.py`
- Create: `tests/rag/test_bm25.py`

**Interfaces:**
- Consumes: `Document` (`app.ir.model`), `DocumentStore` (`app.storage.base`), `ElementRef` (Task 1).
- Produces: `Passage(doc_id, element_id, page_number, element_type, text, x0, y0, x1, y1, source_name)` with `.global_id`; `tokenize(text) -> list[str]`; `Bm25Index(passages, k1=1.5, b=0.75)` with `.search(query, limit=10) -> list[tuple[Passage, float]]`, `.passages_by_id: dict[str, Passage]`, `.from_documents(docs)`, `.from_store(store)`; `passage_from_element_ref(ref, source_name="") -> Passage`.

Identifiers are the reason this exists. `INC-2391` is tokenized both whole and
in parts, so a query for the identifier ranks its element top while a query for
`2391` still finds it.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_bm25.py`:

```python
from __future__ import annotations

from app.parsing.base import InMemoryBlobSink, parse_document
from app.rag.bm25 import Bm25Index, Passage, tokenize


def _passage(element_id: str, text: str, doc_id: str = "d") -> Passage:
    return Passage(
        doc_id=doc_id,
        element_id=element_id,
        page_number=1,
        element_type="paragraph",
        text=text,
        x0=0, y0=0, x1=10, y1=10,
        source_name="doc.md",
    )


CORPUS = [
    _passage("p1e001", "The Payment API depends on Redis for caching."),
    _passage("p1e002", "The Payments Team owns the Payment API service."),
    _passage("p1e003", "Incident INC-2391 affected the Payment API in production."),
    _passage("p1e004", "The ledger is stored in PostgreSQL for durability."),
    _passage("p1e005", "Error code PAYMENT_502 indicates an upstream timeout."),
]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("The Payment API, really!") == ["the", "payment", "api", "really"]


def test_tokenize_keeps_an_identifier_whole_and_in_parts():
    tokens = tokenize("Incident INC-2391 happened")
    assert "inc-2391" in tokens
    assert "inc" in tokens
    assert "2391" in tokens


def test_tokenize_handles_underscored_error_codes():
    tokens = tokenize("code PAYMENT_502 raised")
    assert "payment_502" in tokens
    assert "502" in tokens


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("   ") == []


def test_an_exact_identifier_ranks_its_own_element_first():
    index = Bm25Index(CORPUS)
    top, score = index.search("INC-2391")[0]
    assert top.element_id == "p1e003"
    assert score > 0


def test_an_underscored_error_code_is_found():
    index = Bm25Index(CORPUS)
    assert index.search("PAYMENT_502")[0][0].element_id == "p1e005"


def test_a_common_term_ranks_below_a_rare_one():
    index = Bm25Index(CORPUS)
    ranked = [passage.element_id for passage, _ in index.search("Payment API Redis")]
    assert ranked[0] == "p1e001"


def test_search_respects_the_limit():
    assert len(Bm25Index(CORPUS).search("payment", limit=2)) == 2


def test_a_query_matching_nothing_returns_nothing():
    assert Bm25Index(CORPUS).search("kubernetes helm chart") == []


def test_an_empty_query_returns_nothing():
    assert Bm25Index(CORPUS).search("   ") == []


def test_an_empty_index_searches_safely():
    assert Bm25Index([]).search("anything") == []


def test_passages_are_addressable_by_global_id():
    index = Bm25Index(CORPUS)
    assert index.passages_by_id["d:p1e003"].element_id == "p1e003"


def test_global_id_is_document_scoped():
    assert _passage("p1e001", "x", doc_id="abc").global_id == "abc:p1e001"


def test_scores_are_ordered_descending():
    scores = [score for _, score in Bm25Index(CORPUS).search("Payment API")]
    assert scores == sorted(scores, reverse=True)


def test_an_index_can_be_built_from_parsed_documents():
    doc = parse_document("app/assets/payment_system.md", blobs=InMemoryBlobSink())
    index = Bm25Index.from_documents([doc])

    hit, _score = index.search("Redis connection timeouts")[0]
    assert hit.doc_id == doc.id
    assert "Redis" in hit.text


def test_an_index_built_from_a_store_round_trips(tmp_path):
    from app.ingest import ingest
    from app.storage.filestore import FileDocumentStore

    store = FileDocumentStore(tmp_path)
    ingest("app/assets/payment_system.md", store)

    index = Bm25Index.from_store(store)
    assert index.search("Payments Team")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.bm25'`

- [ ] **Step 3: Implement `app/rag/bm25.py`**

```python
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.ir.model import Document
from app.knowledge.schema import ElementRef
from app.storage.base import DocumentStore

__all__ = ["Passage", "tokenize", "Bm25Index", "passage_from_element_ref"]

# A token is alphanumeric, optionally joined by - _ . so that INC-2391 and
# PAYMENT_502 survive tokenization intact.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_SPLIT = re.compile(r"[-_.]")

SKIPPED_TYPES = frozenset({"header", "footer", "page_number"})


@dataclass(frozen=True)
class Passage:
    doc_id: str
    element_id: str
    page_number: int
    element_type: str
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source_name: str = ""

    @property
    def global_id(self) -> str:
        return f"{self.doc_id}:{self.element_id}"


def passage_from_element_ref(ref: ElementRef, source_name: str = "") -> Passage:
    return Passage(
        doc_id=ref.doc_id,
        element_id=ref.id,
        page_number=ref.page_number,
        element_type=ref.type,
        text=ref.text,
        x0=ref.x0, y0=ref.y0, x1=ref.x1, y1=ref.y1,
        source_name=source_name,
    )


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, with compound identifiers kept whole and also split.

    Keeping only the whole token would lose a query for `2391`; keeping only the
    parts would lose the precision of the full identifier. Emitting both costs
    little and makes identifier search reliable.
    """
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if _SPLIT.search(match):
            tokens.extend(part for part in _SPLIT.split(match) if part)
    return tokens


class Bm25Index:
    """Okapi BM25 over IR element text.

    Held in memory and rebuilt when the corpus changes. Fine for tens of
    documents; real indexing belongs to the serving layer.
    """

    def __init__(
        self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.passages = list(passages)
        self.passages_by_id = {passage.global_id: passage for passage in self.passages}

        self._tokens = [tokenize(passage.text) for passage in self.passages]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._frequencies = [Counter(tokens) for tokens in self._tokens]

        document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())

        total = len(self.passages)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> Bm25Index:
        passages: list[Passage] = []
        for document in documents:
            for element in document.iter_elements():
                text = element.text_content().strip()
                if not text or element.type.value in SKIPPED_TYPES:
                    continue
                passages.append(
                    Passage(
                        doc_id=document.id,
                        element_id=element.id,
                        page_number=element.page_number,
                        element_type=element.type.value,
                        text=text,
                        x0=element.bbox.x0,
                        y0=element.bbox.y0,
                        x1=element.bbox.x1,
                        y1=element.bbox.y1,
                        source_name=document.source_name,
                    )
                )
        return cls(passages)

    @classmethod
    def from_store(cls, store: DocumentStore) -> Bm25Index:
        return cls.from_documents(store.get(doc_id) for doc_id in store.list_ids())

    def search(self, query: str, limit: int = 10) -> list[tuple[Passage, float]]:
        terms = tokenize(query)
        if not terms or not self.passages:
            return []

        scored: list[tuple[Passage, float]] = []
        for index, frequencies in enumerate(self._frequencies):
            length = self._lengths[index]
            if not length:
                continue
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._average_length or 1)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((self.passages[index], score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].global_id))
        return scored[:limit]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rag/test_bm25.py -v`
Expected: PASS, 15 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/rag/bm25.py tests/rag
git status --short
```

---

## Task 8: Retrievers and rank fusion

**Files:**
- Create: `app/rag/retriever.py`
- Create: `tests/rag/test_retriever.py`

**Interfaces:**
- Consumes: `GraphStore`, `FactRecord` (Task 4); `Bm25Index`, `Passage`, `passage_from_element_ref` (Task 7); `StructuredLLM`, `QuestionEntities`, `ResolvedEntity` (Task 1).
- Produces: `Evidence(passage, score, sources)`; `RetrievalResult(question, evidence, facts, seeds, trace)` with `.is_empty`; `reciprocal_rank_fusion(rankings, k=60) -> dict[str, float]`; `QuestionAnalyzer(llm=None)` with `.entity_names(question) -> list[str]`; `Retriever` protocol with `.retrieve(question, *, limit=8) -> RetrievalResult`; `GraphRetriever(store, analyzer, hops=2)`; `Bm25Retriever(index)`; `HybridRetriever(graph, lexical)`.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_retriever.py`:

```python
from __future__ import annotations

from app.graph.memory import InMemoryGraphStore
from app.knowledge.llm import FakeLLM
from app.knowledge.schema import (
    DocumentRef,
    ElementRef,
    EntityType,
    GraphFact,
    GraphUpsert,
    QuestionEntities,
    RelationType,
    ResolvedEntity,
    entity_key,
)
from app.rag.bm25 import Bm25Index, Passage
from app.rag.retriever import (
    Bm25Retriever,
    GraphRetriever,
    HybridRetriever,
    QuestionAnalyzer,
    reciprocal_rank_fusion,
)

PAYMENT = entity_key("Payment API", EntityType.SERVICE)
TEAM = entity_key("Payments Team", EntityType.TEAM)
INCIDENT = entity_key("INC-2391", EntityType.INCIDENT)


def _element(eid: str, text: str) -> ElementRef:
    return ElementRef(
        id=eid, doc_id="d", type="paragraph", page_number=1,
        x0=0, y0=0, x1=10, y1=10, order=0, text=text,
    )


def _store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.apply(
        GraphUpsert(
            document=DocumentRef(id="d", source_name="p.md", mime="text/markdown"),
            entities=[
                ResolvedEntity(key=PAYMENT, name="Payment API", type=EntityType.SERVICE),
                ResolvedEntity(key=TEAM, name="Payments Team", type=EntityType.TEAM),
                ResolvedEntity(key=INCIDENT, name="INC-2391", type=EntityType.INCIDENT),
            ],
            facts=[
                GraphFact(
                    source_key=INCIDENT, target_key=PAYMENT,
                    relation=RelationType.AFFECTS, doc_id="d",
                    element_ids=["p1e003"], evidence="INC-2391 affected the Payment API",
                ),
                GraphFact(
                    source_key=TEAM, target_key=PAYMENT, relation=RelationType.OWNS,
                    doc_id="d", element_ids=["p1e002"],
                    evidence="The Payments Team owns the Payment API",
                ),
            ],
            elements=[
                _element("p1e002", "The Payments Team owns the Payment API."),
                _element("p1e003", "INC-2391 affected the Payment API."),
            ],
        )
    )
    return store


def _passage(element_id: str, text: str) -> Passage:
    return Passage(
        doc_id="d", element_id=element_id, page_number=1, element_type="paragraph",
        text=text, x0=0, y0=0, x1=10, y1=10, source_name="p.md",
    )


def _index() -> Bm25Index:
    return Bm25Index(
        [
            _passage("p1e001", "The Payment API depends on Redis for caching."),
            _passage("p1e002", "The Payments Team owns the Payment API."),
            _passage("p1e003", "INC-2391 affected the Payment API."),
        ]
    )


def test_rrf_favours_an_item_ranked_well_by_both_systems():
    scores = reciprocal_rank_fusion({"graph": ["a", "b"], "bm25": ["b", "a"]})
    assert scores["a"] == scores["b"]

    scores = reciprocal_rank_fusion({"graph": ["a", "b"], "bm25": ["a", "b"]})
    assert scores["a"] > scores["b"]


def test_rrf_includes_items_seen_by_only_one_system():
    scores = reciprocal_rank_fusion({"graph": ["a"], "bm25": ["b"]})
    assert set(scores) == {"a", "b"}


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion({}) == {}


def test_the_analyzer_asks_the_model_for_entity_names():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    assert QuestionAnalyzer(llm).entity_names("What did INC-2391 affect?") == ["INC-2391"]


def test_the_analyzer_falls_back_to_a_heuristic_without_a_model():
    names = QuestionAnalyzer(None).entity_names("Who owns the Payment API after INC-2391?")
    assert "Payment API" in names or "INC-2391" in names


def test_the_analyzer_survives_a_model_failure():
    llm = FakeLLM(responses=[RuntimeError("groq down")])
    assert QuestionAnalyzer(llm).entity_names("Who owns the Payment API?")


def test_graph_retrieval_answers_a_two_hop_question():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    retriever = GraphRetriever(_store(), QuestionAnalyzer(llm), hops=2)

    result = retriever.retrieve("Which team owns the service affected by INC-2391?")

    assert [seed.key for seed in result.seeds] == [INCIDENT]
    assert any(fact.relation is RelationType.OWNS for fact in result.facts)
    assert {e.passage.element_id for e in result.evidence} == {"p1e002", "p1e003"}


def test_graph_evidence_is_tagged_with_its_source():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    result = GraphRetriever(_store(), QuestionAnalyzer(llm)).retrieve("q")
    assert all("graph" in evidence.sources for evidence in result.evidence)


def test_graph_retrieval_is_empty_when_no_entity_matches():
    llm = FakeLLM(responses=[QuestionEntities(names=["Kubernetes"])])
    result = GraphRetriever(_store(), QuestionAnalyzer(llm)).retrieve("q")

    assert result.is_empty
    assert result.facts == []


def test_lexical_retrieval_finds_the_identifier():
    result = Bm25Retriever(_index()).retrieve("INC-2391")
    assert result.evidence[0].passage.element_id == "p1e003"
    assert result.evidence[0].sources == ("bm25",)


def test_hybrid_retrieval_merges_both_rankings():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(
        GraphRetriever(_store(), QuestionAnalyzer(llm)), Bm25Retriever(_index())
    )

    result = hybrid.retrieve("Which team owns the service affected by INC-2391?")
    found = {evidence.passage.element_id for evidence in result.evidence}

    assert {"p1e002", "p1e003"} <= found
    assert result.facts


def test_hybrid_marks_evidence_found_by_both_systems():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(
        GraphRetriever(_store(), QuestionAnalyzer(llm)), Bm25Retriever(_index())
    )

    result = hybrid.retrieve("INC-2391 Payment API")
    both = [e for e in result.evidence if set(e.sources) == {"graph", "bm25"}]
    assert both


def test_hybrid_degrades_to_lexical_when_the_graph_is_empty():
    llm = FakeLLM(responses=[QuestionEntities(names=[])])
    hybrid = HybridRetriever(
        GraphRetriever(InMemoryGraphStore(), QuestionAnalyzer(llm)),
        Bm25Retriever(_index()),
    )

    result = hybrid.retrieve("Payments Team")
    assert result.evidence
    assert result.facts == []


def test_hybrid_respects_the_limit():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(
        GraphRetriever(_store(), QuestionAnalyzer(llm)), Bm25Retriever(_index())
    )
    assert len(hybrid.retrieve("Payment API", limit=1).evidence) == 1


def test_the_trace_records_what_retrieval_did():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(
        GraphRetriever(_store(), QuestionAnalyzer(llm)), Bm25Retriever(_index())
    )

    trace = hybrid.retrieve("INC-2391").trace
    assert trace["seed_names"] == ["INC-2391"]
    assert trace["graph_hits"] >= 1
    assert trace["bm25_hits"] >= 1
    assert "hops" in trace
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.retriever'`

- [ ] **Step 3: Implement `app/rag/retriever.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from app.graph.store import FactRecord, GraphStore
from app.knowledge.llm import StructuredLLM
from app.knowledge.schema import QuestionEntities, ResolvedEntity
from app.rag.bm25 import Bm25Index, Passage, passage_from_element_ref

__all__ = [
    "Evidence",
    "RetrievalResult",
    "Retriever",
    "QuestionAnalyzer",
    "GraphRetriever",
    "Bm25Retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "RRF_K",
]

RRF_K = 60
DEFAULT_LIMIT = 8

QUESTION_PROMPT = """\
List the named entities mentioned in this question — services, teams, people,
products, databases, incidents, error codes, identifiers.

Return names only, exactly as written in the question. Do not add entities that
are not present, and do not include generic words like "team" or "service" on
their own.

Question: {question}
"""

# Fallback when no model is available: runs of capitalised words, plus anything
# shaped like an identifier (INC-2391, PAYMENT_502).
_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ -][A-Z][A-Za-z0-9]*)*\b")
_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*[-_][A-Z0-9]+\b")
_STOPWORDS = {"What", "Which", "Who", "Where", "When", "Why", "How", "The", "A", "An"}


@dataclass(frozen=True)
class Evidence:
    passage: Passage
    score: float
    sources: tuple[str, ...]


@dataclass
class RetrievalResult:
    question: str
    evidence: list[Evidence] = field(default_factory=list)
    facts: list[FactRecord] = field(default_factory=list)
    seeds: list[ResolvedEntity] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.evidence and not self.facts


class Retriever(Protocol):
    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult: ...


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists without weights to tune.

    Each list contributes 1/(k + rank). An item both systems rank highly beats
    one that only a single system likes, which is the whole point of running a
    graph search and a lexical search side by side.
    """
    scores: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    return scores


class QuestionAnalyzer:
    """Pull candidate entity names out of a question."""

    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self.llm = llm

    def entity_names(self, question: str) -> list[str]:
        if self.llm is not None:
            try:
                reply = self.llm.structured(
                    QUESTION_PROMPT.format(question=question), QuestionEntities
                )
                names = [name.strip() for name in reply.names if name.strip()]
                if names:
                    return names
            except Exception:
                # A question is still answerable without entity extraction.
                pass
        return self._heuristic(question)

    @staticmethod
    def _heuristic(question: str) -> list[str]:
        found = list(_IDENTIFIER.findall(question))
        for match in _CAPITALISED.findall(question):
            cleaned = match.strip()
            if cleaned and cleaned not in _STOPWORDS and cleaned not in found:
                found.append(cleaned)
        return found


class GraphRetriever:
    """Seed on entities named in the question, then expand k hops."""

    def __init__(
        self, store: GraphStore, analyzer: QuestionAnalyzer, hops: int = 2
    ) -> None:
        self.store = store
        self.analyzer = analyzer
        self.hops = max(1, min(hops, 3))

    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        names = self.analyzer.entity_names(question)
        seeds = self.store.find_entities(names) if names else []
        trace: dict[str, Any] = {
            "seed_names": names,
            "seed_keys": [seed.key for seed in seeds],
            "hops": self.hops,
        }

        if not seeds:
            trace["graph_hits"] = 0
            return RetrievalResult(question=question, seeds=[], trace=trace)

        facts = self.store.traverse([seed.key for seed in seeds], hops=self.hops)
        trace["graph_hits"] = len(facts)

        ordered_ids: list[str] = []
        for fact in facts:
            for global_id in fact.global_element_ids():
                if global_id not in ordered_ids:
                    ordered_ids.append(global_id)

        refs = {ref.global_id: ref for ref in self.store.elements(ordered_ids)}
        evidence = [
            Evidence(
                passage=passage_from_element_ref(refs[global_id]),
                score=1.0 / rank,
                sources=("graph",),
            )
            for rank, global_id in enumerate(ordered_ids, start=1)
            if global_id in refs
        ]

        return RetrievalResult(
            question=question,
            evidence=evidence[:limit],
            facts=facts,
            seeds=seeds,
            trace=trace,
        )


class Bm25Retriever:
    def __init__(self, index: Bm25Index) -> None:
        self.index = index

    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        hits = self.index.search(question, limit=limit)
        return RetrievalResult(
            question=question,
            evidence=[
                Evidence(passage=passage, score=score, sources=("bm25",))
                for passage, score in hits
            ],
            trace={"bm25_hits": len(hits)},
        )


class HybridRetriever:
    """Graph structure plus lexical matching, fused with RRF."""

    def __init__(self, graph: GraphRetriever, lexical: Bm25Retriever) -> None:
        self.graph = graph
        self.lexical = lexical

    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        # Over-fetch each arm so fusion has something to work with.
        graph_result = self.graph.retrieve(question, limit=limit * 3)
        lexical_result = self.lexical.retrieve(question, limit=limit * 3)

        passages: dict[str, Passage] = {}
        rankings: dict[str, list[str]] = {"graph": [], "bm25": []}

        for name, result in (("graph", graph_result), ("bm25", lexical_result)):
            for evidence in result.evidence:
                global_id = evidence.passage.global_id
                passages.setdefault(global_id, evidence.passage)
                rankings[name].append(global_id)

        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))

        graph_ids = set(rankings["graph"])
        lexical_ids = set(rankings["bm25"])
        evidence: list[Evidence] = []
        for global_id, score in ordered[:limit]:
            sources = tuple(
                name
                for name, ids in (("graph", graph_ids), ("bm25", lexical_ids))
                if global_id in ids
            )
            evidence.append(
                Evidence(passage=passages[global_id], score=score, sources=sources)
            )

        trace = {
            **graph_result.trace,
            **lexical_result.trace,
            "fused": len(fused),
            "returned": len(evidence),
        }
        return RetrievalResult(
            question=question,
            evidence=evidence,
            facts=graph_result.facts,
            seeds=graph_result.seeds,
            trace=trace,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rag -v`
Expected: PASS, 30 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/rag/retriever.py tests/rag/test_retriever.py
git status --short
```

---

## Task 9: Grounded answering with validated citations

**Files:**
- Create: `app/rag/answer.py`
- Create: `tests/rag/test_answer.py`

**Interfaces:**
- Consumes: `StructuredLLM`, `FakeLLM` (Task 1); `RetrievalResult`, `Evidence` (Task 8).
- Produces: `DraftAnswer(answer)`; `Citation(global_id, doc_id, element_id, page_number, bbox, text, source_name)`; `Answer(text, citations, dropped_citations, grounded, fact_count)`; `NO_EVIDENCE: str`; `ANSWER_PROMPT`; `CITATION_PATTERN`; `build_context(result) -> str`; `answer_question(llm, result) -> Answer`.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_answer.py`:

```python
from __future__ import annotations

from app.graph.store import FactRecord
from app.knowledge.llm import FakeLLM
from app.knowledge.schema import EntityType, RelationType
from app.rag.answer import NO_EVIDENCE, DraftAnswer, answer_question, build_context
from app.rag.bm25 import Passage
from app.rag.retriever import Evidence, RetrievalResult


def _evidence(element_id: str, text: str) -> Evidence:
    return Evidence(
        passage=Passage(
            doc_id="d", element_id=element_id, page_number=3, element_type="paragraph",
            text=text, x0=72, y0=100, x1=500, y1=140, source_name="payments.pdf",
        ),
        score=0.5,
        sources=("graph",),
    )


def _fact() -> FactRecord:
    return FactRecord(
        source_key="team|payments team", source_name="Payments Team",
        source_type=EntityType.TEAM, relation=RelationType.OWNS,
        target_key="service|payment api", target_name="Payment API",
        target_type=EntityType.SERVICE, doc_id="d", element_ids=["p1e002"],
        evidence="The Payments Team owns the Payment API.", hop=2,
    )


def _result(**overrides) -> RetrievalResult:
    base = {
        "question": "Which team owns the Payment API?",
        "evidence": [_evidence("p1e002", "The Payments Team owns the Payment API.")],
        "facts": [_fact()],
    }
    base.update(overrides)
    return RetrievalResult(**base)


def test_context_includes_the_facts_and_the_evidence():
    context = build_context(_result())
    assert "Payments Team OWNS Payment API" in context
    assert "[d:p1e002]" in context
    assert "payments.pdf" in context
    assert "page 3" in context


def test_context_of_an_empty_result_is_empty():
    assert build_context(RetrievalResult(question="q")) == ""


def test_an_empty_retrieval_never_reaches_the_model():
    llm = FakeLLM(responses=[])
    answer = answer_question(llm, RetrievalResult(question="q"))

    assert answer.text == NO_EVIDENCE
    assert answer.grounded is False
    assert llm.prompts == []


def test_a_grounded_answer_keeps_resolvable_citations():
    llm = FakeLLM(
        responses=[DraftAnswer(answer="The Payments Team owns it [d:p1e002].")]
    )
    answer = answer_question(llm, _result())

    assert answer.grounded is True
    assert [citation.global_id for citation in answer.citations] == ["d:p1e002"]
    assert answer.citations[0].page_number == 3
    assert answer.citations[0].bbox == (72.0, 100.0, 500.0, 140.0)


def test_a_fabricated_citation_is_dropped_and_reported():
    llm = FakeLLM(
        responses=[
            DraftAnswer(answer="Owned by the team [d:p1e002], see also [d:p9e999].")
        ]
    )
    answer = answer_question(llm, _result())

    assert [c.global_id for c in answer.citations] == ["d:p1e002"]
    assert answer.dropped_citations == ["d:p9e999"]


def test_an_answer_with_no_citations_is_not_marked_grounded():
    llm = FakeLLM(responses=[DraftAnswer(answer="The Payments Team owns it.")])
    answer = answer_question(llm, _result())

    assert answer.citations == []
    assert answer.grounded is False


def test_duplicate_citations_are_reported_once():
    llm = FakeLLM(
        responses=[DraftAnswer(answer="See [d:p1e002] and again [d:p1e002].")]
    )
    assert len(answer_question(llm, _result()).citations) == 1


def test_the_prompt_forbids_answering_beyond_the_evidence():
    llm = FakeLLM(responses=[DraftAnswer(answer="ok [d:p1e002]")])
    answer_question(llm, _result())

    prompt = llm.prompts[0].lower()
    assert "only" in prompt
    assert "evidence" in prompt


def test_the_prompt_carries_the_question_and_the_context():
    llm = FakeLLM(responses=[DraftAnswer(answer="ok [d:p1e002]")])
    answer_question(llm, _result())

    assert "Which team owns the Payment API?" in llm.prompts[0]
    assert "[d:p1e002]" in llm.prompts[0]


def test_a_model_failure_surfaces_as_an_ungrounded_answer():
    llm = FakeLLM(responses=[RuntimeError("groq exploded")])
    answer = answer_question(llm, _result())

    assert answer.grounded is False
    assert "groq exploded" in answer.text


def test_the_fact_count_is_reported():
    llm = FakeLLM(responses=[DraftAnswer(answer="ok [d:p1e002]")])
    assert answer_question(llm, _result()).fact_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/rag/test_answer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.answer'`

- [ ] **Step 3: Implement `app/rag/answer.py`**

```python
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.knowledge.llm import StructuredLLM
from app.rag.retriever import RetrievalResult

__all__ = [
    "DraftAnswer",
    "Citation",
    "Answer",
    "build_context",
    "answer_question",
    "NO_EVIDENCE",
    "ANSWER_PROMPT",
    "CITATION_PATTERN",
]

NO_EVIDENCE = (
    "No evidence was found in the ingested documents for this question. "
    "Try ingesting the relevant document, or rephrasing using terms that "
    "appear in it."
)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.\-]+:p\d+e\d+)\]")

ANSWER_PROMPT = """\
Answer the question using only the evidence below. The evidence is everything
you know; your own knowledge is not admissible here.

Rules:
1. Use only the facts and evidence provided. If they do not answer the
   question, say so plainly and stop.
2. Cite every claim with the evidence id in square brackets, exactly as shown,
   for example [{example}].
3. Never invent an evidence id. Only ids that appear below exist.
4. Be direct. Two or three sentences is usually enough.

Question: {question}

{context}
"""


class DraftAnswer(BaseModel):
    answer: str = Field(description="The answer, with [evidence-id] citations")


class Citation(BaseModel):
    global_id: str
    doc_id: str
    element_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    source_name: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    grounded: bool = False
    fact_count: int = 0


def build_context(result: RetrievalResult) -> str:
    """Render facts and evidence into the only material the model may use."""
    if result.is_empty:
        return ""

    sections: list[str] = []

    if result.facts:
        lines = [
            f"- {fact.sentence()}  (hop {fact.hop}, from "
            f"{', '.join(fact.global_element_ids()) or 'unknown'})"
            for fact in result.facts
        ]
        sections.append("Facts from the knowledge graph:\n" + "\n".join(lines))

    if result.evidence:
        blocks = []
        for evidence in result.evidence:
            passage = evidence.passage
            location = f"{passage.source_name or passage.doc_id}, page {passage.page_number}"
            blocks.append(
                f"[{passage.global_id}] ({location})\n{passage.text.strip()}"
            )
        sections.append("Evidence:\n\n" + "\n\n".join(blocks))

    return "\n\n".join(sections)


def answer_question(llm: StructuredLLM, result: RetrievalResult) -> Answer:
    """Answer strictly from retrieved evidence, validating every citation.

    A citation naming an element that was not retrieved is dropped rather than
    rendered: an invented id that looks real is worse than a missing one.
    """
    if result.is_empty:
        return Answer(text=NO_EVIDENCE, grounded=False)

    context = build_context(result)
    known = {evidence.passage.global_id: evidence.passage for evidence in result.evidence}
    example = next(iter(known), "doc:p1e000")

    prompt = ANSWER_PROMPT.format(
        question=result.question, context=context, example=example
    )

    try:
        draft = llm.structured(prompt, DraftAnswer)
    except Exception as error:
        return Answer(
            text=f"The answer could not be generated: {error}",
            grounded=False,
            fact_count=len(result.facts),
        )

    text = draft.answer.strip()
    citations: list[Citation] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for global_id in CITATION_PATTERN.findall(text):
        if global_id in seen:
            continue
        seen.add(global_id)
        passage = known.get(global_id)
        if passage is None:
            dropped.append(global_id)
            continue
        citations.append(
            Citation(
                global_id=global_id,
                doc_id=passage.doc_id,
                element_id=passage.element_id,
                page_number=passage.page_number,
                bbox=(passage.x0, passage.y0, passage.x1, passage.y1),
                text=passage.text,
                source_name=passage.source_name,
            )
        )

    return Answer(
        text=text,
        citations=citations,
        dropped_citations=dropped,
        grounded=bool(citations),
        fact_count=len(result.facts),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/rag -v`
Expected: PASS, 41 tests.

- [ ] **Step 5: Stage the changes (do not commit)**

```bash
git add app/rag/answer.py tests/rag/test_answer.py
git status --short
```

---

## Task 10: UI — shared services and three modes

**Files:**
- Create: `app/viewer/services.py`, `app/viewer/modes/inspect.py`, `app/viewer/modes/ingest.py`, `app/viewer/modes/ask.py`
- Modify: `app/viewer/main.py` (becomes routing only)
- Create: `tests/viewer/test_services.py`
- Modify: `tests/viewer/test_app.py` (add smoke coverage for the new modes)

**Interfaces:**
- Consumes: everything from Tasks 1–9; `parse_document`, `InMemoryBlobSink` (`app.parsing.base`); `FileDocumentStore` (`app.storage.filestore`); `render_page`, `RenderOptions` (`app.viewer.render`); `panels` (`app.viewer.panels`).
- Produces: `services.STORE_DIR`, `services.document_store()`, `services.graph_store() -> tuple[GraphStore | None, str | None]`, `services.language_model() -> tuple[StructuredLLM | None, str | None]`, `services.bm25_index() -> Bm25Index`, `services.refresh_index()`, `services.build_retriever(graph, index, hops) -> HybridRetriever`, `services.bundled_sources() -> dict[str, Path]`, `services.parse_source(name, data, config) -> tuple[Document, dict[str, bytes]]`; `inspect.render()`, `ingest.render()`, `ask.render()`.

- [ ] **Step 1: Write the failing test**

Create `tests/viewer/test_services.py`:

```python
from __future__ import annotations

from pathlib import Path

from app.viewer import services


def test_bundled_sources_includes_the_project_asset_and_fixtures():
    found = services.bundled_sources()
    assert "payment_system.md" in found
    assert any(name.endswith(".pdf") for name in found)
    assert all(isinstance(path, Path) and path.is_file() for path in found.values())


def test_parse_source_returns_a_document_and_its_blobs():
    path = services.bundled_sources()["payment_system.md"]
    document, blobs = services.parse_source(
        path.name, path.read_bytes(), services.default_config_key()
    )

    assert document.title == "Payment System"
    assert blobs == {}


def test_parse_source_keeps_the_original_file_name():
    path = services.bundled_sources()["payment_system.md"]
    document, _ = services.parse_source(
        "renamed.md", path.read_bytes(), services.default_config_key()
    )
    assert document.source_name == "renamed.md"


def test_build_retriever_wires_graph_and_lexical_arms():
    from app.graph.memory import InMemoryGraphStore
    from app.rag.bm25 import Bm25Index

    retriever = services.build_retriever(InMemoryGraphStore(), Bm25Index([]), hops=2)
    result = retriever.retrieve("anything at all")

    assert result.is_empty
    assert result.trace["hops"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/viewer/test_services.py -v`
Expected: FAIL — `ImportError: cannot import name 'services'`

- [ ] **Step 3: Implement `app/viewer/services.py`**

```python
"""Shared, cached wiring for the viewer's three modes.

Connections are cached as resources; parsing and indexing as data. Every
external dependency degrades to `None` with a message rather than raising, so
the UI stays usable when Neo4j or Groq is unavailable.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.neo4j import Neo4jGraphStore  # noqa: E402
from app.graph.store import GraphStore  # noqa: E402
from app.ir.model import Document  # noqa: E402
from app.knowledge.llm import GroqLLM, StructuredLLM  # noqa: E402
from app.parsing.base import InMemoryBlobSink, parse_document  # noqa: E402
from app.parsing.layout import LayoutConfig  # noqa: E402
from app.parsing.pdf import PdfParser  # noqa: E402
from app.rag.bm25 import Bm25Index  # noqa: E402
from app.rag.retriever import (  # noqa: E402
    Bm25Retriever,
    GraphRetriever,
    HybridRetriever,
    QuestionAnalyzer,
)
from app.storage.filestore import FileDocumentStore  # noqa: E402

__all__ = [
    "ROOT",
    "STORE_DIR",
    "SUPPORTED",
    "document_store",
    "graph_store",
    "language_model",
    "bm25_index",
    "refresh_index",
    "build_retriever",
    "bundled_sources",
    "parse_source",
    "default_config_key",
]

STORE_DIR = ROOT / "store"
SUPPORTED = ["pdf", "md", "markdown", "png", "jpg", "jpeg", "xlsx"]


def document_store() -> FileDocumentStore:
    return FileDocumentStore(STORE_DIR)


@st.cache_resource(show_spinner=False)
def graph_store() -> tuple[GraphStore | None, str | None]:
    """Connect to Neo4j. Returns (store, error) — never raises."""
    try:
        from app.config import settings

        store = Neo4jGraphStore(
            settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password
        )
        store.verify_connectivity()
        store.ensure_schema()
        return store, None
    except Exception as error:
        return None, str(error)


@st.cache_resource(show_spinner=False)
def language_model() -> tuple[StructuredLLM | None, str | None]:
    try:
        return GroqLLM(), None
    except Exception as error:
        return None, str(error)


@st.cache_data(show_spinner=False)
def _index_for(doc_ids: tuple[str, ...]) -> Bm25Index:
    return Bm25Index.from_store(document_store())


def bm25_index() -> Bm25Index:
    """Lexical index over everything committed, rebuilt when the corpus changes."""
    return _index_for(tuple(document_store().list_ids()))


def refresh_index() -> None:
    """Drop the cached lexical index after the corpus changes."""
    _index_for.clear()


def build_retriever(
    graph: GraphStore, index: Bm25Index, hops: int = 2
) -> HybridRetriever:
    model, _error = language_model()
    analyzer = QuestionAnalyzer(model)
    return HybridRetriever(
        GraphRetriever(graph, analyzer, hops=hops), Bm25Retriever(index)
    )


def bundled_sources() -> dict[str, Path]:
    found: dict[str, Path] = {}
    asset = ROOT / "app" / "assets" / "payment_system.md"
    if asset.is_file():
        found[asset.name] = asset
    fixtures = ROOT / "tests" / "fixtures" / "files"
    if fixtures.is_dir():
        for path in sorted(fixtures.iterdir()):
            if path.is_file() and path.suffix.lstrip(".").lower() in SUPPORTED:
                found[path.name] = path
    return found


def default_config_key() -> tuple:
    return tuple(sorted(asdict(LayoutConfig()).items()))


@st.cache_data(show_spinner="Parsing…", max_entries=24)
def parse_source(name: str, data: bytes, config: tuple) -> tuple[Document, dict]:
    """Parse bytes into the IR. Cached on content plus layout configuration."""
    suffix = Path(name).suffix or ".bin"
    sink = InMemoryBlobSink()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        if suffix.lower() == ".pdf":
            document = PdfParser(LayoutConfig(**dict(config))).parse(
                temporary, blobs=sink
            )
        else:
            document = parse_document(temporary, blobs=sink)
    finally:
        temporary.unlink(missing_ok=True)

    document.source_name = name
    return document, sink.blobs
```

- [ ] **Step 4: Move the Inspect mode into `app/viewer/modes/inspect.py`**

Move the existing `sidebar()` function and the body of `main()` out of
`app/viewer/main.py` into `app/viewer/modes/inspect.py`, making exactly these
changes and no others:

1. Rename `main()` to `render()`.
2. Delete the `st.set_page_config(...)` call and the `st.markdown(CSS, ...)`
   call from it — `main.py` now owns both.
3. Delete the local `bundled_sources()` and `parse_source()` definitions and
   import them from `app.viewer.services` instead.
4. Move the `TUNABLE` tuple from `main.py` into `inspect.py` verbatim — it is
   used only by this mode's sidebar. Delete the `SUPPORTED` list from
   `main.py` and import it from `app.viewer.services` instead.
5. Change the `st.title("Document IR")` line to `st.subheader("Document IR")`,
   since `main.py` now renders the page title.

The resulting module exposes exactly one public name: `render()`.

- [ ] **Step 5: Implement `app/viewer/modes/ingest.py`**

```python
"""Parse a document, preview what the model extracted, then commit it."""

from __future__ import annotations

import streamlit as st

from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import GraphUpsert
from app.parsing.base import ParseError
from app.viewer import panels, services

__all__ = ["render"]


def _source_picker() -> tuple[str | None, bytes | None]:
    bundled = services.bundled_sources()
    mode = st.sidebar.radio(
        "Source", ["Bundled", "Upload"], horizontal=True, key="ingest_source"
    )
    if mode == "Bundled" and bundled:
        choice = st.sidebar.selectbox("Document", list(bundled), key="ingest_doc")
        return choice, bundled[choice].read_bytes()
    if mode == "Upload":
        upload = st.sidebar.file_uploader(
            "Document", type=services.SUPPORTED, key="ingest_upload"
        )
        if upload is not None:
            return upload.name, upload.getvalue()
    return None, None


def _preview(upsert: GraphUpsert) -> None:
    columns = st.columns(4)
    columns[0].metric("Entities", len(upsert.entities))
    columns[1].metric("Facts", len(upsert.facts))
    columns[2].metric("Evidence elements", len(upsert.elements))
    columns[3].metric("Warnings", len(upsert.warnings))

    tabs = st.tabs(["Facts", "Entities", "Warnings"])

    with tabs[0]:
        if not upsert.facts:
            st.info("No facts extracted. The text may not state any relationships.")
        else:
            names = {entity.key: entity.name for entity in upsert.entities}
            st.dataframe(
                [
                    {
                        "source": names.get(fact.source_key, fact.source_key),
                        "relation": fact.relation.value,
                        "target": names.get(fact.target_key, fact.target_key),
                        "evidence": fact.evidence,
                        "elements": ", ".join(fact.element_ids),
                        "confidence": fact.confidence,
                    }
                    for fact in upsert.facts
                ],
                hide_index=True,
            )
            st.caption(
                "`confidence` is low when the evidence quote could not be matched "
                "to a single element, so provenance is chunk-level for that fact."
            )

    with tabs[1]:
        st.dataframe(
            [
                {
                    "name": entity.name,
                    "type": entity.type.value,
                    "key": entity.key,
                    "aliases": ", ".join(entity.aliases),
                }
                for entity in upsert.entities
            ],
            hide_index=True,
        )

    with tabs[2]:
        if upsert.warnings:
            st.dataframe(upsert.warnings, hide_index=True)
        else:
            st.success("No warnings — every chunk extracted cleanly.")


def render() -> None:
    st.subheader("Ingest")
    st.caption(
        "Parse a document, review what the model claims and the quote it claims "
        "it from, then commit it to the store and the graph."
    )

    graph, graph_error = services.graph_store()
    model, model_error = services.language_model()

    if graph_error:
        st.warning(f"Neo4j unavailable — committing is disabled. {graph_error}")
    if model_error:
        st.error(f"Groq unavailable — extraction is disabled. {model_error}")

    name, data = _source_picker()
    if name is None or data is None:
        st.info("Pick a bundled document or upload one to begin.")
        _danger_zone(graph)
        return

    try:
        document, blobs = services.parse_source(
            name, data, services.default_config_key()
        )
    except ParseError as error:
        st.error(f"Could not parse **{name}**\n\n```\n{error}\n```")
        return

    st.markdown(f"**{document.title or document.source_name}** · `{document.id}`")
    panels.metrics_strip(document)

    state_key = f"upsert::{document.id}"

    if st.button("Extract knowledge", type="primary", disabled=model is None):
        known = graph.known_entities() if graph else []
        pipeline = KnowledgePipeline(KnowledgeExtractor(model), EntityResolver(model))
        progress = st.progress(0.0, text="Extracting…")

        def report(done: int, total: int) -> None:
            progress.progress(done / max(total, 1), text=f"Chunk {done} of {total}")

        try:
            st.session_state[state_key] = pipeline.run(
                document, known, on_progress=report
            )
        except Exception as error:
            st.error(f"Extraction failed: {error}")
        finally:
            progress.empty()

    upsert = st.session_state.get(state_key)
    if upsert is None:
        _danger_zone(graph)
        return

    _preview(upsert)

    if st.button("Commit to store and graph", type="primary", disabled=graph is None):
        # The store is written first: a document that is searchable without its
        # graph edges is recoverable, whereas edges citing elements no store can
        # resolve are dead citations.
        services.document_store().put(document, blobs)
        stats = graph.apply(upsert)
        services.refresh_index()  # the corpus grew; drop the cached index

        st.success(
            f"Committed {stats.entities} entities, {stats.facts} facts and "
            f"{stats.elements} evidence elements from **{document.source_name}**."
        )
        totals = graph.stats()
        st.caption(
            f"Graph now holds {totals.documents} documents, {totals.entities} "
            f"entities and {totals.facts} facts."
        )

    _danger_zone(graph)


def _danger_zone(graph) -> None:
    with st.expander("Danger zone", expanded=False):
        st.caption(
            "Deletes every Document, Entity and Element node. This cannot be "
            "undone."
        )
        confirmed = st.checkbox("I understand this deletes the graph", key="reset_ok")
        if st.button("Reset graph", disabled=graph is None or not confirmed):
            graph.reset()
            services.refresh_index()
            st.success("Graph reset.")
```

- [ ] **Step 6: Implement `app/viewer/modes/ask.py`**

```python
"""Ask a question; get an answer grounded in retrieved evidence."""

from __future__ import annotations

import time

import streamlit as st

from app.rag.answer import answer_question
from app.viewer import services

__all__ = ["render"]


def _evidence_card(evidence) -> None:
    passage = evidence.passage
    badges = " · ".join(evidence.sources)
    st.markdown(
        f"**`{passage.global_id}`** · {passage.source_name or passage.doc_id} · "
        f"page {passage.page_number} · {badges} · score {evidence.score:.4f}"
    )
    st.caption(
        f"bbox ({passage.x0:.0f}, {passage.y0:.0f}) → "
        f"({passage.x1:.0f}, {passage.y1:.0f})"
    )
    st.markdown(f"> {passage.text.strip()}")
    st.divider()


def render() -> None:
    st.subheader("Ask")
    st.caption(
        "Entities in the question seed a graph traversal; BM25 runs alongside it; "
        "the two rankings are fused, and the answer may use nothing else."
    )

    graph, graph_error = services.graph_store()
    model, model_error = services.language_model()

    if graph_error:
        st.warning(
            f"Neo4j unavailable — answering from lexical search only. {graph_error}"
        )
    if model_error:
        st.error(f"Groq unavailable — answering is disabled. {model_error}")

    hops = st.sidebar.slider("Graph hops", 1, 3, 2)
    limit = st.sidebar.slider("Evidence passages", 3, 20, 8)

    index = services.bm25_index()
    st.sidebar.caption(f"{len(index.passages)} passages indexed")

    question = st.text_input(
        "Question",
        placeholder="Which team owns the service affected by INC-2391?",
    )
    if not st.button("Ask", type="primary", disabled=model is None) or not question:
        return

    from app.graph.memory import InMemoryGraphStore

    active_graph = graph if graph is not None else InMemoryGraphStore()
    retriever = services.build_retriever(active_graph, index, hops=hops)

    started = time.perf_counter()
    result = retriever.retrieve(question, limit=limit)
    retrieved_at = time.perf_counter()
    answer = answer_question(model, result)
    finished = time.perf_counter()

    if answer.grounded:
        st.success(answer.text)
    else:
        st.warning(answer.text)

    if answer.dropped_citations:
        st.error(
            "The model cited evidence that does not exist; those citations were "
            f"dropped: {', '.join(answer.dropped_citations)}"
        )

    columns = st.columns(4)
    columns[0].metric("Evidence", len(result.evidence))
    columns[1].metric("Facts", len(result.facts))
    columns[2].metric("Citations", len(answer.citations))
    columns[3].metric("Total", f"{finished - started:.2f}s")

    tabs = st.tabs(["Citations", "Evidence", "Graph path", "Trace"])

    with tabs[0]:
        if not answer.citations:
            st.info("The answer cited nothing.")
        for citation in answer.citations:
            st.markdown(
                f"**`{citation.global_id}`** · {citation.source_name} · "
                f"page {citation.page_number}"
            )
            st.markdown(f"> {citation.text.strip()}")
            st.divider()

    with tabs[1]:
        if not result.evidence:
            st.info("Retrieval returned nothing.")
        for evidence in result.evidence:
            _evidence_card(evidence)

    with tabs[2]:
        if not result.facts:
            st.info("No graph facts were traversed.")
        else:
            st.dataframe(
                [
                    {
                        "hop": fact.hop,
                        "source": fact.source_name,
                        "relation": fact.relation.value,
                        "target": fact.target_name,
                        "evidence": fact.evidence,
                        "elements": ", ".join(fact.global_element_ids()),
                    }
                    for fact in result.facts
                ],
                hide_index=True,
            )

    with tabs[3]:
        st.json(
            {
                **result.trace,
                "retrieval_seconds": round(retrieved_at - started, 3),
                "generation_seconds": round(finished - retrieved_at, 3),
            }
        )
```

- [ ] **Step 7: Rewrite `app/viewer/main.py` as routing only**

```python
"""Knowledge OS viewer.

    uv run streamlit run app/viewer/main.py

Three modes: inspect the parsed Document IR, ingest a document into the
knowledge graph, and ask questions answered from that graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.viewer.modes import ask, ingest, inspect  # noqa: E402
from app.viewer.theme import CSS  # noqa: E402

MODES = {
    "Inspect": inspect.render,
    "Ingest": ingest.render,
    "Ask": ask.render,
}


def main() -> None:
    st.set_page_config(
        page_title="Knowledge OS",
        page_icon="◎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Knowledge OS")

    mode = st.sidebar.radio("Mode", list(MODES), key="mode")
    st.sidebar.divider()
    MODES[mode]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Extend `tests/viewer/test_app.py`**

Append these tests to the existing file:

```python
def test_every_mode_renders_without_exception():
    for mode in ("Inspect", "Ingest", "Ask"):
        app = AppTest.from_file(APP, default_timeout=120).run()
        app.sidebar.radio[0].set_value(mode).run()
        assert not app.exception, f"{mode} raised {app.exception}"


def test_the_ask_mode_reports_when_nothing_is_ingested():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.radio[0].set_value("Ask").run()
    assert not app.exception
    assert any("passages indexed" in caption.value for caption in app.sidebar.caption)
```

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 10: Launch and click through**

```bash
uv run streamlit run app/viewer/main.py
```

Expected: three modes in the sidebar. In **Ingest**, pick
`payment_system.md`, press *Extract knowledge*, confirm the facts table shows
`Payment API → DEPENDS_ON → Redis` with an evidence quote and an element id,
then *Commit*. In **Ask**, ask "Which team owns the service affected by
INC-2391?" and confirm the answer cites real element ids.

- [ ] **Step 11: Stage the changes (do not commit)**

```bash
git add app/viewer tests/viewer
git status --short
```

---

## Task 11: Neo4j integration tests and documentation

**Files:**
- Create: `tests/graph/test_neo4j_integration.py`
- Modify: `README.md` (append a Knowledge Graph section)

**Interfaces:**
- Consumes: `Neo4jGraphStore` (Task 5); `GraphUpsert` and friends (Task 1).
- Produces: nothing importable — this task adds verification and docs.

These tests touch the real database, so they are skipped unless
`KOS_NEO4J_TESTS=1`. They write under a unique document id and delete exactly
what they created, so they never disturb existing data.

- [ ] **Step 1: Write the integration test**

Create `tests/graph/test_neo4j_integration.py`:

```python
from __future__ import annotations

import os
import uuid

import pytest

from app.knowledge.schema import (
    DocumentRef,
    ElementRef,
    EntityType,
    GraphFact,
    GraphUpsert,
    RelationType,
    ResolvedEntity,
    entity_key,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KOS_NEO4J_TESTS") != "1",
    reason="set KOS_NEO4J_TESTS=1 to run tests against the real Neo4j instance",
)


@pytest.fixture
def store():
    from app.config import settings
    from app.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(
        settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password
    )
    store.verify_connectivity()
    store.ensure_schema()
    yield store
    store.close()


@pytest.fixture
def doc_id() -> str:
    return f"pytest-{uuid.uuid4().hex[:12]}"


@pytest.fixture(autouse=True)
def cleanup(store, doc_id):
    """Delete exactly what this test created, and nothing else."""
    yield
    with store.driver.session() as session:
        session.run(
            "MATCH (e:Element {doc_id: $doc_id}) DETACH DELETE e", doc_id=doc_id
        )
        session.run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=doc_id)
        session.run(
            "MATCH (e:Entity) WHERE e.key STARTS WITH $prefix DETACH DELETE e",
            prefix=f"service|pytest-{doc_id}",
        )
        session.run(
            "MATCH (e:Entity) WHERE e.key STARTS WITH $prefix DETACH DELETE e",
            prefix=f"team|pytest-{doc_id}",
        )
        session.run(
            "MATCH (e:Entity) WHERE e.key STARTS WITH $prefix DETACH DELETE e",
            prefix=f"incident|pytest-{doc_id}",
        )


def _upsert(doc_id: str) -> GraphUpsert:
    service = entity_key(f"pytest-{doc_id} service", EntityType.SERVICE)
    team = entity_key(f"pytest-{doc_id} team", EntityType.TEAM)
    incident = entity_key(f"pytest-{doc_id} incident", EntityType.INCIDENT)

    return GraphUpsert(
        document=DocumentRef(
            id=doc_id, source_name="pytest.md", mime="text/markdown", checksum="c"
        ),
        entities=[
            ResolvedEntity(key=service, name=f"pytest-{doc_id} service", type=EntityType.SERVICE),
            ResolvedEntity(key=team, name=f"pytest-{doc_id} team", type=EntityType.TEAM),
            ResolvedEntity(key=incident, name=f"pytest-{doc_id} incident", type=EntityType.INCIDENT),
        ],
        facts=[
            GraphFact(
                source_key=incident, target_key=service, relation=RelationType.AFFECTS,
                doc_id=doc_id, element_ids=["p1e001"], evidence="the incident hit it",
            ),
            GraphFact(
                source_key=team, target_key=service, relation=RelationType.OWNS,
                doc_id=doc_id, element_ids=["p1e002"], evidence="the team owns it",
            ),
        ],
        elements=[
            ElementRef(
                id="p1e001", doc_id=doc_id, type="paragraph", page_number=1,
                x0=0, y0=0, x1=10, y1=10, order=1, text="the incident hit it",
            ),
            ElementRef(
                id="p1e002", doc_id=doc_id, type="paragraph", page_number=1,
                x0=0, y0=20, x1=10, y1=30, order=2, text="the team owns it",
            ),
        ],
    )


def test_apply_then_traverse_two_hops(store, doc_id):
    upsert = _upsert(doc_id)
    store.apply(upsert)

    incident = upsert.facts[0].source_key
    records = store.traverse([incident], hops=2)

    assert any(record.relation is RelationType.AFFECTS for record in records)
    owns = [record for record in records if record.relation is RelationType.OWNS]
    assert owns, "two-hop traversal did not reach the OWNS edge"
    assert owns[0].hop == 2


def test_applying_twice_does_not_duplicate_entities(store, doc_id):
    upsert = _upsert(doc_id)
    store.apply(upsert)
    before = store.stats()
    store.apply(upsert)
    after = store.stats()

    assert after.entities == before.entities
    assert after.facts == before.facts


def test_elements_round_trip_with_their_geometry(store, doc_id):
    store.apply(_upsert(doc_id))
    found = store.elements([f"{doc_id}:p1e002"])

    assert len(found) == 1
    assert found[0].page_number == 1
    assert found[0].y0 == 20


def test_find_entities_matches_by_name(store, doc_id):
    store.apply(_upsert(doc_id))
    found = store.find_entities([f"pytest-{doc_id} team"])
    assert found and found[0].type is EntityType.TEAM
```

- [ ] **Step 2: Confirm the tests skip by default**

Run: `uv run pytest tests/graph/test_neo4j_integration.py -v`
Expected: 4 skipped.

- [ ] **Step 3: Run them against the real database**

Run: `KOS_NEO4J_TESTS=1 uv run pytest tests/graph/test_neo4j_integration.py -v`
Expected: PASS, 4 tests. Afterwards, confirm nothing was left behind:

```bash
uv run python -c "
from app.config import settings
from app.graph.neo4j import Neo4jGraphStore
store = Neo4jGraphStore(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
with store.driver.session() as session:
    left = session.run(\"MATCH (n) WHERE n.id STARTS WITH 'pytest-' OR n.key CONTAINS 'pytest-' RETURN count(n) AS n\").single()['n']
print('pytest nodes left behind:', left)
store.close()
"
```

Expected: `pytest nodes left behind: 0`

- [ ] **Step 4: Document it in the README**

Append to `README.md`:

````markdown
## Knowledge graph and GraphRAG

Committing a document writes the parsed IR to the document store and its
entities and relationships to Neo4j, where **every fact carries the element it
came from**:

```
(:Document)-[:HAS_ELEMENT]->(:Element)<-[:MENTIONED_IN]-(:Entity)
(:Entity)-[:DEPENDS_ON {doc_id, element_ids, evidence}]->(:Entity)
```

Extraction asks the model for a verbatim evidence quote per fact, then matches
that quote back against the individual elements of the chunk. A matched quote
pins the fact to one element — so a citation resolves to a rectangle on a page,
not merely to a document. Where the quote cannot be matched the fact is kept
and marked `confidence: low` rather than being silently dropped.

Entity resolution is deterministic first: a normalized `type|name` key settles
case, punctuation and whitespace variants for free, and the model is consulted
only when a genuine judgement is needed.

### Retrieval

```
question → entities → seed nodes → k-hop traversal ─┐
                                                     ├─ RRF → evidence → answer
                          BM25 over every element ──┘
```

Graph traversal supplies structure — "which team owns the service affected by
INC-2391" is a path, not a similarity — while BM25 catches exact identifiers
like `INC-2391` and `PAYMENT_502`. Answers may use only retrieved evidence, and
a citation naming an element that was not retrieved is dropped and reported
rather than rendered, so a fabricated citation cannot masquerade as a real one.

### Usage

```bash
uv run streamlit run app/viewer/main.py
```

**Ingest** → pick a document → *Extract knowledge* → review the facts and their
evidence quotes → *Commit*. **Ask** → question → grounded answer with evidence,
the graph path traversed, and a retrieval trace.

### Tests

The suite runs fully offline: every model call goes through a `StructuredLLM`
protocol that tests satisfy with `FakeLLM`, and every graph call through a
`GraphStore` protocol satisfied by `InMemoryGraphStore`.

```bash
uv run pytest                                  # offline, no Neo4j, no Groq
KOS_NEO4J_TESTS=1 uv run pytest tests/graph    # also hit the real database
```
````

- [ ] **Step 5: Run the complete suite one final time**

Run: `uv run pytest -q`
Expected: all tests pass with the Neo4j integration tests skipped.

- [ ] **Step 6: Stage the changes (do not commit)**

```bash
git add tests/graph/test_neo4j_integration.py README.md
git status --short
```

---

## Done

Sub-project 2 is complete when `uv run pytest -q` passes offline, a document
ingested through the UI produces facts whose `element_ids` resolve to real
elements, and "Which team owns the service affected by INC-2391?" is answered
by graph traversal with citations that resolve.

Sub-project 3 then adds embeddings and reranking behind the existing `Retriever`
protocol, and sub-project 5 adds the evaluation harness that measures whether
graph, lexical or hybrid retrieval actually wins on each query class.
