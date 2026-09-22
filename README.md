# Enterprise Multimodal Knowledge Infrastructure

**Enterprise Multimodal Knowledge Infrastructure** is an AI-powered knowledge platform designed to transform an organization's scattered and heterogeneous data into a unified, connected, searchable knowledge layer.

Modern organizations store knowledge across many formats:

- PDFs
    
- Word documents
    
- Markdown
    
- Images
    
- Screenshots
    
- Architecture diagrams
    
- Presentations
    
- Spreadsheets
    
- Tables
    
- Technical documentation
    
- Operational reports
    
- Contracts
    
- Incident reports
    
- Product manuals
    
- Internal knowledge bases
    

Traditional RAG systems generally treat this information as collections of text chunks.

This platform takes a different approach.

It understands **text, images, tables, documents, and relationships between entities** and combines:

- Multimodal processing
    
- Vector search
    
- Keyword search
    
- Knowledge graphs
    
- Hybrid retrieval
    
- Reranking
    
- Multimodal LLMs
    
- Grounded generation
    
- Retrieval evaluation
    

The result is an enterprise knowledge layer that AI applications and agents can query.

The central idea is:
**Don't just index an organization's documents. Build a machine-readable representation of what the organization knows and how that knowledge is connected.**


## Phase 1:
- first, we will setup groq model (groq/openai/gpt-oss-120b) and neo4j
- test the model and neo4j connections
- setup the project structure, environment config and python package

## Phase 2: (Milestone 1 - KnowledgeGraph POC)
- we will prove that we can take, raw text -> LLM -> structured entities + relationships -> Neo4j
- when given any textual information, model should be able to extract the meaningful entities and relationships
- Text -> Knowledge Graph
- we have to successfully create and query nodes + relationships in neo4j

## Phase 3: (Milestone 2 - Reliable Knowledge Graph Ingestion)
- build reusable GraphExtractor and graph ingestion pipdline
- implement entity normalization and entity resolution to prevent duplicate entities
- canonicalize entities and reliably upsert them into Neo4j
---

## Document IR

The first layer of the platform converts source files into a lossless, structured representation — text, hierarchy, tables, images, captions, page locations and reading order — **before** any LLM extraction runs. Nothing downstream re-parses the source file; the IR is the single canonical representation.

```
PDF / Markdown / PNG / XLSX
            ↓
      parse_document()        deterministic: no model calls, no network
            ↓
        Document
          Page
          Element(type, bbox, order, parent_id, text | table | image)
            ↓
     FileDocumentStore        content-addressed JSON + blobs
```

Every element is addressable as `Provenance(doc_id, page_number, element_id, bbox)`, so downstream chunks, extracted facts and graph edges can cite a rectangle on a page rather than a filename.

### What the parsers preserve

| Source | Handling |
|---|---|
| **PDF** | PyMuPDF text spans, fonts and images; pdfplumber table structure. Heading levels from document-wide font statistics, column-aware reading order, running headers/footers/page numbers, figure–caption pairing. Table regions mask the text beneath them so nothing is indexed twice. |
| **Markdown** | Token stream to elements, with synthetic bboxes derived from source line numbers and tagged `synthetic_bbox`. |
| **Images** | One page, one element, whole-image bbox — a screenshot flows through the same pipeline as a PDF page. |
| **XLSX** | One page per sheet; merged title rows lifted out of the header row; formulas recorded alongside their cached values. |

### Guarantees, enforced by tests

- **Lossless** — every non-whitespace character the PDF reports survives into some element.
- **Deterministic** — parsing the same bytes twice produces byte-identical JSON.
- **Referential** — ids unique, `order` strictly increasing, every `parent_id` and `caption_id` resolves, every bbox inside its page.
- **No double-counting** — table text appears in exactly one element.

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
uv run pytest                                   # full suite
uv run python -m tests.fixtures.generate        # rebuild synthetic fixtures
uv run python -m scripts.regenerate_golden      # after intended parser changes
```

Fixtures are generated from code (`tests/fixtures/generate.py`) so ground truth is exact and the files are byte-stable across sessions.

## Viewer

A Streamlit app for seeing exactly what the parser made of a document.

```bash
uv run streamlit run app/viewer/main.py     # http://localhost:8501
```

Pick a bundled fixture or upload a PDF, Markdown file, screenshot or spreadsheet. The page renders with every element boxed and coloured by type, numbered in reading order, and figures joined to their captions by a dashed link. Synthetic bounding boxes — the geometry the parser invents for formats that have none — are drawn with dashed borders so they are never mistaken for measured coordinates.

| Panel | Shows |
|---|---|
| **Page** | Rendered page with overlays; click a row in Elements to highlight one |
| **Inspector** | The selected element's full record, attrs, and its `Provenance` |
| **Elements** | Every element on the page with type, level, parent and bbox |
| **Sections** | The derived heading hierarchy |
| **Tables / Images** | Structured table data; extracted image blobs with their pairings |
| **Warnings** | Every recorded degradation — nothing is dropped silently |
| **JSON** | The canonical `document.json`, downloadable |

The sidebar exposes the `LayoutConfig` thresholds as live controls. Move the heading size ratio, gutter width or caption distance, hit **Apply thresholds**, and watch elements reclassify on the page — which is how you diagnose a real-world PDF that parses badly, rather than editing code and re-running.

Rendering lives in `app/viewer/render.py` as pure functions over the IR, so it is unit-tested independently of Streamlit. `app/ir`, `app/parsing` and `app/storage` never import the viewer.

---

## Knowledge graph and GraphRAG

Committing a document writes the parsed IR to the document store and its entities and relationships to Neo4j, where **every fact carries the element it came from**:

```
(:Document)-[:HAS_ELEMENT]->(:Element)<-[:MENTIONED_IN]-(:Entity)
(:Entity)-[:DEPENDS_ON {doc_id, element_ids, evidence}]->(:Entity)
```

Relationship types are real Cypher types drawn from a closed vocabulary (`OWNS`, `DEPENDS_ON`, `USES`, `AFFECTS`, `DOCUMENTED_BY`, `BELONGS_TO`, `CREATED_BY`, `RESOLVES`, `PART_OF`, `RELATED_TO`), so `MATCH (s)-[:DEPENDS_ON]->(d)` actually traverses. Types are validated against the enum before being interpolated — model output never reaches query construction unchecked.

### Provenance is the point

Extraction asks the model for a **verbatim evidence quote** per fact, then matches that quote back against the individual elements of the chunk. A matched quote pins the fact to one element, so a citation resolves to a rectangle on a page rather than merely to a document. Where the quote cannot be matched, the fact is kept and marked `confidence: low` rather than silently dropped.

Entity resolution is deterministic first: a normalized `type|name` key settles case, punctuation and whitespace variants for free, and the model is consulted only when a genuine judgement is needed. `MERGE` on that key means re-ingesting a document creates no duplicates.

### Retrieval

```
question → entities → seed nodes → k-hop traversal ─┐
                                                     ├─ RRF → evidence → answer
                          BM25 over every element ──┘
```

Graph traversal supplies structure — "which team owns the service affected by INC-2391" is a path, not a similarity — while BM25 catches exact identifiers like `INC-2391` and `PAYMENT_502`, where vector search is weakest. The two rankings fuse with Reciprocal Rank Fusion, which needs no weights to tune.

Answers may use only retrieved evidence. When retrieval returns nothing the app says so rather than letting the model answer from memory, and a citation naming an element that was not retrieved is **dropped and reported** rather than rendered — a fabricated citation that looks real is worse than a missing one.

### Usage

```bash
uv run streamlit run app/viewer/main.py
```

**Ingest** → pick a document → *Extract knowledge* → review each fact with its evidence quote and confidence → *Commit*. **Ask** → question → grounded answer, with tabs for citations, evidence (page and bbox), the graph path traversed, and a retrieval trace.

A **Reset graph** action sits behind a confirmation checkbox in Ingest's danger zone.

### Tests

The suite runs fully offline: every model call goes through a `StructuredLLM` protocol that tests satisfy with `FakeLLM`, and every graph call through a `GraphStore` protocol satisfied by `InMemoryGraphStore`.

```bash
uv run pytest                                  # offline: no Neo4j, no Groq
KOS_NEO4J_TESTS=1 uv run pytest tests/graph    # also hit the real database
```

Integration tests write under a unique document id and delete exactly what they created, by exact key.
