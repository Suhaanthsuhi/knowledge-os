# Evidence-Grounded Knowledge Graph + GraphRAG — Design Spec

Date: 2026-09-22
Status: Approved
Sub-project: 2 of 6, plus a first cut of retrieval and answering

## Context

Sub-project 1 produced a lossless Document IR: every element carries a type,
bounding box, reading order, parent heading and a stable id, addressable as
`Provenance(doc_id, page_number, element_id, bbox)`.

This sub-project turns that IR into a knowledge graph whose every fact is
traceable back to a rectangle on a page, and makes GraphRAG work end to end
from the UI: upload, parse, extract, commit, ask, cite.

### Verified environment

- Neo4j Aura reachable at `neo4j+s://83856c6d.databases.neo4j.io`, currently
  holding 13 legacy nodes from notebook experiments under mixed labels
  (`Company`, `Person`, `Location`, `Date`, `Entity`).
- Groq reachable; `openai/gpt-oss-120b` returns structured extraction in ~1.5s.
- Groq offers no embeddings API.

### What is being replaced

`app/extraction/schema.py`, `extractor.py` and `resolver.py` carry no
provenance and model relationships as a flat `source/relationship/target`
triple. `app/graph/neo4j.py` writes every node as `:Entity` joined by a single
`RELATES_TO` type with the real type demoted to a property, which makes
multi-hop Cypher traversal impossible. Both are rewritten here. The playground
notebook depends on the old names and will break; it is a scratchpad, and that
is accepted.

Legacy graph data is **not** deleted by this work. A Reset action is exposed in
the UI so the destructive step stays a deliberate human click.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Retrieval | Graph traversal + BM25, fused with RRF | No embeddings provider needed; exact identifiers (`INC-2391`) are where lexical beats vectors; a `Retriever` protocol leaves the vector slot open for sub-project 3 |
| Relationship types | Closed vocabulary, real Cypher types | `MATCH (s)-[:DEPENDS_ON*1..3]->(d)` requires real types; an allowlist keeps model output out of query construction |
| Elements in Neo4j | Only those that produced a fact | The full IR lives in `FileDocumentStore`; the graph holds evidence anchors and stays lean |
| Provenance granularity | Per-element, via verbatim evidence quotes | A chunk-level citation is not traceable to a rectangle; matching the quote back to an element is |
| Entity resolution | Deterministic key first, LLM only for ambiguity | Today's one-LLM-call-per-entity is slow and non-deterministic for cases a normalization key settles for free |
| Graph access | `GraphStore` protocol, Neo4j + in-memory | Makes the pipeline testable offline; Neo4j tests become opt-in |
| Commit model | Preview extraction, then commit | A wrong extraction should be a rejected preview, not graph pollution |

## Graph schema

```
(:Document {id, source_name, mime, title, checksum})
    -[:HAS_ELEMENT]->
(:Element {id, doc_id, type, page, x0, y0, x1, y1, order, text})
    <-[:MENTIONED_IN]-
(:Entity {key, name, type, aliases})
    -[:<RELATION> {doc_id, element_ids, evidence, confidence}]->
(:Entity)
```

`Entity.key` is a normalized `name|type` slug, making `MERGE` idempotent:
re-ingesting a document does not duplicate entities.

Constraints and indexes created on connect:

```cypher
CREATE CONSTRAINT entity_key IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.key IS UNIQUE;
CREATE CONSTRAINT element_id IF NOT EXISTS
  FOR (e:Element) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT document_id IF NOT EXISTS
  FOR (d:Document) REQUIRE d.id IS UNIQUE;
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
```

`Element.id` is globally unique as `"{doc_id}:{element_id}"`, since element ids
are only document-scoped in the IR.

### Controlled vocabularies

```python
class EntityType(StrEnum):
    PERSON, TEAM, SERVICE, PRODUCT, DATABASE, TECHNOLOGY,
    ORGANIZATION, INCIDENT, DOCUMENT, POLICY, LOCATION, CONCEPT, OTHER

class RelationType(StrEnum):
    OWNS, DEPENDS_ON, USES, AFFECTS, DOCUMENTED_BY, BELONGS_TO,
    CREATED_BY, RESOLVES, PART_OF, RELATED_TO
```

A relation type is validated against the enum before being interpolated into
Cypher. Model output never reaches query construction unvalidated; an
unrecognised type degrades to `RELATED_TO` with a warning.

## Extraction pipeline

```
Document → chunker → extractor → resolver → GraphUpsert → GraphStore
```

### Commit writes to both stores

Committing persists the parsed IR to `FileDocumentStore` **and** writes the
graph. Both are required: the graph holds facts and evidence anchors, while the
store holds the full element text that BM25 indexes and that answers quote. A
commit that wrote only the graph would leave Ask unable to retrieve any text.

The store write happens first. If the graph write then fails, the document is
still inspectable and lexically searchable, and the commit can be retried —
whereas graph edges pointing at elements no store can resolve would be dead
citations.

**Chunker** (`app/knowledge/chunker.py`) — pure. Groups elements by section
using the existing `section_tree`, splitting at `MAX_CHUNK_CHARS = 1500` and
never splitting mid-element. Each `Chunk` carries `element_ids`, the joined
`text`, and the section heading as context.

**Extractor** (`app/knowledge/extractor.py`) — one Groq structured call per
chunk, returning entities and facts. Each fact carries a short **verbatim
evidence quote** drawn from the chunk text.

**Evidence binding** (`app/knowledge/pipeline.py`) — the quote is matched, after
whitespace normalization, against each element's text in the chunk. A match
binds the fact to that specific element id. No match keeps the fact bound to
the whole chunk, flagged `confidence="low"`.

**Resolver** (`app/knowledge/resolver.py`) — two stages:

1. Normalization key: lowercase, strip punctuation, collapse whitespace,
   combine with the entity type. Exact key match reuses the existing entity.
2. LLM adjudication, only for an entity that fuzzy-matches existing candidates
   (case-insensitive substring, or shared alias) without a key match.

Entities matching neither path are created new.

## Retrieval and answering

```
question
  ↓ entity extraction (one structured call)
seed entities → exact key match, then fuzzy name match
  ↓ k-hop traversal (default 2, capped at 3)
facts + element_ids ────┐
  ↓ fetch elements      │  BM25 over every element of every ingested document
evidence candidates ────┴──→ Reciprocal Rank Fusion (k=60)
  ↓ top-N (default 8)
context = fact table + evidence snippets, each with doc / page / bbox
  ↓
Groq, instructed to cite element ids
  ↓
citations resolved against real element ids; unresolvable ones dropped
```

- `app/rag/bm25.py` — pure-Python Okapi BM25 (`k1=1.5`, `b=0.75`) over IR
  element text. The corpus is every element of every document in
  `FileDocumentStore`, built in memory and cached against the store's document
  id list, so it rebuilds when a commit adds a document. This is adequate for
  tens of documents; real indexing belongs to sub-project 6.
- `app/rag/retriever.py` — `Retriever` protocol; `GraphRetriever`,
  `Bm25Retriever`, `HybridRetriever`. The protocol is the slot vectors occupy
  in sub-project 3.
- `app/rag/answer.py` — context assembly, the grounded prompt, citation parsing.

**Grounding is enforced, not requested.** The prompt receives only retrieved
evidence. When retrieval is empty the app states that no evidence was found
rather than letting the model answer from memory. A citation naming an element
id that does not exist is dropped and reported, so a fabricated citation cannot
render as a real one.

## UI

The viewer gains a mode selector in the sidebar.

| Mode | Behaviour |
|---|---|
| **Inspect** | The existing IR viewer, unchanged |
| **Ingest** | Parse → preview entities and facts with their evidence quotes → **Commit** → summary of nodes and edges written. A separate **Reset graph** action sits behind a confirmation checkbox. |
| **Ask** | Question → grounded answer with inline citations, plus tabs for Evidence (cards with page/bbox and snippet), Graph path (facts traversed), and Trace (seed entities, hop count, timings) |

## Failure behaviour

- Extraction is per-chunk and isolated: one failing chunk is recorded as a
  warning and the remaining chunks still commit.
- Neo4j unreachable: a banner is shown, and Ask degrades to BM25-only rather
  than crashing.
- A question yielding no entities falls back to pure lexical retrieval.
- A Groq error during extraction surfaces the message and writes nothing
  partial to the graph.
- Every degradation is recorded and visible, never silent.

## Module layout

```
app/knowledge/
  schema.py      EntityType, RelationType, ExtractedEntity, ExtractedFact,
                 ChunkExtraction, ResolvedEntity, GraphFact, GraphUpsert
  chunker.py     Document -> list[Chunk]                     (pure)
  extractor.py   Chunk -> ChunkExtraction                    (LLM)
  resolver.py    entity -> ResolvedEntity                    (key, then LLM)
  pipeline.py    Document -> GraphUpsert                     (orchestration)
  llm.py         LLM protocol + GroqLLM + FakeLLM
app/graph/
  store.py       GraphStore protocol, GraphStats
  memory.py      InMemoryGraphStore
  neo4j.py       Neo4jGraphStore (rewritten)
  cypher.py      statement builders + relation-type allowlist
app/rag/
  bm25.py        pure-Python Okapi BM25
  retriever.py   Retriever protocol + graph / bm25 / hybrid
  answer.py      context assembly, grounded prompt, citation parsing
app/viewer/
  modes/inspect.py, modes/ingest.py, modes/ask.py
```

## Testing

Pure and offline by default:

- chunker: element-id tracking, size cap, section grouping, no text lost
- resolver: normalization keys, alias matching, LLM invoked only on ambiguity
- bm25: scoring sanity, exact identifiers outrank generic prose
- fusion: RRF ordering
- cypher: the relation-type allowlist rejects anything outside the enum
- evidence binding: a quote binds to the right element; a non-matching quote
  degrades to low confidence rather than being dropped
- pipeline: `FakeLLM` drives a deterministic document-to-graph run
- traversal: `InMemoryGraphStore` answers a two-hop question correctly
- answer: citation parsing, unresolvable citations dropped, empty-evidence path
- AppTest smoke over all three UI modes

Neo4j integration tests are opt-in via `KOS_NEO4J_TESTS=1`, write under a
dedicated tenant property and delete what they create.

## Success criteria

1. Uploading a document in the UI parses it, previews extracted entities and
   facts with evidence, and on request commits the IR to `FileDocumentStore`
   and the entities and facts to Neo4j.
2. Every committed fact carries `doc_id` and at least one `element_id`.
3. "Which team owns the service affected by INC-2391?" is answered by graph
   traversal, with citations resolving to real elements.
4. Re-ingesting the same document creates no duplicate entities.
5. The full suite passes offline, with no Neo4j and no Groq.
