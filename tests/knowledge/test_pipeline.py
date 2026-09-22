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
        id=eid, type=etype, page_number=1,
        bbox=BBox(x0=0, y0=order * 10, x1=100, y1=order * 10 + 8),
        order=order, level=level, text=text,
    )


def _doc() -> Document:
    elements = [
        _element("p1e000", ElementType.HEADING, 0, "Payment System", level=1),
        _element("p1e001", ElementType.PARAGRAPH, 1, "The Payment API depends on Redis."),
        _element("p1e002", ElementType.PARAGRAPH, 2, "The Payments Team owns the Payment API."),
    ]
    assign_parents(elements)
    return Document(
        id="doc1", source_name="payments.md", mime="text/markdown",
        checksum="c" * 64, title="Payment System",
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
            ExtractedFact(source="Payment API", relation=RelationType.DEPENDS_ON,
                          target="Redis", evidence="The Payment API depends on Redis."),
            ExtractedFact(source="Payments Team", relation=RelationType.OWNS,
                          target="Payment API",
                          evidence="The Payments Team owns the Payment API."),
        ],
    )


def _pipeline(llm: FakeLLM) -> KnowledgePipeline:
    return KnowledgePipeline(KnowledgeExtractor(llm), EntityResolver(llm))


def _chunk(text: str) -> Chunk:
    return Chunk(id="doc1:c000", doc_id="doc1", page_number=1, heading="Payment System",
                 text=text, element_ids=("p1e001", "p1e002"))


def _elements() -> dict[str, Element]:
    return {element.id: element for element in _doc().iter_elements()}


def test_a_quote_binds_to_the_element_it_came_from():
    ids, confidence = bind_evidence("The Payment API depends on Redis.", _chunk("…"), _elements())
    assert ids == ["p1e001"]
    assert confidence == "high"


def test_binding_tolerates_whitespace_and_case_differences():
    ids, confidence = bind_evidence("the payment   api DEPENDS on redis", _chunk("…"), _elements())
    assert ids == ["p1e001"]
    assert confidence == "high"


def test_an_unmatched_quote_falls_back_to_the_whole_chunk_at_low_confidence():
    ids, confidence = bind_evidence("a sentence that appears nowhere", _chunk("…"), _elements())
    assert ids == ["p1e001", "p1e002"]
    assert confidence == "low"


def test_an_empty_quote_falls_back_to_the_whole_chunk():
    ids, confidence = bind_evidence("   ", _chunk("…"), _elements())
    assert ids == ["p1e001", "p1e002"]
    assert confidence == "low"


def test_the_pipeline_produces_facts_with_element_level_provenance():
    upsert = _pipeline(FakeLLM(responses=[_extraction()])).run(_doc())

    depends = next(f for f in upsert.facts if f.relation is RelationType.DEPENDS_ON)
    assert depends.element_ids == ["p1e001"]
    assert depends.doc_id == "doc1"
    assert depends.confidence == "high"


def test_every_fact_resolves_to_a_known_entity_key():
    upsert = _pipeline(FakeLLM(responses=[_extraction()])).run(_doc())

    keys = {entity.key for entity in upsert.entities}
    for fact in upsert.facts:
        assert fact.source_key in keys
        assert fact.target_key in keys


def test_only_evidence_elements_are_written():
    upsert = _pipeline(FakeLLM(responses=[_extraction()])).run(_doc())
    assert {element.id for element in upsert.elements} == {"p1e001", "p1e002"}


def test_the_document_reference_carries_identity():
    upsert = _pipeline(FakeLLM(responses=[_extraction()])).run(_doc())

    assert upsert.document.id == "doc1"
    assert upsert.document.title == "Payment System"
    assert upsert.document.mime == "text/markdown"


def test_a_fact_naming_an_unextracted_entity_is_dropped_with_a_warning():
    extraction = ChunkExtraction(
        entities=[ExtractedEntity(name="Payment API", type=EntityType.SERVICE)],
        facts=[ExtractedFact(source="Payment API", relation=RelationType.USES,
                             target="Kafka", evidence="The Payment API depends on Redis.")],
    )
    upsert = _pipeline(FakeLLM(responses=[extraction])).run(_doc())

    assert upsert.facts == []
    assert upsert.warnings[0]["code"] == "unknown_entity"
    assert "Kafka" in upsert.warnings[0]["detail"]


def test_a_failing_chunk_is_recorded_and_the_rest_still_commits():
    elements = [
        _element("p1e000", ElementType.HEADING, 0, "Payment System", level=1),
        _element("p1e001", ElementType.PARAGRAPH, 1, "x" * 1600),
        _element("p1e002", ElementType.PARAGRAPH, 2, "The Payment API depends on Redis."),
    ]
    assign_parents(elements)
    doc = Document(id="doc2", source_name="p.md", mime="text/markdown", checksum="c",
                   pages=[Page(number=1, width=612, height=792, elements=elements)])

    llm = FakeLLM(responses=[
        RuntimeError("groq timeout"),
        ChunkExtraction(
            entities=[
                ExtractedEntity(name="Payment API", type=EntityType.SERVICE),
                ExtractedEntity(name="Redis", type=EntityType.DATABASE),
            ],
            facts=[ExtractedFact(source="Payment API", relation=RelationType.DEPENDS_ON,
                                 target="Redis",
                                 evidence="The Payment API depends on Redis.")],
        ),
    ])
    upsert = KnowledgePipeline(KnowledgeExtractor(llm), EntityResolver(None)).run(doc)

    assert any(w["code"] == "chunk_failed" for w in upsert.warnings)
    assert len(upsert.facts) == 1


def test_known_entities_are_reused_rather_than_duplicated():
    from app.knowledge.schema import ResolvedEntity

    known = [ResolvedEntity(key=entity_key("Payment API", EntityType.SERVICE),
                            name="Payment API", type=EntityType.SERVICE)]
    upsert = _pipeline(FakeLLM(responses=[_extraction()])).run(_doc(), known)

    assert len([e for e in upsert.entities if e.name == "Payment API"]) == 1


def test_progress_is_reported_per_chunk():
    seen: list[tuple[int, int]] = []
    _pipeline(FakeLLM(responses=[_extraction()])).run(
        _doc(), on_progress=lambda done, total: seen.append((done, total))
    )
    assert seen == [(1, 1)]


def test_a_document_with_no_content_produces_an_empty_upsert():
    doc = Document(id="empty", source_name="e.md", mime="text/markdown", checksum="c")
    upsert = _pipeline(FakeLLM(responses=[])).run(doc)

    assert upsert.entities == []
    assert upsert.facts == []
