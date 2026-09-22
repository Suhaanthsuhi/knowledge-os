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
