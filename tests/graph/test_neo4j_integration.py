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
    """Delete exactly what this test created, and nothing else.

    Deletion is by exact key, not by pattern: `entity_key` strips punctuation,
    so a key derived from `pytest-abc123` reads `pytest abc123` and a CONTAINS
    match on the hyphenated id silently matches nothing.
    """
    yield
    with store.driver.session() as session:
        session.run("MATCH (e:Element {doc_id: $doc_id}) DETACH DELETE e", doc_id=doc_id)
        session.run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=doc_id)
        session.run(
            "MATCH (e:Entity) WHERE e.key IN $keys DETACH DELETE e",
            keys=list(_keys(doc_id)),
        )


def _keys(doc_id: str) -> tuple[str, str, str]:
    return (
        entity_key(f"pytest-{doc_id} service", EntityType.SERVICE),
        entity_key(f"pytest-{doc_id} team", EntityType.TEAM),
        entity_key(f"pytest-{doc_id} incident", EntityType.INCIDENT),
    )


def _upsert(doc_id: str) -> GraphUpsert:
    service, team, incident = _keys(doc_id)
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
            GraphFact(source_key=incident, target_key=service, relation=RelationType.AFFECTS,
                      doc_id=doc_id, element_ids=["p1e001"], evidence="the incident hit it"),
            GraphFact(source_key=team, target_key=service, relation=RelationType.OWNS,
                      doc_id=doc_id, element_ids=["p1e002"], evidence="the team owns it"),
        ],
        elements=[
            ElementRef(id="p1e001", doc_id=doc_id, type="paragraph", page_number=1,
                       x0=0, y0=0, x1=10, y1=10, order=1, text="the incident hit it"),
            ElementRef(id="p1e002", doc_id=doc_id, type="paragraph", page_number=1,
                       x0=0, y0=20, x1=10, y1=30, order=2, text="the team owns it"),
        ],
    )


def test_apply_then_traverse_two_hops(store, doc_id):
    store.apply(_upsert(doc_id))
    _service, _team, incident = _keys(doc_id)

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
