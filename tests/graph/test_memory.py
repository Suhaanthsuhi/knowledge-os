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
        id=eid, doc_id=doc_id, type="paragraph", page_number=1,
        x0=0, y0=0, x1=10, y1=10, order=0, text=text,
    )


def _upsert(doc_id: str = "d") -> GraphUpsert:
    return GraphUpsert(
        document=DocumentRef(id=doc_id, source_name="s.md", mime="text/markdown", checksum="c"),
        entities=[
            ResolvedEntity(key=PAYMENT, name="Payment API", type=EntityType.SERVICE),
            ResolvedEntity(key=REDIS, name="Redis", type=EntityType.DATABASE),
            ResolvedEntity(key=TEAM, name="Payments Team", type=EntityType.TEAM),
            ResolvedEntity(key=INCIDENT, name="INC-2391", type=EntityType.INCIDENT),
        ],
        facts=[
            GraphFact(source_key=PAYMENT, target_key=REDIS, relation=RelationType.DEPENDS_ON,
                      doc_id=doc_id, element_ids=["p1e001"], evidence="depends on Redis"),
            GraphFact(source_key=TEAM, target_key=PAYMENT, relation=RelationType.OWNS,
                      doc_id=doc_id, element_ids=["p1e002"], evidence="owns the Payment API"),
            GraphFact(source_key=INCIDENT, target_key=PAYMENT, relation=RelationType.AFFECTS,
                      doc_id=doc_id, element_ids=["p1e003"], evidence="INC-2391 affected"),
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
        GraphFact(source_key=PAYMENT, target_key=REDIS, relation=RelationType.DEPENDS_ON,
                  doc_id="d", element_ids=["p2e009"], evidence="also on page 2")
    ]
    store.apply(second)

    records = store.traverse([PAYMENT], hops=1)
    depends = next(r for r in records if r.relation is RelationType.DEPENDS_ON)
    assert sorted(depends.element_ids) == ["p1e001", "p2e009"]


def test_find_entities_matches_on_name_case_insensitively():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert [entity.key for entity in store.find_entities(["payment api"])] == [PAYMENT]


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
    assert [e.id for e in store.elements(["d:p1e001", "d:missing"])] == ["p1e001"]


def test_known_entities_returns_everything_for_the_resolver():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    assert len(store.known_entities()) == 4


def test_reset_empties_the_store():
    store = InMemoryGraphStore()
    store.apply(_upsert())
    store.reset()

    totals = store.stats()
    assert (totals.documents, totals.entities, totals.facts, totals.elements) == (0, 0, 0, 0)
