from __future__ import annotations

from app.graph.neo4j import Neo4jGraphStore
from app.knowledge.schema import EntityType


def test_a_conforming_row_parses():
    row = {
        "key": "service|payment api",
        "name": "Payment API",
        "type": "Service",
        "aliases": ["payments api"],
    }
    entity = Neo4jGraphStore._entity(row)

    assert entity is not None
    assert entity.type is EntityType.SERVICE
    assert entity.aliases == ["payments api"]


def test_a_row_with_no_key_is_skipped():
    row = {"key": None, "name": "Apple", "type": "Company", "aliases": None}
    assert Neo4jGraphStore._entity(row) is None


def test_a_row_with_an_unknown_type_is_skipped():
    """Legacy nodes may carry labels we share but a vocabulary we do not."""
    row = {"key": "company|apple", "name": "Apple", "type": "Company", "aliases": []}
    assert Neo4jGraphStore._entity(row) is None


def test_null_aliases_become_an_empty_list():
    row = {"key": "team|payments", "name": "Payments", "type": "Team", "aliases": None}
    entity = Neo4jGraphStore._entity(row)

    assert entity is not None
    assert entity.aliases == []
