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


def test_entity_queries_exclude_rows_without_a_key():
    """Foreign :Entity nodes from other tools must not enter a read."""
    assert "e.key IS NOT NULL" in cypher.KNOWN_ENTITIES
    assert "e.key IS NOT NULL" in cypher.FIND_ENTITIES


def test_counts_exclude_nodes_and_edges_this_pipeline_did_not_write():
    assert "e.key IS NOT NULL" in cypher.COUNTS
    assert "el.gid IS NOT NULL" in cypher.COUNTS
    assert "r.doc_id IS NOT NULL" in cypher.COUNTS
