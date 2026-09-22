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
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE CONSTRAINT element_gid IF NOT EXISTS FOR (e:Element) REQUIRE e.gid IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
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
    e.aliases = [alias IN coalesce(e.aliases, []) + $aliases | alias][0..50]
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
WHERE e.key IS NOT NULL
  AND ( toLower(e.name) = toLower(name)
     OR toLower(e.name) CONTAINS toLower(name)
     OR any(alias IN coalesce(e.aliases, []) WHERE alias = toLower(name)) )
RETURN DISTINCT e.key AS key, e.name AS name, e.type AS type,
       coalesce(e.aliases, []) AS aliases
LIMIT $limit
"""

KNOWN_ENTITIES = """
MATCH (e:Entity)
WHERE e.key IS NOT NULL
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

# Counts only what this pipeline wrote. A graph may hold nodes and edges from
# other tools that share our labels; reporting those would overstate what the
# retrieval path can actually use.
COUNTS = """
MATCH (d:Document)
WITH count(d) AS documents
OPTIONAL MATCH (e:Entity) WHERE e.key IS NOT NULL
WITH documents, count(e) AS entities
OPTIONAL MATCH (el:Element) WHERE el.gid IS NOT NULL
WITH documents, entities, count(el) AS elements
OPTIONAL MATCH ()-[r]->() WHERE r.doc_id IS NOT NULL
RETURN documents, entities, elements, count(r) AS facts
"""

RESET = "MATCH (n) WHERE n:Document OR n:Entity OR n:Element DETACH DELETE n"
