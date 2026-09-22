from __future__ import annotations

from typing import Any, Sequence

from neo4j import GraphDatabase
from pydantic import ValidationError

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
                source_name=element.source_name,
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
    def _entity(record: Any) -> ResolvedEntity | None:
        """Parse a row, or None if it was not written by this schema.

        A graph can hold nodes from other tools — earlier experiments, another
        pipeline — carrying labels we share but properties we do not recognise.
        One such node must not break every read.
        """
        try:
            return ResolvedEntity(
                key=record["key"],
                name=record["name"],
                type=record["type"],
                aliases=list(record["aliases"] or []),
            )
        except ValidationError:
            return None

    def known_entities(self, limit: int = 2000) -> list[ResolvedEntity]:
        with self.driver.session() as session:
            result = session.run(cypher.KNOWN_ENTITIES, limit=limit)
            parsed = (self._entity(record) for record in result)
            return [entity for entity in parsed if entity is not None]

    def find_entities(self, names: Sequence[str], limit: int = 10) -> list[ResolvedEntity]:
        cleaned = [name for name in names if name and name.strip()]
        if not cleaned:
            return []
        with self.driver.session() as session:
            result = session.run(cypher.FIND_ENTITIES, names=cleaned, limit=limit)
            parsed = (self._entity(record) for record in result)
            return [entity for entity in parsed if entity is not None]

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
                    source_name=row["source_name"] or "",
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
