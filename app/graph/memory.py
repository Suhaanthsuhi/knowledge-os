from __future__ import annotations

from typing import Sequence

from app.graph.store import FactRecord, GraphStats
from app.knowledge.schema import (
    DocumentRef,
    ElementRef,
    GraphFact,
    GraphUpsert,
    ResolvedEntity,
    normalize_name,
)

__all__ = ["InMemoryGraphStore"]


def _fact_key(fact: GraphFact) -> tuple[str, str, str, str]:
    return (fact.source_key, fact.relation.value, fact.target_key, fact.doc_id)


class InMemoryGraphStore:
    """A dict-backed GraphStore.

    Exists so the pipeline, retrieval and answering can be tested without a
    database, and so traversal logic is verified independently of Cypher.
    """

    def __init__(self) -> None:
        self.documents: dict[str, DocumentRef] = {}
        self.entities: dict[str, ResolvedEntity] = {}
        self.elements_by_gid: dict[str, ElementRef] = {}
        self.facts: dict[tuple[str, str, str, str], GraphFact] = {}

    def ensure_schema(self) -> None:
        return None

    def apply(self, upsert: GraphUpsert) -> GraphStats:
        self.documents[upsert.document.id] = upsert.document

        for entity in upsert.entities:
            existing = self.entities.get(entity.key)
            self.entities[entity.key] = (
                existing.with_aliases([*entity.aliases, entity.name])
                if existing
                else entity
            )

        for element in upsert.elements:
            self.elements_by_gid[element.global_id] = element

        for fact in upsert.facts:
            key = _fact_key(fact)
            existing_fact = self.facts.get(key)
            if existing_fact is None:
                self.facts[key] = fact.model_copy(deep=True)
                continue
            merged = list(dict.fromkeys([*existing_fact.element_ids, *fact.element_ids]))
            self.facts[key] = existing_fact.model_copy(update={"element_ids": merged})

        return GraphStats(
            documents=1,
            entities=len(upsert.entities),
            facts=len(upsert.facts),
            elements=len(upsert.elements),
        )

    def known_entities(self) -> list[ResolvedEntity]:
        return list(self.entities.values())

    def find_entities(self, names: Sequence[str], limit: int = 10) -> list[ResolvedEntity]:
        found: dict[str, ResolvedEntity] = {}
        for raw in names:
            needle = normalize_name(raw)
            if not needle:
                continue
            for entity in self.entities.values():
                name = normalize_name(entity.name)
                if needle == name or needle in entity.aliases or needle in name:
                    found[entity.key] = entity
        return list(found.values())[:limit]

    def _record(self, fact: GraphFact, hop: int) -> FactRecord:
        source = self.entities[fact.source_key]
        target = self.entities[fact.target_key]
        return FactRecord(
            source_key=source.key,
            source_name=source.name,
            source_type=source.type,
            relation=fact.relation,
            target_key=target.key,
            target_name=target.name,
            target_type=target.type,
            doc_id=fact.doc_id,
            element_ids=list(fact.element_ids),
            evidence=fact.evidence,
            hop=hop,
        )

    def traverse(
        self, keys: Sequence[str], hops: int = 2, limit: int = 200
    ) -> list[FactRecord]:
        """Breadth-first expansion, edges undirected, each edge reported once."""
        frontier = {key for key in keys if key in self.entities}
        seen_edges: set[tuple[str, str, str, str]] = set()
        visited = set(frontier)
        records: list[FactRecord] = []

        for hop in range(1, max(0, hops) + 1):
            if not frontier or len(records) >= limit:
                break
            next_frontier: set[str] = set()

            for key, fact in self.facts.items():
                if key in seen_edges:
                    continue
                if fact.source_key in frontier:
                    other = fact.target_key
                elif fact.target_key in frontier:
                    other = fact.source_key
                else:
                    continue

                seen_edges.add(key)
                records.append(self._record(fact, hop))
                if other not in visited:
                    next_frontier.add(other)
                    visited.add(other)
                if len(records) >= limit:
                    break

            frontier = next_frontier

        return records[:limit]

    def elements(self, global_ids: Sequence[str]) -> list[ElementRef]:
        return [
            self.elements_by_gid[gid] for gid in global_ids if gid in self.elements_by_gid
        ]

    def stats(self) -> GraphStats:
        return GraphStats(
            documents=len(self.documents),
            entities=len(self.entities),
            facts=len(self.facts),
            elements=len(self.elements_by_gid),
        )

    def reset(self) -> None:
        self.documents.clear()
        self.entities.clear()
        self.elements_by_gid.clear()
        self.facts.clear()

    def close(self) -> None:
        return None
