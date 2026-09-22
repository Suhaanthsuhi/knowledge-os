from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

from app.knowledge.schema import ElementRef, EntityType, GraphUpsert, RelationType, ResolvedEntity

__all__ = ["GraphStats", "FactRecord", "GraphStore"]


@dataclass(frozen=True)
class GraphStats:
    documents: int = 0
    entities: int = 0
    facts: int = 0
    elements: int = 0

    def __add__(self, other: GraphStats) -> GraphStats:
        return GraphStats(
            documents=self.documents + other.documents,
            entities=self.entities + other.entities,
            facts=self.facts + other.facts,
            elements=self.elements + other.elements,
        )


class FactRecord(BaseModel):
    """One traversed edge, carrying the provenance needed to cite it."""

    source_key: str
    source_name: str
    source_type: EntityType
    relation: RelationType
    target_key: str
    target_name: str
    target_type: EntityType
    doc_id: str
    element_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    hop: int = 1

    def global_element_ids(self) -> list[str]:
        return [f"{self.doc_id}:{element_id}" for element_id in self.element_ids]

    def sentence(self) -> str:
        return f"{self.source_name} {self.relation.value} {self.target_name}"


@runtime_checkable
class GraphStore(Protocol):
    def ensure_schema(self) -> None: ...
    def apply(self, upsert: GraphUpsert) -> GraphStats: ...
    def known_entities(self) -> list[ResolvedEntity]: ...
    def find_entities(self, names: Sequence[str], limit: int = 10) -> list[ResolvedEntity]: ...
    def traverse(self, keys: Sequence[str], hops: int = 2, limit: int = 200) -> list[FactRecord]: ...
    def elements(self, global_ids: Sequence[str]) -> list[ElementRef]: ...
    def stats(self) -> GraphStats: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
