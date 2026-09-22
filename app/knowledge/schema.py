from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "EntityType",
    "RelationType",
    "normalize_name",
    "entity_key",
    "ExtractedEntity",
    "ExtractedFact",
    "ChunkExtraction",
    "EntityMatch",
    "QuestionEntities",
    "Chunk",
    "ResolvedEntity",
    "ElementRef",
    "DocumentRef",
    "GraphFact",
    "GraphUpsert",
]

_PUNCTUATION = re.compile(r"[^a-z0-9\s]+")
_WHITESPACE = re.compile(r"\s+")


class EntityType(StrEnum):
    PERSON = "Person"
    TEAM = "Team"
    SERVICE = "Service"
    PRODUCT = "Product"
    DATABASE = "Database"
    TECHNOLOGY = "Technology"
    ORGANIZATION = "Organization"
    INCIDENT = "Incident"
    DOCUMENT = "Document"
    POLICY = "Policy"
    LOCATION = "Location"
    CONCEPT = "Concept"
    OTHER = "Other"


class RelationType(StrEnum):
    OWNS = "OWNS"
    DEPENDS_ON = "DEPENDS_ON"
    USES = "USES"
    AFFECTS = "AFFECTS"
    DOCUMENTED_BY = "DOCUMENTED_BY"
    BELONGS_TO = "BELONGS_TO"
    CREATED_BY = "CREATED_BY"
    RESOLVES = "RESOLVES"
    PART_OF = "PART_OF"
    RELATED_TO = "RELATED_TO"


def normalize_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    lowered = name.strip().lower()
    stripped = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()


def entity_key(name: str, entity_type: EntityType) -> str:
    """Stable identity for an entity. Makes MERGE idempotent across ingests."""
    normalized = normalize_name(name)
    if not normalized:
        raise ValueError(f"empty entity name: {name!r}")
    return f"{entity_type.value.lower()}|{normalized}"


class ExtractedEntity(BaseModel):
    name: str = Field(description="The entity's name exactly as written in the text")
    type: EntityType = Field(description="The entity's category")
    aliases: list[str] = Field(default_factory=list)


class ExtractedFact(BaseModel):
    source: str = Field(description="Name of the source entity")
    relation: RelationType = Field(description="How source relates to target")
    target: str = Field(description="Name of the target entity")
    evidence: str = Field(
        description="A short quote copied verbatim from the text that states this fact"
    )


class ChunkExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)


class EntityMatch(BaseModel):
    match: bool = Field(description="Whether the new entity is the same as a candidate")
    matched_key: str | None = Field(
        default=None, description="The exact key of the matching candidate, else null"
    )


class QuestionEntities(BaseModel):
    names: list[str] = Field(
        default_factory=list, description="Entity names mentioned in the question"
    )


@dataclass(frozen=True)
class Chunk:
    """A unit of text handed to the extractor, with the elements it came from."""

    id: str
    doc_id: str
    page_number: int
    heading: str | None
    text: str
    element_ids: tuple[str, ...]


class ResolvedEntity(BaseModel):
    key: str
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)

    def with_aliases(self, extra: list[str]) -> ResolvedEntity:
        """Merge aliases, dropping any that restate the canonical name."""
        canonical = normalize_name(self.name)
        merged = {normalize_name(alias) for alias in [*self.aliases, *extra]}
        merged.discard(canonical)
        merged.discard("")
        return self.model_copy(update={"aliases": sorted(merged)})


class ElementRef(BaseModel):
    """The evidence anchor written to the graph."""

    id: str
    doc_id: str
    type: str
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    order: int
    text: str

    @property
    def global_id(self) -> str:
        return f"{self.doc_id}:{self.id}"


class DocumentRef(BaseModel):
    id: str
    source_name: str
    mime: str
    title: str | None = None
    checksum: str = ""


class GraphFact(BaseModel):
    source_key: str
    target_key: str
    relation: RelationType
    doc_id: str
    element_ids: list[str] = Field(default_factory=list)
    evidence: str = ""
    confidence: Literal["high", "low"] = "high"


class GraphUpsert(BaseModel):
    document: DocumentRef
    entities: list[ResolvedEntity] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    elements: list[ElementRef] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)

    def add_warning(self, code: str, detail: str) -> None:
        self.warnings.append({"code": code, "detail": detail})
