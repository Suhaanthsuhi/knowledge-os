from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(
        ...,
        description="The name of the entity"
    )

    type: str = Field(
        ...,
        description="The type of entity, e.g. Person, Company, Location"
    )


class Relationship(BaseModel):
    source: str = Field(
        ...,
        description="The name of the source entity"
    )

    relationship: str = Field(
        ...,
        description="The relationship between the entities"
    )

    target: str = Field(
        ...,
        description="The name of the target entity"
    )


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


class EntityResolution(BaseModel):
    match: bool = Field(
        ...,
        description="Whether the new entity matches an existing entity"
    )

    matched_entity: str | None = Field(
        ...,
        description="The exact name of the matching existing entity, or null if no match"
    )


__all__ = [
    "Entity",
    "Relationship",
    "KnowledgeGraph",
    "EntityResolution",
]