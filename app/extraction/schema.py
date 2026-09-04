from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Entity schema"""

    name: str = Field(..., description="The name of the entity")
    type: str = Field(..., description="The type of the entity, example: Person, Company, Location, etc.")


class Relationship(BaseModel):
    """Relationship schema"""

    source: str = Field(..., description="The name of the source entity")
    relationship: str = Field(..., description="The type of the relationship")
    target: str = Field(..., description="The name of the target entity")


class KnowledgeGraph(BaseModel):
    """Knowledge graph schema"""

    entities: list[Entity] = Field(..., description="List of entities in the knowledge graph")
    relationships: list[Relationship] = Field(..., description="List of relationships in the knowledge graph")


__all__ = ["Entity", "Relationship", "KnowledgeGraph"]