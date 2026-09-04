from langchain_groq import ChatGroq

from app.config import settings
from app.extraction.schema import Entity, EntityResolution


class EntityResolver:

    def __init__(self):
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=settings.groq_api_key,
        )

        self.structured_llm = self.llm.with_structured_output(
            EntityResolution
        )

    def resolve(
        self,
        entity: Entity,
        existing_entities: list[Entity],
    ) -> Entity:

        if not existing_entities:
            return entity

        candidates = "\n".join(
            f"- {e.name} ({e.type})"
            for e in existing_entities
        )

        prompt = f"""
        You are an entity resolution system.

        Determine whether the new entity refers to
        one of the existing entities.

        New entity:
        Name: {entity.name}
        Type: {entity.type}

        Existing entities:
        {candidates}

        Rules:

        1. Return match=true if the new entity clearly
           refers to an existing entity.

        2. If there is a match, matched_entity MUST be
           the exact name of the existing entity.

        3. If there is no match, return match=false and
           matched_entity=null.

        4. Consider aliases, spelling variations,
           abbreviations and common naming variations.

        5. Do not merge entities merely because their
           names are similar.

        6. Do not invent an entity.
        """

        result = self.structured_llm.invoke(prompt)

        if result.match:
            for existing in existing_entities:
                if existing.name == result.matched_entity:
                    return existing

        return entity


__all__ = ["EntityResolver"]