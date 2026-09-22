from __future__ import annotations

from app.knowledge.llm import StructuredLLM
from app.knowledge.schema import (
    EntityMatch,
    ExtractedEntity,
    ResolvedEntity,
    entity_key,
    normalize_name,
)

__all__ = ["EntityResolver", "RESOLUTION_PROMPT"]

RESOLUTION_PROMPT = """\
Decide whether a newly extracted entity refers to one that already exists.

New entity:
  name: {name}
  type: {type}

Existing candidates:
{candidates}

Rules:
1. Return match=true only if the new entity clearly denotes the same thing.
2. When matching, matched_key must be copied exactly from a candidate's key.
3. Similar names are not enough. "Payment API" and "Payment API v2" are
   different things; "Payment API" and "payments api" are the same thing.
4. Never invent a key.
"""


class EntityResolver:
    """Deterministic first, model second.

    Case, punctuation and whitespace variants are settled by the normalization
    key at no cost. The model is consulted only where a genuine judgement is
    needed, which keeps resolution mostly deterministic and cheap.
    """

    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self.llm = llm

    def resolve(
        self, candidate: ExtractedEntity, known: dict[str, ResolvedEntity]
    ) -> ResolvedEntity:
        key = entity_key(candidate.name, candidate.type)

        existing = known.get(key)
        if existing is not None:
            return existing.with_aliases([*candidate.aliases, candidate.name])

        normalized = normalize_name(candidate.name)
        for entity in known.values():
            if entity.type is not candidate.type:
                continue
            if normalized in entity.aliases:
                return entity.with_aliases([*candidate.aliases, candidate.name])

        fuzzy = self._fuzzy_candidates(normalized, candidate, known)
        if fuzzy and self.llm is not None:
            matched = self._adjudicate(candidate, fuzzy)
            if matched is not None:
                return matched.with_aliases([*candidate.aliases, candidate.name])

        fresh = ResolvedEntity(key=key, name=candidate.name.strip(), type=candidate.type)
        return fresh.with_aliases(candidate.aliases)

    def _fuzzy_candidates(
        self,
        normalized: str,
        candidate: ExtractedEntity,
        known: dict[str, ResolvedEntity],
    ) -> list[ResolvedEntity]:
        found: list[ResolvedEntity] = []
        for entity in known.values():
            if entity.type is not candidate.type:
                continue
            other = normalize_name(entity.name)
            if normalized in other or other in normalized:
                found.append(entity)
        return found

    def _adjudicate(
        self, candidate: ExtractedEntity, candidates: list[ResolvedEntity]
    ) -> ResolvedEntity | None:
        listing = "\n".join(
            f"  - key: {entity.key}\n    name: {entity.name}" for entity in candidates
        )
        prompt = RESOLUTION_PROMPT.format(
            name=candidate.name, type=candidate.type.value, candidates=listing
        )
        verdict = self.llm.structured(prompt, EntityMatch)
        if not verdict.match or not verdict.matched_key:
            return None
        # A key the model invented resolves to nothing; treat it as no match.
        return next((e for e in candidates if e.key == verdict.matched_key), None)
