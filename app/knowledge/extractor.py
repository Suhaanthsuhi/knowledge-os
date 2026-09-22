from __future__ import annotations

from app.knowledge.llm import StructuredLLM
from app.knowledge.schema import Chunk, ChunkExtraction, EntityType, RelationType

__all__ = ["KnowledgeExtractor", "EXTRACTION_PROMPT"]

EXTRACTION_PROMPT = """\
Extract a knowledge graph from one section of an enterprise document.

Entity types (use exactly one of these):
{entity_types}

Relation types (use exactly one of these):
{relation_types}

Rules:
1. Extract only what the text states. Do not infer, complete or guess.
2. For every fact, quote the evidence verbatim from the text below. Copy the
   words exactly; do not paraphrase, summarise or re-punctuate.
3. Every fact's source and target must also appear in the entities list.
4. Prefer specific entity types over Other.
5. If the text states no relationships, return empty lists.

Section heading: {heading}

Text:
{text}
"""


class KnowledgeExtractor:
    """One structured model call per chunk."""

    def __init__(self, llm: StructuredLLM) -> None:
        self.llm = llm

    def extract(self, chunk: Chunk) -> ChunkExtraction:
        if not chunk.text.strip():
            return ChunkExtraction()

        prompt = EXTRACTION_PROMPT.format(
            entity_types="\n".join(f"- {t.value}" for t in EntityType),
            relation_types="\n".join(f"- {r.value}" for r in RelationType),
            heading=chunk.heading or "(none)",
            text=chunk.text,
        )
        return self.llm.structured(prompt, ChunkExtraction)
