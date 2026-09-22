from __future__ import annotations

import pytest

from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.llm import FakeLLM
from app.knowledge.schema import (
    Chunk,
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    RelationType,
)


def _chunk(text: str = "The Payment API depends on Redis.") -> Chunk:
    return Chunk(
        id="d:c000",
        doc_id="d",
        page_number=1,
        heading="Payments",
        text=text,
        element_ids=("p1e001",),
    )


def _extraction() -> ChunkExtraction:
    return ChunkExtraction(
        entities=[
            ExtractedEntity(name="Payment API", type=EntityType.SERVICE),
            ExtractedEntity(name="Redis", type=EntityType.DATABASE),
        ],
        facts=[
            ExtractedFact(
                source="Payment API",
                relation=RelationType.DEPENDS_ON,
                target="Redis",
                evidence="The Payment API depends on Redis.",
            )
        ],
    )


def test_extract_returns_the_model_output():
    llm = FakeLLM(responses=[_extraction()])
    result = KnowledgeExtractor(llm).extract(_chunk())

    assert [entity.name for entity in result.entities] == ["Payment API", "Redis"]
    assert result.facts[0].relation is RelationType.DEPENDS_ON


def test_the_prompt_carries_the_heading_and_the_text():
    llm = FakeLLM(responses=[_extraction()])
    KnowledgeExtractor(llm).extract(_chunk())

    prompt = llm.prompts[0]
    assert "Payments" in prompt
    assert "The Payment API depends on Redis." in prompt


def test_the_prompt_lists_the_allowed_vocabularies():
    llm = FakeLLM(responses=[_extraction()])
    KnowledgeExtractor(llm).extract(_chunk())

    prompt = llm.prompts[0]
    assert "DEPENDS_ON" in prompt
    assert "Service" in prompt


def test_the_prompt_demands_verbatim_evidence():
    llm = FakeLLM(responses=[_extraction()])
    KnowledgeExtractor(llm).extract(_chunk())
    assert "verbatim" in llm.prompts[0].lower()


def test_an_empty_chunk_is_not_sent_to_the_model():
    llm = FakeLLM(responses=[])
    result = KnowledgeExtractor(llm).extract(_chunk(text="   "))

    assert result == ChunkExtraction()
    assert llm.prompts == []


def test_model_failures_propagate():
    llm = FakeLLM(responses=[RuntimeError("groq down")])
    with pytest.raises(RuntimeError, match="groq down"):
        KnowledgeExtractor(llm).extract(_chunk())
