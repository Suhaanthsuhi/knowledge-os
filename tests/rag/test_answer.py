from __future__ import annotations

from app.graph.store import FactRecord
from app.knowledge.llm import FakeLLM
from app.knowledge.schema import EntityType, RelationType
from app.rag.answer import NO_EVIDENCE, DraftAnswer, answer_question, build_context
from app.rag.bm25 import Passage
from app.rag.retriever import Evidence, RetrievalResult


def _evidence(element_id: str, text: str) -> Evidence:
    return Evidence(
        passage=Passage(doc_id="d", element_id=element_id, page_number=3,
                        element_type="paragraph", text=text,
                        x0=72, y0=100, x1=500, y1=140, source_name="payments.pdf"),
        score=0.5,
        sources=("graph",),
    )


def _fact() -> FactRecord:
    return FactRecord(
        source_key="team|payments team", source_name="Payments Team",
        source_type=EntityType.TEAM, relation=RelationType.OWNS,
        target_key="service|payment api", target_name="Payment API",
        target_type=EntityType.SERVICE, doc_id="d", element_ids=["p1e002"],
        evidence="The Payments Team owns the Payment API.", hop=2,
    )


def _result(**overrides) -> RetrievalResult:
    base = {
        "question": "Which team owns the Payment API?",
        "evidence": [_evidence("p1e002", "The Payments Team owns the Payment API.")],
        "facts": [_fact()],
    }
    base.update(overrides)
    return RetrievalResult(**base)


def test_context_includes_the_facts_and_the_evidence():
    context = build_context(_result())
    assert "Payments Team OWNS Payment API" in context
    assert "[d:p1e002]" in context
    assert "payments.pdf" in context
    assert "page 3" in context


def test_context_of_an_empty_result_is_empty():
    assert build_context(RetrievalResult(question="q")) == ""


def test_an_empty_retrieval_never_reaches_the_model():
    llm = FakeLLM(responses=[])
    answer = answer_question(llm, RetrievalResult(question="q"))

    assert answer.text == NO_EVIDENCE
    assert answer.grounded is False
    assert llm.prompts == []


def test_a_grounded_answer_keeps_resolvable_citations():
    llm = FakeLLM(responses=[DraftAnswer(answer="The Payments Team owns it [d:p1e002].")])
    answer = answer_question(llm, _result())

    assert answer.grounded is True
    assert [citation.global_id for citation in answer.citations] == ["d:p1e002"]
    assert answer.citations[0].page_number == 3
    assert answer.citations[0].bbox == (72.0, 100.0, 500.0, 140.0)


def test_a_fabricated_citation_is_dropped_and_reported():
    llm = FakeLLM(responses=[
        DraftAnswer(answer="Owned by the team [d:p1e002], see also [d:p9e999].")
    ])
    answer = answer_question(llm, _result())

    assert [c.global_id for c in answer.citations] == ["d:p1e002"]
    assert answer.dropped_citations == ["d:p9e999"]


def test_an_answer_with_no_citations_is_not_marked_grounded():
    llm = FakeLLM(responses=[DraftAnswer(answer="The Payments Team owns it.")])
    answer = answer_question(llm, _result())

    assert answer.citations == []
    assert answer.grounded is False


def test_duplicate_citations_are_reported_once():
    llm = FakeLLM(responses=[DraftAnswer(answer="See [d:p1e002] and again [d:p1e002].")])
    assert len(answer_question(llm, _result()).citations) == 1


def test_the_prompt_forbids_answering_beyond_the_evidence():
    llm = FakeLLM(responses=[DraftAnswer(answer="ok [d:p1e002]")])
    answer_question(llm, _result())

    prompt = llm.prompts[0].lower()
    assert "only" in prompt
    assert "evidence" in prompt


def test_the_prompt_carries_the_question_and_the_context():
    llm = FakeLLM(responses=[DraftAnswer(answer="ok [d:p1e002]")])
    answer_question(llm, _result())

    assert "Which team owns the Payment API?" in llm.prompts[0]
    assert "[d:p1e002]" in llm.prompts[0]


def test_a_model_failure_surfaces_as_an_ungrounded_answer():
    llm = FakeLLM(responses=[RuntimeError("groq exploded")])
    answer = answer_question(llm, _result())

    assert answer.grounded is False
    assert "groq exploded" in answer.text


def test_the_fact_count_is_reported():
    llm = FakeLLM(responses=[DraftAnswer(answer="ok [d:p1e002]")])
    assert answer_question(llm, _result()).fact_count == 1
