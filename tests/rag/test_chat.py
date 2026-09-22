from __future__ import annotations

from app.graph.memory import InMemoryGraphStore
from app.knowledge.llm import FakeLLM
from app.knowledge.schema import (
    DocumentRef,
    ElementRef,
    EntityType,
    GraphFact,
    GraphUpsert,
    QuestionEntities,
    RelationType,
    ResolvedEntity,
    entity_key,
)
from app.rag.answer import NO_EVIDENCE, DraftAnswer
from app.rag.bm25 import Bm25Index, Passage
from app.rag.chat import ChatEngine, ChatTurn, QueryContextualizer, StandaloneQuestion
from app.rag.retriever import Bm25Retriever, GraphRetriever, HybridRetriever, QuestionAnalyzer

PAYMENT = entity_key("Payment API", EntityType.SERVICE)
TEAM = entity_key("Payments Team", EntityType.TEAM)
INCIDENT = entity_key("INC-2391", EntityType.INCIDENT)


def _store() -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    store.apply(
        GraphUpsert(
            document=DocumentRef(id="d", source_name="p.md", mime="text/markdown"),
            entities=[
                ResolvedEntity(key=PAYMENT, name="Payment API", type=EntityType.SERVICE),
                ResolvedEntity(key=TEAM, name="Payments Team", type=EntityType.TEAM),
                ResolvedEntity(key=INCIDENT, name="INC-2391", type=EntityType.INCIDENT),
            ],
            facts=[
                GraphFact(source_key=INCIDENT, target_key=PAYMENT,
                          relation=RelationType.AFFECTS, doc_id="d",
                          element_ids=["p1e003"], evidence="INC-2391 affected it"),
                GraphFact(source_key=TEAM, target_key=PAYMENT, relation=RelationType.OWNS,
                          doc_id="d", element_ids=["p1e002"], evidence="team owns it"),
            ],
            elements=[
                ElementRef(id="p1e002", doc_id="d", type="paragraph", page_number=1,
                           x0=0, y0=0, x1=10, y1=10, order=2,
                           text="The Payments Team owns the Payment API."),
                ElementRef(id="p1e003", doc_id="d", type="paragraph", page_number=1,
                           x0=0, y0=20, x1=10, y1=30, order=3,
                           text="INC-2391 affected the Payment API."),
            ],
        )
    )
    return store


def _index() -> Bm25Index:
    return Bm25Index([
        Passage(doc_id="d", element_id="p1e002", page_number=1, element_type="paragraph",
                text="The Payments Team owns the Payment API.", x0=0, y0=0, x1=10, y1=10,
                source_name="p.md"),
        Passage(doc_id="d", element_id="p1e003", page_number=1, element_type="paragraph",
                text="INC-2391 affected the Payment API.", x0=0, y0=20, x1=10, y1=30,
                source_name="p.md"),
    ])


def _turn(question: str, text: str) -> ChatTurn:
    from app.rag.answer import Answer
    from app.rag.retriever import RetrievalResult

    return ChatTurn(question=question, rewritten=question,
                    answer=Answer(text=text), result=RetrievalResult(question=question))


def test_the_first_question_is_not_rewritten_and_costs_no_call():
    llm = FakeLLM(responses=[])
    assert QueryContextualizer(llm).rewrite("What is INC-2391?", []) == "What is INC-2391?"
    assert llm.prompts == []


def test_a_follow_up_is_rewritten_into_a_standalone_question():
    llm = FakeLLM(responses=[
        StandaloneQuestion(question="Who owns the Payment API affected by INC-2391?")
    ])
    history = [_turn("What is INC-2391?", "An incident affecting the Payment API.")]

    rewritten = QueryContextualizer(llm).rewrite("who owns it?", history)

    assert rewritten == "Who owns the Payment API affected by INC-2391?"
    assert "What is INC-2391?" in llm.prompts[0]


def test_the_rewrite_prompt_carries_only_recent_turns():
    from app.rag.chat import HISTORY_TURNS

    llm = FakeLLM(responses=[StandaloneQuestion(question="x")])
    history = [_turn(f"question {i}", f"answer {i}") for i in range(10)]
    QueryContextualizer(llm).rewrite("and then?", history)

    prompt = llm.prompts[0]
    assert "question 9" in prompt
    assert "question 0" not in prompt
    # Count only the history turns; the prompt's worked example has a User line
    # of its own.
    assert prompt.count("User: question ") == HISTORY_TURNS


def test_a_failed_rewrite_falls_back_to_the_raw_question():
    llm = FakeLLM(responses=[RuntimeError("groq down")])
    history = [_turn("What is INC-2391?", "An incident.")]

    assert QueryContextualizer(llm).rewrite("who owns it?", history) == "who owns it?"


def test_an_empty_rewrite_falls_back_to_the_raw_question():
    llm = FakeLLM(responses=[StandaloneQuestion(question="   ")])
    history = [_turn("What is INC-2391?", "An incident.")]

    assert QueryContextualizer(llm).rewrite("who owns it?", history) == "who owns it?"


def test_rewriting_is_skipped_without_a_model():
    assert QueryContextualizer(None).rewrite("who owns it?", [_turn("q", "a")]) == "who owns it?"


def _engine(llm: FakeLLM) -> ChatEngine:
    retriever = HybridRetriever(
        GraphRetriever(_store(), QuestionAnalyzer(llm)), Bm25Retriever(_index())
    )
    return ChatEngine(retriever, llm)


def test_a_first_turn_retrieves_and_answers():
    llm = FakeLLM(responses=[
        QuestionEntities(names=["INC-2391"]),
        DraftAnswer(answer="It affected the Payment API [d:p1e003]."),
    ])
    turn = _engine(llm).ask("What is INC-2391?")

    assert turn.answer.grounded is True
    assert turn.was_rewritten is False
    assert turn.result.facts


def test_a_follow_up_retrieves_on_the_rewritten_question():
    llm = FakeLLM(responses=[
        StandaloneQuestion(question="Who owns the Payment API?"),
        QuestionEntities(names=["Payment API"]),
        DraftAnswer(answer="The Payments Team [d:p1e002]."),
    ])
    history = [_turn("What is INC-2391?", "An incident affecting the Payment API.")]

    turn = _engine(llm).ask("who owns it?", history)

    assert turn.rewritten == "Who owns the Payment API?"
    assert turn.was_rewritten is True
    assert turn.result.question == "Who owns the Payment API?"
    assert turn.answer.grounded is True


def test_the_trace_records_both_forms_of_the_question():
    llm = FakeLLM(responses=[
        StandaloneQuestion(question="Who owns the Payment API?"),
        QuestionEntities(names=["Payment API"]),
        DraftAnswer(answer="The Payments Team [d:p1e002]."),
    ])
    turn = _engine(llm).ask("who owns it?", [_turn("What is INC-2391?", "An incident.")])

    assert turn.result.trace["question"] == "who owns it?"
    assert turn.result.trace["rewritten"] == "Who owns the Payment API?"
    assert turn.result.trace["history_turns"] == 1


def test_an_empty_corpus_produces_the_no_evidence_answer():
    llm = FakeLLM(responses=[QuestionEntities(names=[])])
    engine = ChatEngine(
        HybridRetriever(
            GraphRetriever(InMemoryGraphStore(), QuestionAnalyzer(llm)),
            Bm25Retriever(Bm25Index([])),
        ),
        llm,
    )
    turn = engine.ask("Which team owns the Payment API?")

    assert turn.answer.text == NO_EVIDENCE
    assert turn.answer.grounded is False


def test_timings_are_recorded_per_phase():
    llm = FakeLLM(responses=[
        QuestionEntities(names=["INC-2391"]),
        DraftAnswer(answer="ok [d:p1e003]"),
    ])
    turn = _engine(llm).ask("What is INC-2391?")

    assert turn.retrieval_seconds >= 0
    assert turn.generation_seconds >= 0
    assert turn.total_seconds == turn.retrieval_seconds + turn.generation_seconds


def test_seeds_are_carried_forward_when_a_follow_up_names_nothing():
    """A follow-up whose rewrite names no known entity still reaches the graph."""
    llm = FakeLLM(responses=[
        # A defensible but unhelpful rewrite: nothing here seeds the graph.
        StandaloneQuestion(question="Who owns the incident?"),
        QuestionEntities(names=["the incident"]),
        DraftAnswer(answer="The Payments Team [d:p1e002]."),
    ])
    first = _turn("What is INC-2391?", "An incident affecting the Payment API.")
    first.result.seeds = [
        ResolvedEntity(key=INCIDENT, name="INC-2391", type=EntityType.INCIDENT)
    ]

    turn = _engine(llm).ask("who owns it?", [first])

    assert turn.result.trace["carried_names"] == ["INC-2391"]
    assert turn.result.facts, "carried seed did not reach the graph"
    assert any(fact.relation is RelationType.OWNS for fact in turn.result.facts)


def test_the_first_turn_carries_nothing():
    llm = FakeLLM(responses=[
        QuestionEntities(names=["INC-2391"]),
        DraftAnswer(answer="ok [d:p1e003]"),
    ])
    turn = _engine(llm).ask("What is INC-2391?")
    assert turn.result.trace["carried_names"] == []


def test_carried_seeds_are_capped():
    from app.rag.chat import CARRIED_SEEDS

    llm = FakeLLM(responses=[
        StandaloneQuestion(question="and then?"),
        QuestionEntities(names=[]),
        DraftAnswer(answer="ok [d:p1e002]"),
    ])
    first = _turn("q", "a")
    first.result.seeds = [
        ResolvedEntity(key=f"service|s{i}", name=f"S{i}", type=EntityType.SERVICE)
        for i in range(10)
    ]

    turn = _engine(llm).ask("and then?", [first])
    assert len(turn.result.trace["carried_names"]) == CARRIED_SEEDS


def test_lexical_retrieval_ignores_carried_names():
    """Carried seeds anchor the graph; injecting them into BM25 would drag every
    earlier topic into the ranking."""
    from app.rag.retriever import Bm25Retriever

    plain = Bm25Retriever(_index()).retrieve("INC-2391")
    carried = Bm25Retriever(_index()).retrieve("INC-2391", extra_names=["Payments Team"])

    assert [e.passage.element_id for e in plain.evidence] == [
        e.passage.element_id for e in carried.evidence
    ]
