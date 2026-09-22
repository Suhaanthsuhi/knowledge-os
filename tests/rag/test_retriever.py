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
from app.rag.bm25 import Bm25Index, Passage
from app.rag.retriever import (
    Bm25Retriever,
    GraphRetriever,
    HybridRetriever,
    QuestionAnalyzer,
    reciprocal_rank_fusion,
)

PAYMENT = entity_key("Payment API", EntityType.SERVICE)
TEAM = entity_key("Payments Team", EntityType.TEAM)
INCIDENT = entity_key("INC-2391", EntityType.INCIDENT)


def _element(eid: str, text: str) -> ElementRef:
    return ElementRef(id=eid, doc_id="d", type="paragraph", page_number=1,
                      x0=0, y0=0, x1=10, y1=10, order=0, text=text)


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
                          element_ids=["p1e003"], evidence="INC-2391 affected the Payment API"),
                GraphFact(source_key=TEAM, target_key=PAYMENT, relation=RelationType.OWNS,
                          doc_id="d", element_ids=["p1e002"],
                          evidence="The Payments Team owns the Payment API"),
            ],
            elements=[
                _element("p1e002", "The Payments Team owns the Payment API."),
                _element("p1e003", "INC-2391 affected the Payment API."),
            ],
        )
    )
    return store


def _passage(element_id: str, text: str) -> Passage:
    return Passage(doc_id="d", element_id=element_id, page_number=1,
                   element_type="paragraph", text=text, x0=0, y0=0, x1=10, y1=10,
                   source_name="p.md")


def _index() -> Bm25Index:
    return Bm25Index([
        _passage("p1e001", "The Payment API depends on Redis for caching."),
        _passage("p1e002", "The Payments Team owns the Payment API."),
        _passage("p1e003", "INC-2391 affected the Payment API."),
    ])


def test_rrf_favours_an_item_ranked_well_by_both_systems():
    scores = reciprocal_rank_fusion({"graph": ["a", "b"], "bm25": ["b", "a"]})
    assert scores["a"] == scores["b"]

    scores = reciprocal_rank_fusion({"graph": ["a", "b"], "bm25": ["a", "b"]})
    assert scores["a"] > scores["b"]


def test_rrf_includes_items_seen_by_only_one_system():
    assert set(reciprocal_rank_fusion({"graph": ["a"], "bm25": ["b"]})) == {"a", "b"}


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion({}) == {}


def test_the_analyzer_asks_the_model_for_entity_names():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    assert QuestionAnalyzer(llm).entity_names("What did INC-2391 affect?") == ["INC-2391"]


def test_the_analyzer_falls_back_to_a_heuristic_without_a_model():
    names = QuestionAnalyzer(None).entity_names("Who owns the Payment API after INC-2391?")
    assert "Payment API" in names or "INC-2391" in names


def test_the_analyzer_survives_a_model_failure():
    llm = FakeLLM(responses=[RuntimeError("groq down")])
    assert QuestionAnalyzer(llm).entity_names("Who owns the Payment API?")


def test_graph_retrieval_answers_a_two_hop_question():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    result = GraphRetriever(_store(), QuestionAnalyzer(llm), hops=2).retrieve(
        "Which team owns the service affected by INC-2391?"
    )

    assert [seed.key for seed in result.seeds] == [INCIDENT]
    assert any(fact.relation is RelationType.OWNS for fact in result.facts)
    assert {e.passage.element_id for e in result.evidence} == {"p1e002", "p1e003"}


def test_graph_evidence_is_tagged_with_its_source():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    result = GraphRetriever(_store(), QuestionAnalyzer(llm)).retrieve("q")
    assert all("graph" in evidence.sources for evidence in result.evidence)


def test_graph_retrieval_is_empty_when_no_entity_matches():
    llm = FakeLLM(responses=[QuestionEntities(names=["Kubernetes"])])
    result = GraphRetriever(_store(), QuestionAnalyzer(llm)).retrieve("q")

    assert result.is_empty
    assert result.facts == []


def test_lexical_retrieval_finds_the_identifier():
    result = Bm25Retriever(_index()).retrieve("INC-2391")
    assert result.evidence[0].passage.element_id == "p1e003"
    assert result.evidence[0].sources == ("bm25",)


def test_hybrid_retrieval_merges_both_rankings():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(GraphRetriever(_store(), QuestionAnalyzer(llm)),
                             Bm25Retriever(_index()))

    result = hybrid.retrieve("Which team owns the service affected by INC-2391?")
    found = {evidence.passage.element_id for evidence in result.evidence}

    assert {"p1e002", "p1e003"} <= found
    assert result.facts


def test_hybrid_marks_evidence_found_by_both_systems():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(GraphRetriever(_store(), QuestionAnalyzer(llm)),
                             Bm25Retriever(_index()))

    result = hybrid.retrieve("INC-2391 Payment API")
    assert [e for e in result.evidence if set(e.sources) == {"graph", "bm25"}]


def test_hybrid_degrades_to_lexical_when_the_graph_is_empty():
    llm = FakeLLM(responses=[QuestionEntities(names=[])])
    hybrid = HybridRetriever(
        GraphRetriever(InMemoryGraphStore(), QuestionAnalyzer(llm)),
        Bm25Retriever(_index()),
    )

    result = hybrid.retrieve("Payments Team")
    assert result.evidence
    assert result.facts == []


def test_hybrid_respects_the_limit():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(GraphRetriever(_store(), QuestionAnalyzer(llm)),
                             Bm25Retriever(_index()))
    assert len(hybrid.retrieve("Payment API", limit=1).evidence) == 1


def test_the_trace_records_what_retrieval_did():
    llm = FakeLLM(responses=[QuestionEntities(names=["INC-2391"])])
    hybrid = HybridRetriever(GraphRetriever(_store(), QuestionAnalyzer(llm)),
                             Bm25Retriever(_index()))

    trace = hybrid.retrieve("INC-2391").trace
    assert trace["seed_names"] == ["INC-2391"]
    assert trace["graph_hits"] >= 1
    assert trace["bm25_hits"] >= 1
    assert "hops" in trace
