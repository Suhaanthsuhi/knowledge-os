from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from app.graph.store import FactRecord, GraphStore
from app.knowledge.llm import StructuredLLM
from app.knowledge.schema import QuestionEntities, ResolvedEntity
from app.rag.bm25 import Bm25Index, Passage, passage_from_element_ref

__all__ = [
    "Evidence",
    "RetrievalResult",
    "Retriever",
    "QuestionAnalyzer",
    "GraphRetriever",
    "Bm25Retriever",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "RRF_K",
    "DEFAULT_LIMIT",
]

RRF_K = 60
DEFAULT_LIMIT = 8

QUESTION_PROMPT = """\
List the named entities mentioned in this question — services, teams, people,
products, databases, incidents, error codes, identifiers.

Return names only, exactly as written in the question. Do not add entities that
are not present, and do not include generic words like "team" or "service" on
their own.

Question: {question}
"""

# Fallback when no model is available: runs of capitalised words, plus anything
# shaped like an identifier (INC-2391, PAYMENT_502).
_CAPITALISED = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ -][A-Z][A-Za-z0-9]*)*\b")
_IDENTIFIER = re.compile(r"\b[A-Z][A-Z0-9]*[-_][A-Z0-9]+\b")
_STOPWORDS = {"What", "Which", "Who", "Where", "When", "Why", "How", "The", "A", "An"}


@dataclass(frozen=True)
class Evidence:
    passage: Passage
    score: float
    sources: tuple[str, ...]


@dataclass
class RetrievalResult:
    question: str
    evidence: list[Evidence] = field(default_factory=list)
    facts: list[FactRecord] = field(default_factory=list)
    seeds: list[ResolvedEntity] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.evidence and not self.facts


class Retriever(Protocol):
    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult: ...


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], k: int = RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists without weights to tune.

    Each list contributes 1/(k + rank). An item both systems rank highly beats
    one that only a single system likes, which is the whole point of running a
    graph search and a lexical search side by side.
    """
    scores: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (k + rank)
    return scores


class QuestionAnalyzer:
    """Pull candidate entity names out of a question."""

    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self.llm = llm

    def entity_names(self, question: str) -> list[str]:
        if self.llm is not None:
            try:
                reply = self.llm.structured(
                    QUESTION_PROMPT.format(question=question), QuestionEntities
                )
                names = [name.strip() for name in reply.names if name.strip()]
                if names:
                    return names
            except Exception:
                # A question is still answerable without entity extraction.
                pass
        return self._heuristic(question)

    @staticmethod
    def _heuristic(question: str) -> list[str]:
        found = list(_IDENTIFIER.findall(question))
        for match in _CAPITALISED.findall(question):
            cleaned = match.strip()
            if cleaned and cleaned not in _STOPWORDS and cleaned not in found:
                found.append(cleaned)
        return found


class GraphRetriever:
    """Seed on entities named in the question, then expand k hops."""

    def __init__(self, store: GraphStore, analyzer: QuestionAnalyzer, hops: int = 2) -> None:
        self.store = store
        self.analyzer = analyzer
        self.hops = max(1, min(hops, 3))

    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        names = self.analyzer.entity_names(question)
        seeds = self.store.find_entities(names) if names else []
        trace: dict[str, Any] = {
            "seed_names": names,
            "seed_keys": [seed.key for seed in seeds],
            "hops": self.hops,
        }

        if not seeds:
            trace["graph_hits"] = 0
            return RetrievalResult(question=question, seeds=[], trace=trace)

        facts = self.store.traverse([seed.key for seed in seeds], hops=self.hops)
        trace["graph_hits"] = len(facts)

        ordered_ids: list[str] = []
        for fact in facts:
            for global_id in fact.global_element_ids():
                if global_id not in ordered_ids:
                    ordered_ids.append(global_id)

        refs = {ref.global_id: ref for ref in self.store.elements(ordered_ids)}
        evidence = [
            Evidence(
                passage=passage_from_element_ref(refs[global_id]),
                score=1.0 / rank,
                sources=("graph",),
            )
            for rank, global_id in enumerate(ordered_ids, start=1)
            if global_id in refs
        ]

        return RetrievalResult(
            question=question,
            evidence=evidence[:limit],
            facts=facts,
            seeds=seeds,
            trace=trace,
        )


class Bm25Retriever:
    def __init__(self, index: Bm25Index) -> None:
        self.index = index

    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        hits = self.index.search(question, limit=limit)
        return RetrievalResult(
            question=question,
            evidence=[
                Evidence(passage=passage, score=score, sources=("bm25",))
                for passage, score in hits
            ],
            trace={"bm25_hits": len(hits)},
        )


class HybridRetriever:
    """Graph structure plus lexical matching, fused with RRF."""

    def __init__(self, graph: GraphRetriever, lexical: Bm25Retriever) -> None:
        self.graph = graph
        self.lexical = lexical

    def retrieve(self, question: str, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        # Over-fetch each arm so fusion has something to work with.
        graph_result = self.graph.retrieve(question, limit=limit * 3)
        lexical_result = self.lexical.retrieve(question, limit=limit * 3)

        passages: dict[str, Passage] = {}
        rankings: dict[str, list[str]] = {"graph": [], "bm25": []}

        for name, result in (("graph", graph_result), ("bm25", lexical_result)):
            for evidence in result.evidence:
                global_id = evidence.passage.global_id
                passages.setdefault(global_id, evidence.passage)
                rankings[name].append(global_id)

        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))

        graph_ids = set(rankings["graph"])
        lexical_ids = set(rankings["bm25"])
        evidence: list[Evidence] = []
        for global_id, score in ordered[:limit]:
            sources = tuple(
                name
                for name, ids in (("graph", graph_ids), ("bm25", lexical_ids))
                if global_id in ids
            )
            evidence.append(
                Evidence(passage=passages[global_id], score=score, sources=sources)
            )

        trace = {
            **graph_result.trace,
            **lexical_result.trace,
            "fused": len(fused),
            "returned": len(evidence),
        }
        return RetrievalResult(
            question=question,
            evidence=evidence,
            facts=graph_result.facts,
            seeds=graph_result.seeds,
            trace=trace,
        )
