"""Conversational layer over retrieval.

A chat differs from a search box in one respect that matters to retrieval: a
follow-up question often names nothing. "Who owns it?" has no entity to seed a
graph traversal on and no distinctive term for BM25. So each turn is rewritten
into a standalone question first, and that rewrite drives retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.knowledge.llm import StructuredLLM
from app.rag.answer import Answer, answer_question
from app.rag.retriever import RetrievalResult, Retriever

__all__ = [
    "StandaloneQuestion",
    "ChatTurn",
    "QueryContextualizer",
    "ChatEngine",
    "HISTORY_TURNS",
    "REWRITE_PROMPT",
]

HISTORY_TURNS = 4
CARRIED_SEEDS = 3

REWRITE_PROMPT = """\
Rewrite the user's latest question into one that can be understood on its own,
without the conversation.

Rules:
1. Every pronoun and implicit reference — "it", "that", "they", "the service",
   "the team" — MUST be replaced with the specific name it refers to, taken
   from the conversation. A rewritten question containing a pronoun is wrong.
2. Keep the user's intent exactly. Do not answer it, narrow it or broaden it.
3. Only if the question contains no references to the conversation at all,
   return it unchanged.
4. Return only the rewritten question, nothing else.

Example:
  Conversation:
    User: What is INC-2391?
    Assistant: An incident that affected the Payment API.
  Latest question: who owns it?
  Rewritten: Who owns the Payment API?

Conversation so far:
{history}

Latest question: {question}
"""


class StandaloneQuestion(BaseModel):
    question: str = Field(description="The rewritten, self-contained question")


@dataclass
class ChatTurn:
    question: str
    rewritten: str
    answer: Answer
    result: RetrievalResult
    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0

    @property
    def was_rewritten(self) -> bool:
        return self.rewritten.strip() != self.question.strip()

    @property
    def total_seconds(self) -> float:
        return self.retrieval_seconds + self.generation_seconds


class QueryContextualizer:
    """Turn a context-dependent question into a standalone one."""

    def __init__(self, llm: StructuredLLM | None = None) -> None:
        self.llm = llm

    def rewrite(self, question: str, history: list[ChatTurn]) -> str:
        # The first question is standalone by definition; skip the call.
        if not history or self.llm is None:
            return question

        recent = history[-HISTORY_TURNS:]
        transcript = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer.text}" for turn in recent
        )
        try:
            reply = self.llm.structured(
                REWRITE_PROMPT.format(history=transcript, question=question),
                StandaloneQuestion,
            )
        except Exception:
            # A failed rewrite must not cost the user their question.
            return question

        rewritten = reply.question.strip()
        return rewritten or question


class ChatEngine:
    """One turn: contextualize, retrieve, answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: StructuredLLM,
        contextualizer: QueryContextualizer | None = None,
        *,
        limit: int = 8,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.contextualizer = contextualizer or QueryContextualizer(llm)
        self.limit = limit

    @staticmethod
    def _carried_seeds(history: list[ChatTurn]) -> list[str]:
        """Entity names the previous turn actually matched in the graph.

        Rewriting resolves most references, but not reliably: "who owns it?"
        after a question about an incident may be rewritten as "who owns the
        incident?", which names nothing the graph can seed on. Carrying the
        previous turn's matched entities forward keeps the traversal anchored.
        """
        if not history:
            return []
        return [seed.name for seed in history[-1].result.seeds][:CARRIED_SEEDS]

    def ask(self, question: str, history: list[ChatTurn] | None = None) -> ChatTurn:
        history = history or []

        rewritten = self.contextualizer.rewrite(question, history)
        carried = self._carried_seeds(history)

        started = time.perf_counter()
        result = self.retriever.retrieve(
            rewritten, limit=self.limit, extra_names=carried
        )
        retrieved_at = time.perf_counter()
        answer = answer_question(self.llm, result)
        finished = time.perf_counter()

        result.trace["question"] = question
        result.trace["rewritten"] = rewritten
        result.trace["history_turns"] = len(history)

        return ChatTurn(
            question=question,
            rewritten=rewritten,
            answer=answer,
            result=result,
            retrieval_seconds=round(retrieved_at - started, 3),
            generation_seconds=round(finished - retrieved_at, 3),
        )
