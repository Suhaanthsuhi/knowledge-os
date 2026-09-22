from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.knowledge.llm import StructuredLLM
from app.rag.retriever import RetrievalResult

__all__ = [
    "DraftAnswer",
    "Citation",
    "Answer",
    "build_context",
    "answer_question",
    "NO_EVIDENCE",
    "ANSWER_PROMPT",
    "CITATION_PATTERN",
]

NO_EVIDENCE = (
    "No evidence was found in the ingested documents for this question. "
    "Try ingesting the relevant document, or rephrasing using terms that "
    "appear in it."
)

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_.\-]+:p\d+e\d+)\]")

ANSWER_PROMPT = """\
Answer the question using only the evidence below. The evidence is everything
you know; your own knowledge is not admissible here.

Rules:
1. Use only the facts and evidence provided. If they do not answer the
   question, say so plainly and stop.
2. Cite every claim with the evidence id in square brackets, exactly as shown,
   for example [{example}].
3. Never invent an evidence id. Only ids that appear below exist.
4. Be direct. Two or three sentences is usually enough.

Question: {question}

{context}
"""


class DraftAnswer(BaseModel):
    answer: str = Field(description="The answer, with [evidence-id] citations")


class Citation(BaseModel):
    global_id: str
    doc_id: str
    element_id: str
    page_number: int
    bbox: tuple[float, float, float, float]
    text: str
    source_name: str = ""


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    dropped_citations: list[str] = Field(default_factory=list)
    grounded: bool = False
    fact_count: int = 0


def build_context(result: RetrievalResult) -> str:
    """Render facts and evidence into the only material the model may use."""
    if result.is_empty:
        return ""

    sections: list[str] = []

    if result.facts:
        lines = []
        for fact in result.facts:
            # Render provenance in the same bracket form as evidence, so a fact
            # is citable too. Without this the model reads a fact, answers from
            # it, and cites nothing.
            cites = " ".join(f"[{gid}]" for gid in fact.global_element_ids())
            lines.append(f"- {fact.sentence()}  (hop {fact.hop}) {cites}".rstrip())
        sections.append("Facts from the knowledge graph:\n" + "\n".join(lines))

    if result.evidence:
        blocks = []
        for evidence in result.evidence:
            passage = evidence.passage
            location = f"{passage.source_name or passage.doc_id}, page {passage.page_number}"
            blocks.append(f"[{passage.global_id}] ({location})\n{passage.text.strip()}")
        sections.append("Evidence:\n\n" + "\n\n".join(blocks))

    return "\n\n".join(sections)


def answer_question(llm: StructuredLLM, result: RetrievalResult) -> Answer:
    """Answer strictly from retrieved evidence, validating every citation.

    A citation naming an element that was not retrieved is dropped rather than
    rendered: an invented id that looks real is worse than a missing one.
    """
    if result.is_empty:
        return Answer(text=NO_EVIDENCE, grounded=False)

    context = build_context(result)
    known = {evidence.passage.global_id: evidence.passage for evidence in result.evidence}
    example = next(iter(known), "doc:p1e000")

    prompt = ANSWER_PROMPT.format(
        question=result.question, context=context, example=example
    )

    try:
        draft = llm.structured(prompt, DraftAnswer)
    except Exception as error:
        return Answer(
            text=f"The answer could not be generated: {error}",
            grounded=False,
            fact_count=len(result.facts),
        )

    text = draft.answer.strip()
    citations: list[Citation] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for global_id in CITATION_PATTERN.findall(text):
        if global_id in seen:
            continue
        seen.add(global_id)
        passage = known.get(global_id)
        if passage is None:
            dropped.append(global_id)
            continue
        citations.append(
            Citation(
                global_id=global_id,
                doc_id=passage.doc_id,
                element_id=passage.element_id,
                page_number=passage.page_number,
                bbox=(passage.x0, passage.y0, passage.x1, passage.y1),
                text=passage.text,
                source_name=passage.source_name,
            )
        )

    return Answer(
        text=text,
        citations=citations,
        dropped_citations=dropped,
        grounded=bool(citations),
        fact_count=len(result.facts),
    )
