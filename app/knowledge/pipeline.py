from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from app.ir.model import Document, Element
from app.knowledge.chunker import chunk_document
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import (
    Chunk,
    DocumentRef,
    ElementRef,
    GraphFact,
    GraphUpsert,
    ResolvedEntity,
    normalize_name,
)

__all__ = ["KnowledgePipeline", "bind_evidence", "element_ref"]

_WHITESPACE = re.compile(r"\s+")


def _flatten(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


def bind_evidence(
    quote: str, chunk: Chunk, elements: Mapping[str, Element]
) -> tuple[list[str], str]:
    """Locate the element a fact's evidence quote came from.

    Chunk-level provenance would only narrow a fact to a few paragraphs. Matching
    the quote pins it to one element, which is what makes a citation a rectangle
    on a page rather than a region of a document.
    """
    needle = _flatten(quote)
    if needle:
        for element_id in chunk.element_ids:
            element = elements.get(element_id)
            if element is None:
                continue
            if needle in _flatten(element.text_content()):
                return [element_id], "high"

    # No match: keep the fact, but say honestly that it is only chunk-level.
    return list(chunk.element_ids), "low"


def element_ref(doc: Document, element: Element) -> ElementRef:
    return ElementRef(
        id=element.id,
        doc_id=doc.id,
        type=element.type.value,
        page_number=element.page_number,
        x0=element.bbox.x0,
        y0=element.bbox.y0,
        x1=element.bbox.x1,
        y1=element.bbox.y1,
        order=element.order,
        text=element.text_content(),
    )


class KnowledgePipeline:
    """Document in, GraphUpsert out.

    Extraction is isolated per chunk: a model failure on one chunk is recorded
    and the remaining chunks still contribute, so a single bad call does not
    cost the whole document.
    """

    def __init__(self, extractor: KnowledgeExtractor, resolver: EntityResolver) -> None:
        self.extractor = extractor
        self.resolver = resolver

    def run(
        self,
        doc: Document,
        known: Sequence[ResolvedEntity] = (),
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> GraphUpsert:
        upsert = GraphUpsert(
            document=DocumentRef(
                id=doc.id,
                source_name=doc.source_name,
                mime=doc.mime,
                title=doc.title,
                checksum=doc.checksum,
            )
        )

        elements = {element.id: element for element in doc.iter_elements()}
        registry: dict[str, ResolvedEntity] = {entity.key: entity for entity in known}
        touched_entities: dict[str, ResolvedEntity] = {}
        evidence_elements: dict[str, ElementRef] = {}

        chunks = chunk_document(doc)
        for index, chunk in enumerate(chunks, start=1):
            try:
                extraction = self.extractor.extract(chunk)
            except Exception as error:  # one bad chunk must not cost the document
                upsert.add_warning("chunk_failed", f"{chunk.id}: {error}")
                if on_progress:
                    on_progress(index, len(chunks))
                continue

            by_name: dict[str, ResolvedEntity] = {}
            for candidate in extraction.entities:
                if not candidate.name.strip():
                    continue
                resolved = self.resolver.resolve(candidate, registry)
                registry[resolved.key] = resolved
                touched_entities[resolved.key] = resolved
                by_name[normalize_name(candidate.name)] = resolved
                for alias in candidate.aliases:
                    by_name.setdefault(normalize_name(alias), resolved)

            for fact in extraction.facts:
                source = by_name.get(normalize_name(fact.source))
                target = by_name.get(normalize_name(fact.target))
                if source is None or target is None:
                    missing = fact.source if source is None else fact.target
                    upsert.add_warning(
                        "unknown_entity",
                        f"{chunk.id}: fact names {missing!r}, which was not extracted",
                    )
                    continue
                if source.key == target.key:
                    upsert.add_warning(
                        "self_reference", f"{chunk.id}: {source.name} relates to itself"
                    )
                    continue

                element_ids, confidence = bind_evidence(fact.evidence, chunk, elements)
                for element_id in element_ids:
                    element = elements.get(element_id)
                    if element is not None:
                        evidence_elements[element_id] = element_ref(doc, element)

                upsert.facts.append(
                    GraphFact(
                        source_key=source.key,
                        target_key=target.key,
                        relation=fact.relation,
                        doc_id=doc.id,
                        element_ids=element_ids,
                        evidence=fact.evidence.strip(),
                        confidence=confidence,
                    )
                )

            if on_progress:
                on_progress(index, len(chunks))

        upsert.entities = list(touched_entities.values())
        upsert.elements = list(evidence_elements.values())
        return upsert
