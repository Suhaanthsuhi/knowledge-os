from __future__ import annotations

from app.ir.model import Document, Element, ElementType
from app.ir.tree import SectionNode
from app.knowledge.schema import Chunk

__all__ = ["chunk_document", "MAX_CHUNK_CHARS", "SKIPPED_TYPES"]

MAX_CHUNK_CHARS = 1500

# Running headers, footers and page numbers repeat on every page. Feeding them
# to the extractor would produce the same non-facts once per page.
SKIPPED_TYPES = frozenset(
    {ElementType.HEADER, ElementType.FOOTER, ElementType.PAGE_NUMBER}
)


def _content(node: SectionNode) -> list[Element]:
    return [
        element
        for element in node.elements
        if element.type not in SKIPPED_TYPES and element.text_content().strip()
    ]


def _flush(
    doc: Document, heading: str | None, batch: list[Element], counter: list[int]
) -> Chunk:
    chunk = Chunk(
        id=f"{doc.id}:c{counter[0]:03d}",
        doc_id=doc.id,
        page_number=batch[0].page_number,
        heading=heading,
        text="\n\n".join(element.text_content().strip() for element in batch),
        element_ids=tuple(element.id for element in batch),
    )
    counter[0] += 1
    return chunk


def _walk(
    doc: Document,
    node: SectionNode,
    max_chars: int,
    counter: list[int],
    out: list[Chunk],
) -> None:
    heading = node.title() or None
    batch: list[Element] = []
    size = 0

    for element in _content(node):
        length = len(element.text_content())
        # Never split an element: an oversized one becomes its own chunk.
        if batch and size + length > max_chars:
            out.append(_flush(doc, heading, batch, counter))
            batch, size = [], 0
        batch.append(element)
        size += length
        if size >= max_chars:
            out.append(_flush(doc, heading, batch, counter))
            batch, size = [], 0

    if batch:
        out.append(_flush(doc, heading, batch, counter))

    for child in node.children:
        _walk(doc, child, max_chars, counter, out)


def chunk_document(doc: Document, *, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split a document into extraction units that remember where they came from.

    Sections are the natural boundary: a heading gives the extractor context its
    paragraphs lack on their own.
    """
    chunks: list[Chunk] = []
    counter = [0]
    for node in doc.section_tree():
        _walk(doc, node, max_chars, counter, chunks)
    return chunks
