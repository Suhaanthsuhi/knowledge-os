from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from app.ir.model import Document, Element, ElementType

__all__ = [
    "element_index",
    "iter_elements",
    "SectionNode",
    "section_tree",
    "assign_parents",
]


def element_index(doc: Document) -> dict[str, Element]:
    """O(1) lookup of every element in the document by id."""
    index: dict[str, Element] = {}
    for page in doc.pages:
        for element in page.elements:
            if element.id in index:
                raise ValueError(f"duplicate element id: {element.id}")
            index[element.id] = element
    return index


def iter_elements(doc: Document) -> Iterator[Element]:
    """Walk every element in global reading order, across page boundaries."""
    everything = [element for page in doc.pages for element in page.elements]
    yield from sorted(everything, key=lambda e: e.order)


def assign_parents(elements: list[Element]) -> None:
    """Set `parent_id` on each element to its nearest enclosing heading.

    Mutates in place. A heading's parent is the nearest preceding heading of a
    strictly smaller level.
    """
    stack: list[Element] = []
    for element in sorted(elements, key=lambda e: e.order):
        if element.type is ElementType.HEADING:
            level = element.level or 1
            while stack and (stack[-1].level or 1) >= level:
                stack.pop()
            element.parent_id = stack[-1].id if stack else None
            stack.append(element)
        else:
            element.parent_id = stack[-1].id if stack else None


@dataclass
class SectionNode:
    """A heading and everything beneath it. `heading` is None for preamble content."""

    heading: Element | None
    elements: list[Element] = field(default_factory=list)
    children: list[SectionNode] = field(default_factory=list)

    @property
    def level(self) -> int:
        if self.heading is None:
            return 0
        return self.heading.level or 1

    def title(self) -> str:
        return (self.heading.text or "") if self.heading else ""


def section_tree(doc: Document) -> list[SectionNode]:
    """Derive the heading hierarchy on demand. Storage stays flat."""
    roots: list[SectionNode] = []
    stack: list[SectionNode] = []

    for element in iter_elements(doc):
        if element.type is ElementType.HEADING:
            node = SectionNode(heading=element)
            level = element.level or 1
            # A preamble node holds content that precedes every heading; it is a
            # sibling of the top-level sections, never their parent.
            while stack and (stack[-1].heading is None or stack[-1].level >= level):
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        else:
            if not stack:
                preamble = SectionNode(heading=None)
                roots.append(preamble)
                stack.append(preamble)
            stack[-1].elements.append(element)

    return roots
