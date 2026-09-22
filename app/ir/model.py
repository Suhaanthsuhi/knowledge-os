from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterator

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "BBox",
    "ElementType",
    "TableData",
    "ImageRef",
    "Provenance",
    "Element",
    "Page",
    "Document",
]


class BBox(BaseModel):
    """Axis-aligned rectangle in PDF points, origin top-left, y increasing downward."""

    x0: float
    y0: float
    x1: float
    y1: float

    @field_validator("x0", "y0", "x1", "y1")
    @classmethod
    def _round(cls, value: float) -> float:
        # Fixed precision keeps serialized output byte-identical across runs.
        return round(float(value), 2)

    @property
    def width(self) -> float:
        return round(self.x1 - self.x0, 2)

    @property
    def height(self) -> float:
        return round(self.y1 - self.y0, 2)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, other: BBox) -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def horizontal_overlap_ratio(self, other: BBox) -> float:
        """Overlap of the shared x-range as a fraction of *self*'s width."""
        if self.width <= 0:
            return 0.0
        overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        return max(0.0, overlap) / self.width


class ElementType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


class TableData(BaseModel):
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    cell_bboxes: list[list[BBox | None]] | None = None

    @model_validator(mode="after")
    def _check_dimensions(self) -> TableData:
        if self.n_rows != len(self.rows):
            raise ValueError(f"n_rows={self.n_rows} but got {len(self.rows)} rows")
        for row in self.rows:
            if len(row) != self.n_cols:
                raise ValueError(f"row has {len(row)} cells, expected {self.n_cols}")
        if self.headers and len(self.headers) != self.n_cols:
            raise ValueError(f"{len(self.headers)} headers, expected {self.n_cols}")
        return self

    def as_text(self) -> str:
        """Flat text rendering, used for lossless-text checks and fallbacks."""
        lines = []
        if self.headers:
            lines.append(" | ".join(self.headers))
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)


class ImageRef(BaseModel):
    blob_ref: str | None = None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    caption_id: str | None = None


class Provenance(BaseModel):
    """The single value every downstream chunk, fact and graph edge carries."""

    doc_id: str
    page_number: int
    element_id: str
    bbox: BBox


class Element(BaseModel):
    id: str
    type: ElementType
    page_number: int
    bbox: BBox
    order: int
    parent_id: str | None = None
    text: str | None = None
    level: int | None = None
    table: TableData | None = None
    image: ImageRef | None = None
    attrs: dict[str, Any] = Field(default_factory=dict)

    def provenance(self, doc_id: str) -> Provenance:
        return Provenance(
            doc_id=doc_id,
            page_number=self.page_number,
            element_id=self.id,
            bbox=self.bbox,
        )

    def text_content(self) -> str:
        """All text this element carries, including table cells."""
        if self.table is not None:
            return self.table.as_text()
        return self.text or ""


class Page(BaseModel):
    number: int
    width: float
    height: float
    elements: list[Element] = Field(default_factory=list)


class Document(BaseModel):
    id: str
    source_name: str
    mime: str
    checksum: str
    title: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    pages: list[Page] = Field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def add_warning(self, element_id: str | None, code: str, detail: str) -> None:
        """Record a per-element degradation. Content is never silently dropped."""
        self.meta.setdefault("warnings", []).append(
            {"element_id": element_id, "code": code, "detail": detail}
        )

    def element(self, element_id: str) -> Element:
        from app.ir.tree import element_index

        return element_index(self)[element_id]

    def iter_elements(self) -> Iterator[Element]:
        from app.ir.tree import iter_elements

        return iter_elements(self)

    def section_tree(self):
        from app.ir.tree import section_tree

        return section_tree(self)
