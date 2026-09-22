from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from markdown_it import MarkdownIt
from markdown_it.token import Token

from app.ir.ids import element_id
from app.ir.model import BBox, Document, Element, ElementType, Page, TableData
from app.ir.tree import assign_parents
from app.parsing.base import BlobSink, ParseError, source_identity

__all__ = ["MarkdownParser", "LINE_HEIGHT", "PAGE_WIDTH", "MARGIN"]

LINE_HEIGHT = 14.0
PAGE_WIDTH = 612.0
MARGIN = 72.0


class MarkdownParser:
    """Markdown token stream to IR.

    Markdown carries no geometry, so bboxes are synthetic: one continuous page
    whose y coordinates derive from source line numbers. Every element is
    tagged `attrs["synthetic_bbox"] = True` so downstream code can tell
    "position unknown" from a real rectangle.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".md", ".markdown"})
    version: ClassVar[str] = "1"

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        path = Path(path)
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(f"{path} is not valid UTF-8: {exc}") from exc

        short_id, checksum = source_identity(path)
        # gfm-like brings tables and strikethrough. linkify is switched off: it
        # needs an extra dependency and autolinking adds nothing to the IR.
        tokens = MarkdownIt("gfm-like", {"linkify": False}).parse(text)

        builder = _ElementBuilder()
        builder.walk(tokens)
        elements = builder.elements

        assign_parents(elements)

        height = max((e.bbox.y1 for e in elements), default=MARGIN) + MARGIN
        page = Page(
            number=1, width=PAGE_WIDTH, height=round(height, 2), elements=elements
        )

        title = next(
            (
                e.text
                for e in elements
                if e.type is ElementType.HEADING and e.level == 1 and e.text
            ),
            None,
        )

        return Document(
            id=short_id,
            source_name=path.name,
            mime="text/markdown",
            checksum=checksum,
            title=title,
            meta={
                "page_count": 1,
                "parser": "markdown",
                "parser_version": self.version,
                "line_count": text.count("\n") + 1,
            },
            pages=[page],
        )


def _bbox_for(token_map: list[int] | None, fallback_line: int) -> BBox:
    """Synthetic geometry: y derives from source lines, x spans the text column."""
    if token_map:
        start, end = token_map[0], max(token_map[1], token_map[0] + 1)
    else:
        start, end = fallback_line, fallback_line + 1
    return BBox(
        x0=MARGIN,
        y0=MARGIN + start * LINE_HEIGHT,
        x1=PAGE_WIDTH - MARGIN,
        y1=MARGIN + end * LINE_HEIGHT,
    )


class _ElementBuilder:
    """Turns a markdown-it token stream into a flat, ordered element list."""

    def __init__(self) -> None:
        self.elements: list[Element] = []
        self._order = 0
        self._last_line = 0
        self._list_depth = 0

    def _emit(
        self,
        etype: ElementType,
        token_map: list[int] | None,
        *,
        text: str | None = None,
        level: int | None = None,
        table: TableData | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        bbox = _bbox_for(token_map, self._last_line)
        if token_map:
            self._last_line = max(token_map[1], token_map[0] + 1)
        else:
            self._last_line += 1

        element_attrs: dict[str, Any] = {"synthetic_bbox": True}
        if attrs:
            element_attrs.update(attrs)

        self.elements.append(
            Element(
                id=element_id(1, self._order),
                type=etype,
                page_number=1,
                bbox=bbox,
                order=self._order,
                text=text,
                level=level,
                table=table,
                attrs=element_attrs,
            )
        )
        self._order += 1

    def walk(self, tokens: list[Token]) -> None:
        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                inline = tokens[i + 1]
                self._emit(
                    ElementType.HEADING,
                    token.map,
                    text=inline.content.strip(),
                    level=int(token.tag[1:]),
                )
                i += 3
                continue

            if token.type in {"bullet_list_open", "ordered_list_open"}:
                self._list_depth += 1
                i += 1
                continue

            if token.type in {"bullet_list_close", "ordered_list_close"}:
                self._list_depth -= 1
                i += 1
                continue

            if token.type == "paragraph_open":
                inline = tokens[i + 1]
                etype = (
                    ElementType.LIST_ITEM
                    if self._list_depth > 0
                    else ElementType.PARAGRAPH
                )
                self._emit(etype, token.map, text=inline.content.strip())
                i += 3
                continue

            if token.type in {"fence", "code_block"}:
                self._emit(
                    ElementType.CODE,
                    token.map,
                    text=token.content,
                    attrs={"language": (token.info or "").strip() or None},
                )
                i += 1
                continue

            if token.type == "table_open":
                end = _matching_close(tokens, i, "table_open", "table_close")
                table = _build_table(tokens[i : end + 1])
                self._emit(ElementType.TABLE, token.map, table=table)
                i = end + 1
                continue

            i += 1


def _matching_close(
    tokens: list[Token], start: int, open_type: str, close_type: str
) -> int:
    depth = 0
    for index in range(start, len(tokens)):
        if tokens[index].type == open_type:
            depth += 1
        elif tokens[index].type == close_type:
            depth -= 1
            if depth == 0:
                return index
    raise ParseError(f"unbalanced {open_type} in markdown token stream")


def _build_table(tokens: list[Token]) -> TableData:
    headers: list[str] = []
    rows: list[list[str]] = []
    current: list[str] = []
    in_header = False

    for token in tokens:
        if token.type == "thead_open":
            in_header = True
        elif token.type == "thead_close":
            in_header = False
        elif token.type == "tr_open":
            current = []
        elif token.type == "tr_close":
            if in_header:
                headers = current
            else:
                rows.append(current)
        elif token.type == "inline":
            current.append(token.content.strip())

    n_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
    normalized = [row + [""] * (n_cols - len(row)) for row in rows]
    return TableData(
        headers=headers, rows=normalized, n_rows=len(normalized), n_cols=n_cols
    )
