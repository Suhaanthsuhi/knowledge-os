from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import fitz  # PyMuPDF
import pdfplumber

from app.ir.ids import blob_ref, element_id
from app.ir.model import BBox, Document, Element, ElementType, ImageRef, Page, TableData
from app.ir.tree import assign_parents
from app.parsing.base import BlobSink, ParseError, source_identity
from app.parsing.layout import (
    Block,
    LayoutConfig,
    body_font_size,
    find_gutters,
    heading_levels,
    margin_blocks,
    pair_captions,
    reading_order,
)

__all__ = ["PdfParser"]

BOLD_FLAG = 1 << 4


@dataclass
class _TableRegion:
    bbox: BBox
    table: TableData


class _Counter:
    """Hands out document-wide unique block indices."""

    def __init__(self) -> None:
        self._value = -1

    def next(self) -> int:
        self._value += 1
        return self._value


def _to_table_data(rows: list[list[Any]]) -> TableData | None:
    cleaned = [
        [("" if cell is None else str(cell)).strip() for cell in row] for row in rows
    ]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if not cleaned:
        return None

    n_cols = max(len(row) for row in cleaned)
    padded = [row + [""] * (n_cols - len(row)) for row in cleaned]

    headers: list[str] = []
    if len(padded) >= 2 and all(cell for cell in padded[0]):
        headers, padded = padded[0], padded[1:]

    return TableData(headers=headers, rows=padded, n_rows=len(padded), n_cols=n_cols)


class PdfParser:
    """PyMuPDF for text, geometry and images; pdfplumber for table structure.

    Both libraries report coordinates with a top-left origin and y increasing
    downward, so their boxes are directly comparable for unrotated pages. A
    rotated page is parsed anyway but records a warning.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".pdf"})
    version: ClassVar[str] = "1"

    def __init__(self, cfg: LayoutConfig | None = None) -> None:
        self.cfg = cfg or LayoutConfig()

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        path = Path(path)
        short_id, checksum = source_identity(path)
        document = Document(
            id=short_id,
            source_name=path.name,
            mime="application/pdf",
            checksum=checksum,
            meta={"parser": "pdf", "parser_version": self.version},
        )

        try:
            pdf = fitz.open(str(path))
        except Exception as exc:
            raise ParseError(f"cannot open {path} as a PDF: {exc}") from exc

        regions = self._find_tables(path, document)
        counter = _Counter()
        blocks_by_page: dict[int, list[Block]] = {}
        geometry: dict[int, tuple[float, float]] = {}

        try:
            with pdf:
                if pdf.page_count == 0:
                    raise ParseError(f"{path} contains no pages")
                for number, pdf_page in enumerate(pdf, start=1):
                    if pdf_page.rotation:
                        document.add_warning(
                            None,
                            "rotated_page",
                            f"page {number} rotation={pdf_page.rotation}",
                        )
                    geometry[number] = (
                        float(pdf_page.rect.width),
                        float(pdf_page.rect.height),
                    )
                    page_regions = regions.get(number, [])
                    page_blocks = self._text_blocks(
                        pdf_page, number, page_regions, counter
                    )
                    page_blocks += self._table_blocks(number, page_regions, counter)
                    page_blocks += self._image_blocks(
                        pdf, pdf_page, number, counter, blobs, document
                    )
                    blocks_by_page[number] = page_blocks
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"failed to parse {path}: {exc}") from exc

        self._assemble(document, blocks_by_page, geometry)
        return document

    # ---------- extraction ----------

    def _find_tables(self, path: Path, document: Document) -> dict[int, list[_TableRegion]]:
        regions: dict[int, list[_TableRegion]] = {}
        try:
            plumber = pdfplumber.open(str(path))
        except Exception as exc:
            document.add_warning(None, "table_scan_failed", str(exc))
            return regions

        with plumber:
            for number, page in enumerate(plumber.pages, start=1):
                found: list[_TableRegion] = []
                try:
                    candidates = page.find_tables()
                except Exception as exc:
                    document.add_warning(None, "table_scan_failed", f"page {number}: {exc}")
                    continue

                for candidate in candidates:
                    try:
                        rows = candidate.extract()
                    except Exception as exc:
                        # Not masked, so the text survives as ordinary paragraphs.
                        document.add_warning(
                            None, "table_fallback", f"page {number}: {exc}"
                        )
                        continue
                    table = _to_table_data(rows)
                    if table is None:
                        document.add_warning(
                            None, "table_fallback", f"page {number}: no extractable cells"
                        )
                        continue
                    x0, top, x1, bottom = candidate.bbox
                    found.append(
                        _TableRegion(bbox=BBox(x0=x0, y0=top, x1=x1, y1=bottom), table=table)
                    )
                if found:
                    regions[number] = found
        return regions

    def _text_blocks(
        self,
        pdf_page: Any,
        page_number: int,
        regions: list[_TableRegion],
        counter: _Counter,
    ) -> list[Block]:
        collected: list[Block] = []
        for raw in pdf_page.get_text("dict").get("blocks", []):
            if raw.get("type") != 0:
                continue
            lines = raw.get("lines") or []
            spans = [span for line in lines for span in line.get("spans", [])]
            if not spans:
                continue
            text = "\n".join(
                "".join(span["text"] for span in line.get("spans", [])) for line in lines
            ).strip()
            if not text:
                continue

            x0, y0, x1, y1 = raw["bbox"]
            bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
            centre_x, centre_y = bbox.center
            # Masked: this text belongs to a table that was extracted structurally.
            if any(region.bbox.contains_point(centre_x, centre_y) for region in regions):
                continue

            collected.append(
                Block(
                    index=counter.next(),
                    page_number=page_number,
                    bbox=bbox,
                    text=text,
                    max_size=max(float(span["size"]) for span in spans),
                    is_bold=any(int(span.get("flags", 0)) & BOLD_FLAG for span in spans),
                    line_count=len(lines),
                    kind="text",
                )
            )

        collected.sort(key=lambda block: (block.bbox.y0, block.bbox.x0))
        for current, following in zip(collected, collected[1:]):
            current.gap_below = round(max(0.0, following.bbox.y0 - current.bbox.y1), 2)
        return collected

    def _table_blocks(
        self, page_number: int, regions: list[_TableRegion], counter: _Counter
    ) -> list[Block]:
        return [
            Block(
                index=counter.next(),
                page_number=page_number,
                bbox=region.bbox,
                text=region.table.as_text(),
                kind="table",
                attrs={"table": region.table},
            )
            for region in regions
        ]

    def _image_blocks(
        self,
        pdf: Any,
        pdf_page: Any,
        page_number: int,
        counter: _Counter,
        blobs: BlobSink,
        document: Document,
    ) -> list[Block]:
        collected: list[Block] = []
        for position, info in enumerate(pdf_page.get_images(full=True)):
            xref = info[0]
            rects = pdf_page.get_image_rects(xref)
            if not rects:
                document.add_warning(
                    None, "image_not_placed", f"xref {xref} on page {page_number}"
                )
                continue
            rect = rects[0]
            bbox = BBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)

            try:
                extracted = pdf.extract_image(xref)
                ext = str(extracted.get("ext") or "png")
                ref = blob_ref(page_number, position, ext)
                blobs.add(ref, extracted["image"])
                image = ImageRef(
                    blob_ref=ref,
                    width=extracted.get("width"),
                    height=extracted.get("height"),
                    format=ext,
                )
            except Exception as exc:
                # Degrade: keep the element and its geometry, lose only the bytes.
                document.add_warning(
                    None,
                    "image_decode_failed",
                    f"xref {xref} on page {page_number}: {exc}",
                )
                image = ImageRef()

            collected.append(
                Block(
                    index=counter.next(),
                    page_number=page_number,
                    bbox=bbox,
                    text="",
                    kind="image",
                    attrs={"image": image},
                )
            )
        return collected

    # ---------- assembly ----------

    def _assemble(
        self,
        document: Document,
        blocks_by_page: dict[int, list[Block]],
        geometry: dict[int, tuple[float, float]],
    ) -> None:
        all_blocks = [block for blocks in blocks_by_page.values() for block in blocks]
        body_size = body_font_size(all_blocks, self.cfg)
        heading_map = heading_levels(all_blocks, body_size, self.cfg)

        page_height = geometry[min(geometry)][1] if geometry else 792.0
        margin_map = margin_blocks(all_blocks, page_height, len(blocks_by_page), self.cfg)

        figures = [block for block in all_blocks if block.kind in {"image", "table"}]
        caption_candidates = [
            block
            for block in all_blocks
            if block.kind == "text" and block.index not in margin_map
        ]
        caption_map = pair_captions(figures, caption_candidates, body_size, self.cfg)
        caption_owner = {caption: figure for figure, caption in caption_map.items()}

        order = 0
        element_by_block: dict[int, Element] = {}

        for page_number in sorted(blocks_by_page):
            width, height = geometry[page_number]
            page_blocks = blocks_by_page[page_number]
            body_blocks = [
                block
                for block in page_blocks
                if block.index not in margin_map and block.kind == "text"
            ]
            gutters = find_gutters(body_blocks, self.cfg)
            ordered = reading_order(page_blocks, gutters, margin_map)

            elements: list[Element] = []
            for position, block in enumerate(ordered):
                element = self._to_element(
                    block,
                    page_number,
                    position,
                    order,
                    heading_map,
                    margin_map,
                    caption_owner,
                )
                elements.append(element)
                element_by_block[block.index] = element
                order += 1

            document.pages.append(
                Page(number=page_number, width=width, height=height, elements=elements)
            )

        for figure_index, caption_index in caption_map.items():
            figure = element_by_block[figure_index]
            caption = element_by_block[caption_index]
            if figure.image is not None:
                figure.image.caption_id = caption.id
            else:
                figure.attrs["caption_id"] = caption.id
            caption.attrs["captions"] = figure.id

        assign_parents([e for page in document.pages for e in page.elements])

        document.meta["page_count"] = len(document.pages)
        document.title = next(
            (
                element.text
                for element in document.iter_elements()
                if element.type is ElementType.HEADING
                and element.level == 1
                and element.text
            ),
            None,
        )

    def _to_element(
        self,
        block: Block,
        page_number: int,
        position: int,
        order: int,
        heading_map: dict[int, int],
        margin_map: dict[int, ElementType],
        caption_owner: dict[int, int],
    ) -> Element:
        level: int | None = None
        attrs: dict[str, Any] = {}

        if block.index in margin_map:
            element_type = margin_map[block.index]
        elif block.kind == "table":
            element_type = ElementType.TABLE
        elif block.kind == "image":
            element_type = ElementType.IMAGE
        elif block.index in caption_owner:
            element_type = ElementType.CAPTION
        elif block.index in heading_map:
            element_type = ElementType.HEADING
            level = heading_map[block.index]
        else:
            element_type = ElementType.PARAGRAPH

        if block.max_size:
            attrs["font_size"] = round(block.max_size, 2)
        if block.is_bold:
            attrs["bold"] = True

        return Element(
            id=element_id(page_number, position),
            type=element_type,
            page_number=page_number,
            bbox=block.bbox,
            order=order,
            level=level,
            text=None if block.kind in {"table", "image"} else block.text,
            table=block.attrs.get("table"),
            image=block.attrs.get("image"),
            attrs=attrs,
        )
