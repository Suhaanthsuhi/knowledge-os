from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.ir.model import BBox, ElementType

__all__ = [
    "LayoutConfig",
    "Block",
    "ZONE_RANK",
    "body_font_size",
    "heading_levels",
    "find_gutters",
    "column_index",
    "reading_order",
    "margin_blocks",
    "pair_captions",
]

CAPTION_PATTERN = re.compile(r"^(figure|fig\.|table|chart)\s*\d+", re.IGNORECASE)
PAGE_NUMBER_PATTERN = re.compile(
    r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$|^[ivxlcdm]+$", re.IGNORECASE
)
MAX_HEADING_LEVEL = 6


@dataclass(frozen=True)
class LayoutConfig:
    """Every layout threshold in one place, so they are tunable and visible."""

    heading_size_ratio: float = 1.15
    heading_max_chars: int = 80
    heading_gap_ratio: float = 0.5
    body_line_height: float = 12.0
    gutter_min_width: float = 18.0
    gutter_min_coverage: float = 0.60
    margin_band_ratio: float = 0.08
    repeat_page_ratio: float = 0.50
    caption_max_distance: float = 40.0
    caption_max_chars: int = 200
    caption_min_overlap: float = 0.50


@dataclass
class Block:
    """A positioned run of content, independent of any PDF library."""

    index: int
    page_number: int
    bbox: BBox
    text: str
    max_size: float = 0.0
    is_bold: bool = False
    line_count: int = 1
    gap_below: float = 0.0
    kind: str = "text"  # "text" | "table" | "image"
    attrs: dict[str, Any] = field(default_factory=dict)


ZONE_RANK: dict[ElementType, int] = {
    ElementType.HEADER: 0,
    ElementType.FOOTER: 2,
    ElementType.PAGE_NUMBER: 2,
}


def _text_blocks(blocks: list[Block]) -> list[Block]:
    return [b for b in blocks if b.kind == "text" and b.max_size > 0]


def body_font_size(blocks: list[Block], cfg: LayoutConfig) -> float:
    """Character-weighted median font size across the whole document body.

    Document-wide rather than per page: a title page's few huge glyphs would
    otherwise drag a per-page median upward and suppress real headings.
    """
    weighted: list[tuple[float, int]] = [
        (b.max_size, max(len(b.text.strip()), 1)) for b in _text_blocks(blocks)
    ]
    if not weighted:
        return 0.0

    weighted.sort(key=lambda pair: pair[0])
    total = sum(weight for _, weight in weighted)
    seen = 0
    for size, weight in weighted:
        seen += weight
        if seen * 2 >= total:
            return size
    return weighted[-1][0]


def _is_heading_candidate(block: Block, body_size: float, cfg: LayoutConfig) -> bool:
    if block.kind != "text" or not block.text.strip():
        return False
    if body_size > 0 and block.max_size >= body_size * cfg.heading_size_ratio:
        return True
    return (
        block.is_bold
        and block.line_count == 1
        and len(block.text.strip()) <= cfg.heading_max_chars
        and block.gap_below >= cfg.body_line_height * cfg.heading_gap_ratio
    )


def heading_levels(
    blocks: list[Block], body_size: float, cfg: LayoutConfig
) -> dict[int, int]:
    """Map block index -> heading level, consistent across the whole document."""
    candidates = [b for b in blocks if _is_heading_candidate(b, body_size, cfg)]
    if not candidates:
        return {}

    def bucket(size: float) -> float:
        return round(size * 2) / 2

    ranked = sorted({bucket(b.max_size) for b in candidates}, reverse=True)
    level_of = {size: min(rank + 1, MAX_HEADING_LEVEL) for rank, size in enumerate(ranked)}
    return {b.index: level_of[bucket(b.max_size)] for b in candidates}


def _span(blocks: list[Block]) -> tuple[float, float]:
    """The vertical extent covered by a set of blocks."""
    return min(b.bbox.y0 for b in blocks), max(b.bbox.y1 for b in blocks)


def _columns_run_alongside(
    left: list[Block], right: list[Block], cfg: LayoutConfig
) -> bool:
    """True when the two sides are genuine parallel columns.

    Measured as vertical overlap against the shorter side, not against the
    whole page's text height: a section heading or title sitting above the
    columns belongs to one side only and would otherwise sink the ratio.
    Requiring two blocks a side rejects a lone stray element off in the margin.
    """
    if len(left) < 2 or len(right) < 2:
        return False

    left_top, left_bottom = _span(left)
    right_top, right_bottom = _span(right)
    overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
    shorter = min(left_bottom - left_top, right_bottom - right_top)
    if shorter <= 0:
        return False
    return overlap >= cfg.gutter_min_coverage * shorter


def find_gutters(blocks: list[Block], cfg: LayoutConfig) -> list[float]:
    """Centres of vertical whitespace bands that separate text columns."""
    positioned = [b for b in blocks if b.bbox.width > 0]
    if len(positioned) < 4:
        return []

    top, bottom = _span(positioned)
    if bottom - top <= 0:
        return []

    x_min = min(b.bbox.x0 for b in positioned)
    x_max = max(b.bbox.x1 for b in positioned)
    bin_count = int(math.ceil(x_max - x_min)) + 1
    occupied = [False] * bin_count
    for b in positioned:
        start = max(0, int(b.bbox.x0 - x_min))
        end = min(bin_count, int(math.ceil(b.bbox.x1 - x_min)))
        for i in range(start, end):
            occupied[i] = True

    gutters: list[float] = []
    i = 0
    while i < bin_count:
        if occupied[i]:
            i += 1
            continue
        j = i
        while j < bin_count and not occupied[j]:
            j += 1
        if (j - i) >= cfg.gutter_min_width and i > 0 and j < bin_count:
            centre = x_min + (i + j) / 2
            left = [b for b in positioned if b.bbox.x1 <= centre]
            right = [b for b in positioned if b.bbox.x0 >= centre]
            if _columns_run_alongside(left, right, cfg):
                gutters.append(round(centre, 2))
        i = j

    return gutters


def column_index(bbox: BBox, gutters: list[float]) -> int:
    return sum(1 for gutter in gutters if bbox.x0 >= gutter)


def reading_order(
    blocks: list[Block],
    gutters: list[float],
    zones: dict[int, ElementType] | None = None,
) -> list[Block]:
    """Column-aware ordering: down each column in turn, else top to bottom.

    `zones` maps block index -> margin element type. Running headers sort ahead
    of the body and footers behind it, so margin furniture keeps a natural
    position without polluting the body's column order.
    """
    zones = zones or {}
    return sorted(
        blocks,
        key=lambda b: (
            b.page_number,
            ZONE_RANK.get(zones.get(b.index), 1),
            column_index(b.bbox, gutters),
            b.bbox.y0,
            b.bbox.x0,
        ),
    )


def _normalize(text: str) -> str:
    collapsed = " ".join(text.split()).lower()
    return re.sub(r"\d+", "#", collapsed)


def margin_blocks(
    blocks: list[Block],
    page_height: float,
    page_count: int,
    cfg: LayoutConfig,
) -> dict[int, ElementType]:
    """Identify running headers, footers and page numbers.

    A block qualifies by sitting wholly inside the top or bottom margin band
    AND either matching a page-number pattern or repeating across pages.
    """
    band = page_height * cfg.margin_band_ratio
    threshold = max(2, math.ceil(page_count * cfg.repeat_page_ratio))

    in_band: list[tuple[Block, ElementType]] = []
    for block in blocks:
        if block.kind != "text" or not block.text.strip():
            continue
        if block.bbox.y1 <= band:
            in_band.append((block, ElementType.HEADER))
        elif block.bbox.y0 >= page_height - band:
            in_band.append((block, ElementType.FOOTER))

    counts = Counter(_normalize(block.text) for block, _ in in_band)

    result: dict[int, ElementType] = {}
    for block, zone in in_band:
        stripped = block.text.strip()
        if PAGE_NUMBER_PATTERN.fullmatch(stripped):
            result[block.index] = ElementType.PAGE_NUMBER
        elif counts[_normalize(block.text)] >= threshold:
            result[block.index] = zone
    return result


def pair_captions(
    figures: list[Block],
    text_blocks: list[Block],
    body_size: float,
    cfg: LayoutConfig,
) -> dict[int, int]:
    """Map figure block index -> caption block index.

    Searched below first, then above. A `Figure 3: ...` style label always
    outranks a merely-smaller block; ties break toward the closer block. Each
    caption is claimed by at most one figure.
    """
    claimed: set[int] = set()
    paired: dict[int, int] = {}

    for figure in sorted(figures, key=lambda f: (f.page_number, f.bbox.y0)):
        best_key: tuple[int, float, int] | None = None
        best_index: int | None = None

        for below in (True, False):
            for candidate in text_blocks:
                if candidate.page_number != figure.page_number:
                    continue
                if candidate.index in claimed or candidate.kind != "text":
                    continue
                stripped = candidate.text.strip()
                if not stripped or len(stripped) > cfg.caption_max_chars:
                    continue
                if (
                    candidate.bbox.horizontal_overlap_ratio(figure.bbox)
                    < cfg.caption_min_overlap
                ):
                    continue

                if below:
                    distance = candidate.bbox.y0 - figure.bbox.y1
                else:
                    distance = figure.bbox.y0 - candidate.bbox.y1
                if distance < 0 or distance > cfg.caption_max_distance:
                    continue

                is_labelled = bool(CAPTION_PATTERN.match(stripped))
                is_smaller = 0 < candidate.max_size < body_size
                if not (is_labelled or is_smaller):
                    continue

                key = (0 if is_labelled else 1, distance, candidate.index)
                if best_key is None or key < best_key:
                    best_key, best_index = key, candidate.index

            if best_index is not None:
                break

        if best_index is not None:
            paired[figure.index] = best_index
            claimed.add(best_index)

    return paired
