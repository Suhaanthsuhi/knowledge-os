from __future__ import annotations

from app.ir.model import BBox, ElementType
from app.parsing.layout import (
    Block,
    LayoutConfig,
    body_font_size,
    column_index,
    find_gutters,
    heading_levels,
    margin_blocks,
    pair_captions,
    reading_order,
)

CFG = LayoutConfig()


def block(
    index: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str = "body text here",
    *,
    size: float = 11.0,
    bold: bool = False,
    lines: int = 1,
    gap_below: float = 0.0,
    page: int = 1,
    kind: str = "text",
) -> Block:
    return Block(
        index=index,
        page_number=page,
        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
        text=text,
        max_size=size,
        is_bold=bold,
        line_count=lines,
        gap_below=gap_below,
        kind=kind,
    )


def test_body_font_size_is_the_char_weighted_median():
    blocks = [
        block(0, 72, 100, 300, 120, "Title", size=18.0, bold=True),
        block(1, 72, 130, 500, 200, "x" * 400, size=11.0),
        block(2, 72, 210, 500, 280, "y" * 400, size=11.0),
    ]
    assert body_font_size(blocks, CFG) == 11.0


def test_body_font_size_ignores_non_text_blocks():
    blocks = [
        block(0, 72, 100, 500, 200, "x" * 100, size=10.0),
        block(1, 72, 210, 500, 400, "", size=0.0, kind="image"),
    ]
    assert body_font_size(blocks, CFG) == 10.0


def test_larger_text_becomes_headings_ranked_by_size():
    blocks = [
        block(0, 72, 100, 300, 120, "Payment Architecture", size=18.0, bold=True),
        block(1, 72, 130, 500, 200, "x" * 200, size=11.0),
        block(2, 72, 210, 300, 228, "Dependencies", size=14.0, bold=True),
        block(3, 72, 240, 500, 320, "y" * 200, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {0: 1, 2: 2}


def test_bold_short_line_with_a_gap_is_a_heading_at_body_size():
    blocks = [
        block(0, 72, 100, 200, 114, "Overview", size=11.0, bold=True, lines=1, gap_below=12.0),
        block(1, 72, 130, 500, 300, "x" * 400, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {0: 1}


def test_bold_but_long_paragraph_is_not_a_heading():
    blocks = [
        block(0, 72, 100, 500, 200, "b" * 300, size=11.0, bold=True, lines=6, gap_below=12.0),
        block(1, 72, 210, 500, 300, "x" * 400, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {}


def test_heading_sizes_are_clustered_to_half_points():
    blocks = [
        block(0, 72, 100, 300, 120, "A", size=18.02, bold=True),
        block(1, 72, 130, 300, 150, "B", size=17.98, bold=True),
        block(2, 72, 160, 500, 400, "x" * 400, size=11.0),
    ]
    assert heading_levels(blocks, 11.0, CFG) == {0: 1, 1: 1}


def test_find_gutters_detects_a_two_column_layout():
    left = [block(i, 72, 100 + i * 40, 272, 130 + i * 40, "x" * 80) for i in range(8)]
    right = [block(8 + i, 340, 100 + i * 40, 540, 130 + i * 40, "y" * 80) for i in range(8)]
    gutters = find_gutters(left + right, CFG)

    assert len(gutters) == 1
    assert 272 < gutters[0] < 340


def test_find_gutters_returns_nothing_for_a_single_column():
    blocks = [block(i, 72, 100 + i * 40, 540, 130 + i * 40, "x" * 80) for i in range(8)]
    assert find_gutters(blocks, CFG) == []


def test_column_index_counts_gutters_to_the_left():
    assert column_index(BBox(x0=72, y0=0, x1=272, y1=10), [306.0]) == 0
    assert column_index(BBox(x0=340, y0=0, x1=540, y1=10), [306.0]) == 1


def test_reading_order_reads_down_each_column_in_turn():
    blocks = [
        block(0, 72, 100, 272, 130, "left top"),
        block(1, 340, 100, 540, 130, "right top"),
        block(2, 72, 200, 272, 230, "left bottom"),
        block(3, 340, 200, 540, 230, "right bottom"),
    ]
    ordered = reading_order(blocks, [306.0])
    assert [b.text for b in ordered] == [
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    ]


def test_reading_order_falls_back_to_top_to_bottom_without_gutters():
    blocks = [
        block(0, 300, 200, 500, 230, "second"),
        block(1, 72, 100, 272, 130, "first"),
        block(2, 72, 200, 272, 230, "middle"),
    ]
    assert [b.text for b in reading_order(blocks, [])] == ["first", "middle", "second"]


def test_reading_order_puts_headers_first_and_footers_last():
    blocks = [
        block(0, 72, 300, 540, 340, "body"),
        block(1, 72, 30, 540, 50, "running header"),
        block(2, 72, 750, 540, 765, "running footer"),
    ]
    zones = {1: ElementType.HEADER, 2: ElementType.FOOTER}
    ordered = reading_order(blocks, [], zones)
    assert [b.text for b in ordered] == ["running header", "body", "running footer"]


def test_repeated_bottom_margin_text_becomes_a_footer():
    blocks = [
        block(i, 72, 740, 540, 760, "Knowledge OS Internal", size=8.0, page=i + 1)
        for i in range(3)
    ]
    blocks.append(block(99, 72, 300, 540, 400, "real body text", size=11.0))

    result = margin_blocks(blocks, page_height=792, page_count=3, cfg=CFG)
    assert result == {0: ElementType.FOOTER, 1: ElementType.FOOTER, 2: ElementType.FOOTER}


def test_repeated_top_margin_text_becomes_a_header():
    blocks = [
        block(i, 72, 30, 540, 50, "Payment Platform Handbook", size=9.0, page=i + 1)
        for i in range(4)
    ]
    result = margin_blocks(blocks, page_height=792, page_count=4, cfg=CFG)
    assert set(result.values()) == {ElementType.HEADER}


def test_bare_numbers_in_the_margin_become_page_numbers():
    blocks = [
        block(0, 300, 750, 320, 765, "1", size=9.0, page=1),
        block(1, 300, 750, 320, 765, "2", size=9.0, page=2),
    ]
    result = margin_blocks(blocks, page_height=792, page_count=2, cfg=CFG)
    assert result == {0: ElementType.PAGE_NUMBER, 1: ElementType.PAGE_NUMBER}


def test_body_text_in_the_margin_band_is_left_alone_when_it_does_not_repeat():
    blocks = [
        block(0, 72, 740, 540, 760, "a unique closing sentence", size=11.0, page=1),
        block(1, 72, 740, 540, 760, "a different closing sentence", size=11.0, page=2),
    ]
    assert margin_blocks(blocks, page_height=792, page_count=2, cfg=CFG) == {}


def test_caption_below_a_figure_is_paired():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    caption = block(1, 100, 410, 400, 424, "Figure 1: Payment architecture", size=9.0)
    body = block(2, 72, 500, 540, 600, "x" * 200, size=11.0)

    assert pair_captions([figure], [caption, body], 11.0, CFG) == {0: 1}


def test_caption_above_is_used_when_nothing_sits_below():
    figure = block(0, 100, 300, 400, 500, "", size=0.0, kind="image")
    caption = block(1, 100, 275, 400, 292, "Table 2: Service owners", size=9.0)

    assert pair_captions([figure], [caption], 11.0, CFG) == {0: 1}


def test_a_distant_block_is_not_treated_as_a_caption():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    far = block(1, 100, 600, 400, 620, "Figure 9: unrelated", size=9.0)

    assert pair_captions([figure], [far], 11.0, CFG) == {}


def test_a_block_that_barely_overlaps_the_figure_is_not_a_caption():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    aside = block(1, 420, 410, 540, 424, "Figure 4: sidebar", size=9.0)

    assert pair_captions([figure], [aside], 11.0, CFG) == {}


def test_a_caption_is_claimed_by_only_one_figure():
    upper = block(0, 100, 100, 400, 300, "", size=0.0, kind="image")
    lower = block(1, 100, 340, 400, 500, "", size=0.0, kind="image")
    caption = block(2, 100, 310, 400, 326, "Figure 1: shared", size=9.0)

    paired = pair_captions([upper, lower], [caption], 11.0, CFG)
    assert list(paired.values()) == [2]
    assert len(paired) == 1


def test_pattern_match_outranks_a_merely_smaller_block():
    figure = block(0, 100, 200, 400, 400, "", size=0.0, kind="image")
    small = block(1, 100, 405, 400, 415, "some small note", size=9.0)
    labelled = block(2, 100, 420, 400, 434, "Figure 7: the real caption", size=9.0)

    assert pair_captions([figure], [small, labelled], 11.0, CFG) == {0: 2}


def test_a_heading_above_the_columns_does_not_defeat_gutter_detection():
    """The real two-column shape: a section heading spanning only the left."""
    heading = block(0, 72, 60, 200, 80, "Section 1", size=16.0, bold=True)
    left = [block(1 + i, 72, 110 + i * 56, 272, 150 + i * 56, "x" * 60) for i in range(3)]
    right = [block(4 + i, 340, 110 + i * 56, 540, 150 + i * 56, "y" * 60) for i in range(3)]

    gutters = find_gutters([heading, *left, *right], CFG)
    assert len(gutters) == 1
    assert 272 < gutters[0] < 340


def test_one_stray_block_off_to_the_side_is_not_a_column():
    body = [block(i, 72, 100 + i * 40, 272, 130 + i * 40, "x" * 80) for i in range(6)]
    stray = block(99, 460, 100, 540, 118, "v2.1")

    assert find_gutters([*body, stray], CFG) == []
