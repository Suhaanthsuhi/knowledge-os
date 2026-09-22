"""Colours and small drawing helpers shared by the viewer.

Kept free of Streamlit so the rendering layer stays unit-testable.
"""

from __future__ import annotations

from app.ir.model import ElementType

__all__ = [
    "COLORS",
    "SYNTHETIC_BG",
    "rgb",
    "rgba",
    "color_for",
    "TYPE_ORDER",
    "CSS",
]

# One hue per element type. Chosen to stay legible drawn over a white page and
# to keep the margin furniture (header/footer/page number) visually quiet.
COLORS: dict[ElementType, str] = {
    ElementType.HEADING: "#4F46E5",
    ElementType.PARAGRAPH: "#475569",
    ElementType.LIST_ITEM: "#0D9488",
    ElementType.TABLE: "#D97706",
    ElementType.IMAGE: "#7C3AED",
    ElementType.CAPTION: "#DB2777",
    ElementType.CODE: "#0891B2",
    ElementType.HEADER: "#94A3B8",
    ElementType.FOOTER: "#94A3B8",
    ElementType.PAGE_NUMBER: "#94A3B8",
}

FALLBACK = "#64748B"
SYNTHETIC_BG = (250, 250, 252)

# Reading order, used for legends and filters.
TYPE_ORDER: tuple[ElementType, ...] = (
    ElementType.HEADING,
    ElementType.PARAGRAPH,
    ElementType.LIST_ITEM,
    ElementType.CODE,
    ElementType.TABLE,
    ElementType.IMAGE,
    ElementType.CAPTION,
    ElementType.HEADER,
    ElementType.FOOTER,
    ElementType.PAGE_NUMBER,
)


def rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    red, green, blue = rgb(value)
    return (red, green, blue, alpha)


def color_for(element_type: ElementType) -> str:
    return COLORS.get(element_type, FALLBACK)


CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  [data-testid="stMetricValue"] { font-size: 1.45rem; font-variant-numeric: tabular-nums; }
  [data-testid="stMetricLabel"] { opacity: 0.68; font-size: 0.78rem;
      text-transform: uppercase; letter-spacing: 0.06em; }
  .kos-legend { display: flex; flex-wrap: wrap; gap: 0.4rem 0.9rem; margin: 0.2rem 0 0.9rem; }
  .kos-chip { display: inline-flex; align-items: center; gap: 0.42rem;
      font-size: 0.76rem; opacity: 0.85; }
  .kos-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
  .kos-card { border: 1px solid rgba(128,128,128,0.22); border-radius: 10px;
      padding: 0.85rem 1rem; margin-bottom: 0.6rem; }
  .kos-kv { display: grid; grid-template-columns: 8.5rem 1fr; gap: 0.3rem 0.8rem;
      font-size: 0.86rem; }
  .kos-kv dt { opacity: 0.6; }
  .kos-kv dd { margin: 0; font-variant-numeric: tabular-nums; }
  .kos-mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.82rem; }
  .kos-quote { border-left: 3px solid rgba(128,128,128,0.35); padding-left: 0.7rem;
      white-space: pre-wrap; font-size: 0.86rem; max-height: 14rem; overflow-y: auto; }
</style>
"""
