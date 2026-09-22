"""Streamlit panels. Presentation only - no parsing logic lives here."""

from __future__ import annotations

import html
import io
import json
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from app.ir.model import Document, Element, ElementType
from app.ir.tree import SectionNode
from app.storage.base import serialize_document
from app.viewer.theme import TYPE_ORDER, color_for

__all__ = [
    "metrics_strip",
    "legend",
    "elements_table",
    "inspector",
    "sections_panel",
    "tables_panel",
    "images_panel",
    "warnings_panel",
    "json_panel",
    "stretch",
]


def stretch(component: Any, *args: Any, **kwargs: Any) -> Any:
    """Call a Streamlit component full-width across API versions."""
    try:
        return component(*args, width="stretch", **kwargs)
    except TypeError:
        return component(*args, use_container_width=True, **kwargs)


def _counts(doc: Document) -> dict[ElementType, int]:
    tally: dict[ElementType, int] = {}
    for element in doc.iter_elements():
        tally[element.type] = tally.get(element.type, 0) + 1
    return tally


def metrics_strip(doc: Document) -> None:
    tally = _counts(doc)
    total = sum(tally.values())
    warnings = len(doc.meta.get("warnings", []))

    columns = st.columns(6)
    columns[0].metric("Pages", doc.page_count)
    columns[1].metric("Elements", total)
    columns[2].metric("Headings", tally.get(ElementType.HEADING, 0))
    columns[3].metric("Tables", tally.get(ElementType.TABLE, 0))
    columns[4].metric("Images", tally.get(ElementType.IMAGE, 0))
    columns[5].metric("Warnings", warnings, delta=None if not warnings else "check tab")


def legend(doc: Document) -> None:
    tally = _counts(doc)
    present = [t for t in TYPE_ORDER if tally.get(t)]
    chips = "".join(
        f'<span class="kos-chip"><span class="kos-dot" '
        f'style="background:{color_for(t)}"></span>{t.value} · {tally[t]}</span>'
        for t in present
    )
    st.markdown(f'<div class="kos-legend">{chips}</div>', unsafe_allow_html=True)


def _preview(element: Element, limit: int = 90) -> str:
    text = " ".join(element.text_content().split())
    if element.type is ElementType.IMAGE and not text:
        ref = element.image.blob_ref if element.image else None
        text = f"[image {ref or 'unresolved'}]"
    return text[:limit] + ("…" if len(text) > limit else "")


def elements_table(doc: Document, page_number: int, key: str) -> str | None:
    """Render the element list for one page. Returns the selected element id."""
    page = next((p for p in doc.pages if p.number == page_number), None)
    if page is None or not page.elements:
        st.info("This page has no elements.")
        return None

    frame = pd.DataFrame(
        [
            {
                "order": element.order,
                "id": element.id,
                "type": element.type.value,
                "level": element.level,
                "parent": element.parent_id,
                "text": _preview(element),
                "x0": element.bbox.x0,
                "y0": element.bbox.y0,
                "x1": element.bbox.x1,
                "y1": element.bbox.y1,
            }
            for element in sorted(page.elements, key=lambda e: e.order)
        ]
    )

    event = stretch(
        st.dataframe,
        frame,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )

    rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if rows:
        return str(frame.iloc[rows[0]]["id"])
    return None


def inspector(doc: Document, element: Element | None) -> None:
    if element is None:
        st.markdown(
            '<div class="kos-card">Select a row in <b>Elements</b> to inspect it, '
            "or hide types in the sidebar to isolate what you are checking.</div>",
            unsafe_allow_html=True,
        )
        return

    provenance = element.provenance(doc.id)
    rows: list[tuple[str, str]] = [
        ("element", element.id),
        ("type", element.type.value),
        ("order", str(element.order)),
        ("page", str(element.page_number)),
        ("parent", element.parent_id or "—"),
        ("level", str(element.level) if element.level is not None else "—"),
        (
            "bbox",
            f"{element.bbox.x0}, {element.bbox.y0} → {element.bbox.x1}, {element.bbox.y1}",
        ),
        ("size", f"{element.bbox.width} × {element.bbox.height} pt"),
    ]
    body = "".join(
        f"<dt>{html.escape(key)}</dt><dd class='kos-mono'>{html.escape(value)}</dd>"
        for key, value in rows
    )
    st.markdown(
        f'<div class="kos-card"><dl class="kos-kv">{body}</dl></div>',
        unsafe_allow_html=True,
    )

    text = element.text_content()
    if text:
        st.markdown(
            f'<div class="kos-quote">{html.escape(text)}</div>', unsafe_allow_html=True
        )

    if element.attrs:
        with st.expander("attrs", expanded=False):
            st.json(element.attrs)

    with st.expander("provenance", expanded=False):
        st.code(
            json.dumps(provenance.model_dump(mode="json"), indent=2), language="json"
        )


def _walk(node: SectionNode, depth: int, lines: list[str]) -> None:
    title = node.title() or "(preamble)"
    marker = "·" * depth
    lines.append(
        f"{'  ' * depth}{marker} **{title}** "
        f"<span style='opacity:.55'>{len(node.elements)} elements</span>"
    )
    for child in node.children:
        _walk(child, depth + 1, lines)


def sections_panel(doc: Document) -> None:
    tree = doc.section_tree()
    if not tree:
        st.info("No sections — the document has no headings.")
        return
    lines: list[str] = []
    for node in tree:
        _walk(node, 0, lines)
    st.markdown("  \n".join(lines), unsafe_allow_html=True)


def tables_panel(doc: Document) -> None:
    tables = [e for e in doc.iter_elements() if e.table is not None]
    if not tables:
        st.info("No tables in this document.")
        return

    for element in tables:
        title = element.attrs.get("table_title") or element.attrs.get("sheet")
        caption = f" · {title}" if title else ""
        st.caption(
            f"`{element.id}` · page {element.page_number} · "
            f"{element.table.n_rows}×{element.table.n_cols}{caption}"
        )
        frame = pd.DataFrame(
            element.table.rows,
            columns=element.table.headers or None,
        )
        stretch(st.dataframe, frame, hide_index=True)

        formulas = element.attrs.get("formulas")
        if formulas:
            st.caption("formulas")
            st.json(formulas)


def images_panel(doc: Document, blobs: Mapping[str, bytes]) -> None:
    images = [e for e in doc.iter_elements() if e.image is not None]
    if not images:
        st.info("No images in this document.")
        return

    index = {e.id: e for e in doc.iter_elements()}
    for element in images:
        left, right = st.columns([1, 1])
        ref = element.image.blob_ref
        with left:
            if ref and ref in blobs:
                st.image(io.BytesIO(blobs[ref]))
            else:
                st.warning("Image bytes unavailable — element kept, blob missing.")
        with right:
            caption = index.get(element.image.caption_id or "")
            st.markdown(
                f"**`{element.id}`** · page {element.page_number}  \n"
                f"blob `{ref or '—'}` · {element.image.width}×{element.image.height} "
                f"{element.image.format or ''}  \n"
                f"caption: {caption.text if caption else '— none paired —'}"
            )


def warnings_panel(doc: Document) -> None:
    warnings = doc.meta.get("warnings", [])
    if not warnings:
        st.success("No warnings — every element parsed cleanly.")
        return
    st.caption(
        "Degradations are recorded, never silent. Each entry names what was lost "
        "and where."
    )
    stretch(st.dataframe, pd.DataFrame(warnings), hide_index=True)


def json_panel(doc: Document) -> None:
    payload = serialize_document(doc)
    st.download_button(
        "Download document.json",
        data=payload,
        file_name=f"{doc.id}.json",
        mime="application/json",
    )
    st.code(payload, language="json")
