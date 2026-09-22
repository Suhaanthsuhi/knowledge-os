"""Inspect mode: the Document IR viewer.

Renders a parsed page with its elements boxed, coloured by type and numbered in
reading order, and exposes the layout thresholds as live controls.
"""

from __future__ import annotations

from dataclasses import asdict, fields

import streamlit as st

from app.ir.model import ElementType
from app.parsing.base import ParseError
from app.parsing.layout import LayoutConfig
from app.viewer import panels, services
from app.viewer.render import RenderOptions, is_synthetic, render_page
from app.viewer.theme import TYPE_ORDER

__all__ = ["render"]

TUNABLE = (

    ("heading_size_ratio", 1.0, 2.0, 0.01, "Heading size ratio", "× body font size"),
    ("heading_max_chars", 20, 200, 5, "Heading max chars", "bold-line rule"),
    ("gutter_min_width", 4.0, 80.0, 1.0, "Gutter min width", "points"),
    ("gutter_min_coverage", 0.1, 1.0, 0.05, "Gutter min overlap", "of shorter column"),
    ("margin_band_ratio", 0.01, 0.30, 0.01, "Margin band", "of page height"),
    ("caption_max_distance", 5.0, 150.0, 5.0, "Caption max distance", "points"),
    ("caption_min_overlap", 0.1, 1.0, 0.05, "Caption min overlap", "of caption width"),
)


def sidebar() -> tuple[str | None, bytes | None, LayoutConfig, RenderOptions, bool]:
    st.sidebar.markdown("### Source")
    bundled = services.bundled_sources()
    mode = st.sidebar.radio(
        "Source", ["Bundled", "Upload"], horizontal=True, label_visibility="collapsed"
    )

    name: str | None = None
    data: bytes | None = None

    if mode == "Bundled" and bundled:
        choice = st.sidebar.selectbox("Document", list(bundled))
        name, data = choice, bundled[choice].read_bytes()
    elif mode == "Bundled":
        st.sidebar.warning("No bundled fixtures found.")
    else:
        upload = st.sidebar.file_uploader("Document", type=services.SUPPORTED)
        if upload is not None:
            name, data = upload.name, upload.getvalue()

    st.sidebar.markdown("### Display")
    dpi = st.sidebar.select_slider("Render DPI", [72, 96, 120, 144, 180, 220], value=144)
    show_boxes = st.sidebar.toggle("Element boxes", value=True)
    show_order = st.sidebar.toggle("Reading-order badges", value=True)
    show_connectors = st.sidebar.toggle("Figure → caption links", value=True)
    chosen = st.sidebar.multiselect(
        "Element types",
        [t.value for t in TYPE_ORDER],
        default=[t.value for t in TYPE_ORDER],
    )

    st.sidebar.markdown("### Layout thresholds")
    st.sidebar.caption(
        "Applies to PDFs. Move a threshold and watch elements reclassify."
    )
    defaults = LayoutConfig()
    values: dict[str, float | int] = {}
    with st.sidebar.form("layout", border=False):
        for field_name, low, high, step, label, helptext in TUNABLE:
            current = getattr(defaults, field_name)
            values[field_name] = st.slider(
                label,
                min_value=type(current)(low),
                max_value=type(current)(high),
                value=current,
                step=type(current)(step),
                help=helptext,
            )
        applied = st.form_submit_button("Apply thresholds", type="primary")

    cfg = LayoutConfig(**values)
    changed = any(
        getattr(cfg, f.name) != getattr(defaults, f.name) for f in fields(LayoutConfig)
    )
    if changed:
        st.sidebar.info("Thresholds differ from defaults.")

    options = RenderOptions(
        dpi=dpi,
        show_boxes=show_boxes,
        show_order=show_order,
        show_connectors=show_connectors,
        visible_types=frozenset(ElementType(value) for value in chosen),
    )
    return name, data, cfg, options, applied



def render() -> None:
    name, data, cfg, options, _applied = sidebar()

    st.subheader("Document IR")
    st.caption(
        "Lossless, coordinate-preserving structure extracted before any LLM sees "
        "the document."
    )

    if name is None or data is None:
        st.info("Pick a bundled document or upload one to begin.")
        return

    config_key = tuple(sorted(asdict(cfg).items()))
    try:
        document, blobs = services.parse_source(name, data, config_key)
    except ParseError as error:
        st.error(f"Could not parse **{name}**\n\n```\n{error}\n```")
        return

    st.markdown(
        f"**{document.title or document.source_name}** · `{document.id}` · "
        f"{document.mime} · parser `{document.meta.get('parser')}`"
    )
    panels.metrics_strip(document)
    panels.legend(document)

    if is_synthetic(document):
        st.caption(
            "This format carries no page geometry, so the parser invented it. "
            "Dashed borders mark synthetic bounding boxes."
        )

    page_numbers = [page.number for page in document.pages]
    page_number = page_numbers[0]
    if len(page_numbers) > 1:
        page_number = st.select_slider("Page", page_numbers, value=page_numbers[0])

    selected_key = f"selected::{document.id}::{page_number}"
    selected_id = st.session_state.get(selected_key)

    left, right = st.columns([3, 2], gap="large")

    with left:
        image = render_page(
            document,
            page_number,
            RenderOptions(**{**options.__dict__, "selected_id": selected_id}),
            source=data,
            blobs=blobs,
        )
        panels.stretch(st.image, image)

    with right:
        st.markdown("#### Inspector")
        element = None
        if selected_id:
            try:
                element = document.element(selected_id)
            except KeyError:
                element = None
        panels.inspector(document, element)

    tabs = st.tabs(["Elements", "Sections", "Tables", "Images", "Warnings", "JSON"])

    with tabs[0]:
        picked = panels.elements_table(
            document, page_number, key=f"table::{document.id}::{page_number}"
        )
        if picked and picked != selected_id:
            st.session_state[selected_key] = picked
            st.rerun()
    with tabs[1]:
        panels.sections_panel(document)
    with tabs[2]:
        panels.tables_panel(document)
    with tabs[3]:
        panels.images_panel(document, blobs)
    with tabs[4]:
        panels.warnings_panel(document)
    with tabs[5]:
        panels.json_panel(document)


