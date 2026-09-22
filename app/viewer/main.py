"""Document IR viewer.

    uv run streamlit run app/viewer/main.py

Upload or pick a document, see exactly what the parser made of it: every
element boxed on the rendered page, coloured by type, numbered in reading
order, with figure-to-caption links drawn in. The sidebar exposes the layout
thresholds, so you can watch classification change as you move them.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

# `streamlit run` puts this file's directory on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.ir.model import Document, ElementType  # noqa: E402
from app.parsing.base import InMemoryBlobSink, ParseError, parse_document  # noqa: E402
from app.parsing.layout import LayoutConfig  # noqa: E402
from app.parsing.pdf import PdfParser  # noqa: E402
from app.viewer import panels  # noqa: E402
from app.viewer.render import RenderOptions, is_synthetic, render_page  # noqa: E402
from app.viewer.theme import CSS, TYPE_ORDER  # noqa: E402

SUPPORTED = ["pdf", "md", "markdown", "png", "jpg", "jpeg", "xlsx"]
TUNABLE = (
    ("heading_size_ratio", 1.0, 2.0, 0.01, "Heading size ratio", "× body font size"),
    ("heading_max_chars", 20, 200, 5, "Heading max chars", "bold-line rule"),
    ("gutter_min_width", 4.0, 80.0, 1.0, "Gutter min width", "points"),
    ("gutter_min_coverage", 0.1, 1.0, 0.05, "Gutter min overlap", "of shorter column"),
    ("margin_band_ratio", 0.01, 0.30, 0.01, "Margin band", "of page height"),
    ("caption_max_distance", 5.0, 150.0, 5.0, "Caption max distance", "points"),
    ("caption_min_overlap", 0.1, 1.0, 0.05, "Caption min overlap", "of caption width"),
)


def bundled_sources() -> dict[str, Path]:
    found: dict[str, Path] = {}
    asset = ROOT / "app" / "assets" / "payment_system.md"
    if asset.is_file():
        found[asset.name] = asset
    fixtures = ROOT / "tests" / "fixtures" / "files"
    if fixtures.is_dir():
        for path in sorted(fixtures.iterdir()):
            if path.is_file() and path.suffix.lstrip(".").lower() in SUPPORTED:
                found[path.name] = path
    return found


@st.cache_data(show_spinner="Parsing…", max_entries=24)
def parse_source(name: str, data: bytes, config: tuple) -> tuple[Document, dict]:
    """Parse bytes into the IR. Cached on content plus layout configuration."""
    suffix = Path(name).suffix or ".bin"
    sink = InMemoryBlobSink()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        if suffix.lower() == ".pdf":
            cfg = LayoutConfig(**dict(config))
            document = PdfParser(cfg).parse(temporary, blobs=sink)
        else:
            document = parse_source_generic(temporary, sink)
    finally:
        temporary.unlink(missing_ok=True)

    document.source_name = name
    return document, sink.blobs


def parse_source_generic(path: Path, sink: InMemoryBlobSink) -> Document:
    return parse_document(path, blobs=sink)


def sidebar() -> tuple[str | None, bytes | None, LayoutConfig, RenderOptions, bool]:
    st.sidebar.markdown("### Source")
    bundled = bundled_sources()
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
        upload = st.sidebar.file_uploader("Document", type=SUPPORTED)
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


def main() -> None:
    st.set_page_config(
        page_title="Document IR · Knowledge OS",
        page_icon="◎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    name, data, cfg, options, _applied = sidebar()

    st.title("Document IR")
    st.caption(
        "Lossless, coordinate-preserving structure extracted before any LLM sees "
        "the document."
    )

    if name is None or data is None:
        st.info("Pick a bundled document or upload one to begin.")
        return

    config_key = tuple(sorted(asdict(cfg).items()))
    try:
        document, blobs = parse_source(name, data, config_key)
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


if __name__ == "__main__":
    main()
