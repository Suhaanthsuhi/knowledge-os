"""Ingest mode: parse a document, preview what the model extracted, then commit."""

from __future__ import annotations

import streamlit as st

from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import GraphUpsert
from app.parsing.base import ParseError
from app.viewer import panels, services

__all__ = ["render"]


def _source_picker() -> tuple[str | None, bytes | None]:
    bundled = services.bundled_sources()
    mode = st.sidebar.radio(
        "Source", ["Bundled", "Upload"], horizontal=True, key="ingest_source"
    )
    if mode == "Bundled" and bundled:
        choice = st.sidebar.selectbox("Document", list(bundled), key="ingest_doc")
        return choice, bundled[choice].read_bytes()
    if mode == "Upload":
        upload = st.sidebar.file_uploader(
            "Document", type=services.SUPPORTED, key="ingest_upload"
        )
        if upload is not None:
            return upload.name, upload.getvalue()
    return None, None


def _preview(upsert: GraphUpsert) -> None:
    columns = st.columns(4)
    columns[0].metric("Entities", len(upsert.entities))
    columns[1].metric("Facts", len(upsert.facts))
    columns[2].metric("Evidence elements", len(upsert.elements))
    columns[3].metric("Warnings", len(upsert.warnings))

    tabs = st.tabs(["Facts", "Entities", "Warnings"])

    with tabs[0]:
        if not upsert.facts:
            st.info("No facts extracted. The text may not state any relationships.")
        else:
            names = {entity.key: entity.name for entity in upsert.entities}
            panels.stretch(
                st.dataframe,
                [
                    {
                        "source": names.get(fact.source_key, fact.source_key),
                        "relation": fact.relation.value,
                        "target": names.get(fact.target_key, fact.target_key),
                        "evidence": fact.evidence,
                        "elements": ", ".join(fact.element_ids),
                        "confidence": fact.confidence,
                    }
                    for fact in upsert.facts
                ],
                hide_index=True,
            )
            st.caption(
                "`confidence` is low when the evidence quote could not be matched "
                "to a single element, so provenance is chunk-level for that fact."
            )

    with tabs[1]:
        panels.stretch(
            st.dataframe,
            [
                {
                    "name": entity.name,
                    "type": entity.type.value,
                    "key": entity.key,
                    "aliases": ", ".join(entity.aliases),
                }
                for entity in upsert.entities
            ],
            hide_index=True,
        )

    with tabs[2]:
        if upsert.warnings:
            panels.stretch(st.dataframe, upsert.warnings, hide_index=True)
        else:
            st.success("No warnings — every chunk extracted cleanly.")


def _danger_zone(graph) -> None:
    with st.expander("Danger zone", expanded=False):
        st.caption(
            "Deletes every Document, Entity and Element node. This cannot be undone."
        )
        confirmed = st.checkbox("I understand this deletes the graph", key="reset_ok")
        if st.button("Reset graph", disabled=graph is None or not confirmed):
            graph.reset()
            services.refresh_index()
            st.success("Graph reset.")


def render() -> None:
    st.subheader("Ingest")
    st.caption(
        "Parse a document, review what the model claims and the quote it claims "
        "it from, then commit it to the store and the graph."
    )

    graph, graph_error = services.graph_store()
    model, model_error = services.language_model()

    if graph_error:
        st.warning(f"Neo4j unavailable — committing is disabled. {graph_error}")
    if model_error:
        st.error(f"Groq unavailable — extraction is disabled. {model_error}")

    if graph is not None:
        totals = graph.stats()
        columns = st.columns(4)
        columns[0].metric("Documents in graph", totals.documents)
        columns[1].metric("Entities", totals.entities)
        columns[2].metric("Facts", totals.facts)
        columns[3].metric("Evidence elements", totals.elements)

    name, data = _source_picker()
    if name is None or data is None:
        st.info("Pick a bundled document or upload one to begin.")
        _danger_zone(graph)
        return

    try:
        document, blobs = services.parse_source(name, data, services.default_config_key())
    except ParseError as error:
        st.error(f"Could not parse **{name}**\n\n```\n{error}\n```")
        return

    st.markdown(f"**{document.title or document.source_name}** · `{document.id}`")
    panels.metrics_strip(document)

    state_key = f"upsert::{document.id}"

    if st.button("Extract knowledge", type="primary", disabled=model is None):
        known = graph.known_entities() if graph else []
        pipeline = KnowledgePipeline(KnowledgeExtractor(model), EntityResolver(model))
        progress = st.progress(0.0, text="Extracting…")

        def report(done: int, total: int) -> None:
            progress.progress(done / max(total, 1), text=f"Chunk {done} of {total}")

        try:
            st.session_state[state_key] = pipeline.run(document, known, on_progress=report)
        except Exception as error:
            st.error(f"Extraction failed: {error}")
        finally:
            progress.empty()

    upsert = st.session_state.get(state_key)
    if upsert is None:
        _danger_zone(graph)
        return

    _preview(upsert)

    if st.button("Commit to store and graph", type="primary", disabled=graph is None):
        # The store is written first: a document that is searchable without its
        # graph edges is recoverable, whereas edges citing elements no store can
        # resolve are dead citations.
        services.document_store().put(document, blobs)
        stats = graph.apply(upsert)
        services.refresh_index()  # the corpus grew; drop the cached index

        st.success(
            f"Committed {stats.entities} entities, {stats.facts} facts and "
            f"{stats.elements} evidence elements from **{document.source_name}**."
        )
        totals = graph.stats()
        st.caption(
            f"Graph now holds {totals.documents} documents, {totals.entities} "
            f"entities and {totals.facts} facts."
        )

    _danger_zone(graph)
