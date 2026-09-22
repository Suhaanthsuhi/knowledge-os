"""Corpus mode: add many documents to the knowledge base at once."""

from __future__ import annotations

import streamlit as st

from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.resolver import EntityResolver
from app.viewer import panels, services
from app.viewer.ingestion import SourceFile, Stage, ingest_batch

__all__ = ["render"]

STAGE_ICON = {
    Stage.QUEUED: "·",
    Stage.PARSING: "◔",
    Stage.EXTRACTING: "◑",
    Stage.COMMITTING: "◕",
    Stage.DONE: "✓",
    Stage.FAILED: "✗",
}


def _overview(graph) -> None:
    index = services.bm25_index()
    totals = graph.stats() if graph else None

    columns = st.columns(5)
    columns[0].metric("Documents", totals.documents if totals else "—")
    columns[1].metric("Entities", totals.entities if totals else "—")
    columns[2].metric("Facts", totals.facts if totals else "—")
    columns[3].metric("Evidence elements", totals.elements if totals else "—")
    columns[4].metric("Passages indexed", len(index.passages))


def _collect_files() -> list[SourceFile]:
    uploads = st.file_uploader(
        "Add documents",
        type=services.SUPPORTED,
        accept_multiple_files=True,
        help="PDF, Markdown, PNG/JPEG screenshots, or XLSX spreadsheets.",
    )
    files = [SourceFile(name=item.name, data=item.getvalue()) for item in uploads or []]

    bundled = services.bundled_sources()
    with st.expander("…or use the bundled sample documents", expanded=not files):
        chosen = st.multiselect(
            "Samples", list(bundled), default=[], label_visibility="collapsed"
        )
        files.extend(
            SourceFile(name=name, data=bundled[name].read_bytes()) for name in chosen
        )

    return files


def _render_outcomes(outcomes) -> None:
    panels.stretch(
        st.dataframe,
        [
            {
                "": STAGE_ICON.get(outcome.stage, "·"),
                "file": outcome.name,
                "stage": outcome.stage.value,
                "entities": outcome.entity_count,
                "facts": outcome.fact_count,
                "warnings": len(outcome.warnings),
                "error": outcome.error or "",
            }
            for outcome in outcomes
        ],
        hide_index=True,
    )


def _detail(outcome) -> None:
    if outcome.upsert is None:
        st.caption(outcome.error or "Nothing extracted.")
        return

    names = {entity.key: entity.name for entity in outcome.upsert.entities}
    tabs = st.tabs(["Facts", "Entities", "Warnings"])

    with tabs[0]:
        if not outcome.upsert.facts:
            st.caption("No relationships were stated in this document.")
        else:
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
                    for fact in outcome.upsert.facts
                ],
                hide_index=True,
            )
            st.caption(
                "`confidence` is low where the evidence quote could not be matched "
                "to a single element, so that fact's provenance is chunk-level."
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
                for entity in outcome.upsert.entities
            ],
            hide_index=True,
        )

    with tabs[2]:
        if outcome.warnings:
            panels.stretch(st.dataframe, outcome.warnings, hide_index=True)
        else:
            st.success("No warnings — every chunk extracted cleanly.")


def _danger_zone(graph) -> None:
    with st.expander("Danger zone", expanded=False):
        st.caption(
            "Deletes every Document, Entity and Element node written by this "
            "pipeline. This cannot be undone."
        )
        confirmed = st.checkbox("I understand this deletes the graph", key="reset_ok")
        if st.button("Reset knowledge base", disabled=graph is None or not confirmed):
            graph.reset()
            services.refresh_index()
            st.session_state.pop("batch", None)
            st.success("Knowledge base reset.")


def render() -> None:
    st.subheader("Corpus")
    st.caption(
        "Add documents to the knowledge base. Each is parsed into structured "
        "elements, mined for entities and relationships, and committed with the "
        "evidence each fact came from."
    )

    graph, graph_error = services.graph_store()
    model, model_error = services.language_model()

    if graph_error:
        st.warning(f"Neo4j unavailable — committing is disabled. {graph_error}")
    if model_error:
        st.error(f"Groq unavailable — extraction is disabled. {model_error}")

    _overview(graph)
    st.divider()

    files = _collect_files()
    review = st.sidebar.toggle(
        "Review before committing",
        value=False,
        help="Extract and show the facts, but do not write them to the graph.",
    )
    workers = st.sidebar.slider("Extraction workers", 1, 8, 4)

    disabled = not files or model is None or graph is None
    label = (
        f"Add {len(files)} document{'s' if len(files) != 1 else ''} to knowledge base"
        if files
        else "Add documents to knowledge base"
    )

    if st.button(label, type="primary", disabled=disabled):
        pipeline = KnowledgePipeline(
            KnowledgeExtractor(model), EntityResolver(model), max_workers=workers
        )
        status = st.empty()
        progress = st.progress(0.0)
        total = len(files)
        finished = {"count": 0}

        def on_update(outcome) -> None:
            fraction = finished["count"] / total
            if outcome.stage in (Stage.DONE, Stage.FAILED):
                finished["count"] += 1
                fraction = finished["count"] / total
            detail = (
                f" · chunk {outcome.chunks_done}/{outcome.chunks_total}"
                if outcome.chunks_total
                else ""
            )
            status.markdown(
                f"**{outcome.name}** — {outcome.stage.value}{detail} "
                f"({finished['count']}/{total} complete)"
            )
            progress.progress(min(fraction, 1.0))

        result = ingest_batch(
            files,
            parse=lambda name, data: services.parse_source(
                name, data, services.default_config_key()
            ),
            pipeline=pipeline,
            documents=services.document_store(),
            graph=graph,
            commit=not review,
            on_update=on_update,
        )

        services.refresh_index()
        status.empty()
        progress.empty()
        st.session_state["batch"] = result

    result = st.session_state.get("batch")
    if result is None:
        return

    totals = result.totals()
    if result.failed:
        st.warning(
            f"{len(result.succeeded)} of {len(result.outcomes)} documents added. "
            f"{len(result.failed)} failed — see the table."
        )
    elif review:
        st.info(
            f"Extracted {totals.facts} facts from {len(result.succeeded)} documents. "
            "Nothing was committed — turn off *Review before committing* to write them."
        )
    else:
        st.success(
            f"Added {len(result.succeeded)} documents · {totals.entities} entities · "
            f"{totals.facts} facts · {totals.elements} evidence elements."
        )

    _render_outcomes(result.outcomes)

    for outcome in result.outcomes:
        with st.expander(f"{outcome.name} — {outcome.fact_count} facts", expanded=False):
            _detail(outcome)

    _danger_zone(graph)
