"""Ask mode: a question answered only from retrieved evidence."""

from __future__ import annotations

import time

import streamlit as st

from app.graph.memory import InMemoryGraphStore
from app.rag.answer import answer_question
from app.viewer import panels, services

__all__ = ["render"]


def _evidence_card(evidence) -> None:
    passage = evidence.passage
    badges = " · ".join(evidence.sources)
    st.markdown(
        f"**`{passage.global_id}`** · {passage.source_name or passage.doc_id} · "
        f"page {passage.page_number} · {badges} · score {evidence.score:.4f}"
    )
    st.caption(
        f"bbox ({passage.x0:.0f}, {passage.y0:.0f}) → ({passage.x1:.0f}, {passage.y1:.0f})"
    )
    st.markdown(f"> {passage.text.strip()}")
    st.divider()


def render() -> None:
    st.subheader("Ask")
    st.caption(
        "Entities in the question seed a graph traversal; BM25 runs alongside it; "
        "the two rankings are fused, and the answer may use nothing else."
    )

    graph, graph_error = services.graph_store()
    model, model_error = services.language_model()

    if graph_error:
        st.warning(f"Neo4j unavailable — answering from lexical search only. {graph_error}")
    if model_error:
        st.error(f"Groq unavailable — answering is disabled. {model_error}")

    hops = st.sidebar.slider("Graph hops", 1, 3, 2)
    limit = st.sidebar.slider("Evidence passages", 3, 20, 8)

    index = services.bm25_index()
    st.sidebar.caption(f"{len(index.passages)} passages indexed")

    if not index.passages:
        st.info(
            "Nothing has been committed yet. Use **Ingest** to add a document first."
        )

    question = st.text_input(
        "Question", placeholder="Which team owns the service affected by INC-2391?"
    )
    if not st.button("Ask", type="primary", disabled=model is None) or not question:
        return

    active_graph = graph if graph is not None else InMemoryGraphStore()
    retriever = services.build_retriever(active_graph, index, hops=hops)

    started = time.perf_counter()
    result = retriever.retrieve(question, limit=limit)
    retrieved_at = time.perf_counter()
    answer = answer_question(model, result)
    finished = time.perf_counter()

    if answer.grounded:
        st.success(answer.text)
    else:
        st.warning(answer.text)

    if answer.dropped_citations:
        st.error(
            "The model cited evidence that does not exist; those citations were "
            f"dropped: {', '.join(answer.dropped_citations)}"
        )

    columns = st.columns(4)
    columns[0].metric("Evidence", len(result.evidence))
    columns[1].metric("Facts", len(result.facts))
    columns[2].metric("Citations", len(answer.citations))
    columns[3].metric("Total", f"{finished - started:.2f}s")

    tabs = st.tabs(["Citations", "Evidence", "Graph path", "Trace"])

    with tabs[0]:
        if not answer.citations:
            st.info("The answer cited nothing.")
        for citation in answer.citations:
            st.markdown(
                f"**`{citation.global_id}`** · {citation.source_name} · "
                f"page {citation.page_number}"
            )
            st.markdown(f"> {citation.text.strip()}")
            st.divider()

    with tabs[1]:
        if not result.evidence:
            st.info("Retrieval returned nothing.")
        for evidence in result.evidence:
            _evidence_card(evidence)

    with tabs[2]:
        if not result.facts:
            st.info("No graph facts were traversed.")
        else:
            panels.stretch(
                st.dataframe,
                [
                    {
                        "hop": fact.hop,
                        "source": fact.source_name,
                        "relation": fact.relation.value,
                        "target": fact.target_name,
                        "evidence": fact.evidence,
                        "elements": ", ".join(fact.global_element_ids()),
                    }
                    for fact in result.facts
                ],
                hide_index=True,
            )

    with tabs[3]:
        st.json(
            {
                **result.trace,
                "retrieval_seconds": round(retrieved_at - started, 3),
                "generation_seconds": round(finished - retrieved_at, 3),
            }
        )
