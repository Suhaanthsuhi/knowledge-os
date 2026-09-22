"""Chat mode: ask the knowledge base, get grounded answers with citations."""

from __future__ import annotations

import streamlit as st

from app.graph.memory import InMemoryGraphStore
from app.rag.chat import ChatEngine, ChatTurn
from app.viewer import panels, services

__all__ = ["render"]

HISTORY_KEY = "chat_history"

SUGGESTIONS = [
    "Which team owns the service affected by INC-2391?",
    "What does the Payment API depend on?",
    "What happened in INC-2391?",
]


def _history() -> list[ChatTurn]:
    return st.session_state.setdefault(HISTORY_KEY, [])


def _sources(turn: ChatTurn) -> None:
    answer, result = turn.answer, turn.result

    tabs = st.tabs(
        [
            f"Citations ({len(answer.citations)})",
            f"Evidence ({len(result.evidence)})",
            f"Graph path ({len(result.facts)})",
            "Trace",
        ]
    )

    with tabs[0]:
        if not answer.citations:
            st.caption("This answer cited nothing.")
        for citation in answer.citations:
            st.markdown(
                f"**`{citation.global_id}`** · {citation.source_name or citation.doc_id}"
                f" · page {citation.page_number} · bbox "
                f"({citation.bbox[0]:.0f}, {citation.bbox[1]:.0f}) → "
                f"({citation.bbox[2]:.0f}, {citation.bbox[3]:.0f})"
            )
            st.markdown(f"> {citation.text.strip()}")

    with tabs[1]:
        if not result.evidence:
            st.caption("Retrieval returned nothing.")
        for evidence in result.evidence:
            passage = evidence.passage
            st.markdown(
                f"**`{passage.global_id}`** · {' · '.join(evidence.sources)} · "
                f"score {evidence.score:.4f}"
            )
            st.markdown(f"> {passage.text.strip()}")

    with tabs[2]:
        if not result.facts:
            st.caption("No graph facts were traversed.")
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
                    }
                    for fact in result.facts
                ],
                hide_index=True,
            )

    with tabs[3]:
        st.json(
            {
                **result.trace,
                "retrieval_seconds": turn.retrieval_seconds,
                "generation_seconds": turn.generation_seconds,
            }
        )


def _render_turn(turn: ChatTurn) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)

    with st.chat_message("assistant"):
        if turn.answer.grounded:
            st.markdown(turn.answer.text)
        else:
            st.warning(turn.answer.text)

        if turn.was_rewritten:
            st.caption(f"Interpreted as: _{turn.rewritten}_")

        if turn.answer.dropped_citations:
            st.error(
                "Dropped citations that name evidence which does not exist: "
                + ", ".join(turn.answer.dropped_citations)
            )

        summary = (
            f"{len(turn.answer.citations)} citations · "
            f"{len(turn.result.facts)} graph facts · "
            f"{turn.total_seconds:.2f}s"
        )
        with st.expander(f"Sources — {summary}", expanded=False):
            _sources(turn)


def render() -> None:
    graph, graph_error = services.graph_store()
    model, model_error = services.language_model()
    index = services.bm25_index()

    hops = st.sidebar.slider("Graph hops", 1, 3, 2, key="chat_hops")
    limit = st.sidebar.slider("Evidence passages", 3, 20, 8, key="chat_limit")
    st.sidebar.caption(
        f"{len(index.passages)} passages · "
        f"{graph.stats().facts if graph else 0} facts in the graph"
    )
    if st.sidebar.button("Clear conversation"):
        st.session_state[HISTORY_KEY] = []
        st.rerun()

    if graph_error:
        st.warning(f"Neo4j unavailable — answering from lexical search only. {graph_error}")
    if model_error:
        st.error(f"Groq unavailable — answering is disabled. {model_error}")

    history = _history()

    if not index.passages and not history:
        st.info(
            "Your knowledge base is empty. Add documents in **Corpus**, then come "
            "back and ask about them."
        )

    if not history:
        st.caption("Try one of these:")
        columns = st.columns(len(SUGGESTIONS))
        for column, suggestion in zip(columns, SUGGESTIONS):
            if column.button(suggestion, key=f"suggest::{suggestion}"):
                st.session_state["pending_question"] = suggestion
                st.rerun()

    for turn in history:
        _render_turn(turn)

    pending = st.session_state.pop("pending_question", None)
    typed = st.chat_input("Ask about your documents…", disabled=model is None)
    question = typed or pending
    if not question:
        return

    active_graph = graph if graph is not None else InMemoryGraphStore()
    engine = ChatEngine(
        services.build_retriever(active_graph, index, hops=hops), model, limit=limit
    )

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Retrieving and answering…"):
        turn = engine.ask(question, history)

    history.append(turn)
    st.rerun()
