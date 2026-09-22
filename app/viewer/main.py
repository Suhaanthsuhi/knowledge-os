"""Knowledge OS viewer.

    uv run streamlit run app/viewer/main.py

Three modes: inspect the parsed Document IR, ingest a document into the
knowledge graph, and ask questions answered from that graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.viewer.modes import ask, ingest, inspect  # noqa: E402
from app.viewer.theme import CSS  # noqa: E402

MODES = {
    "Inspect": inspect.render,
    "Ingest": ingest.render,
    "Ask": ask.render,
}


def main() -> None:
    st.set_page_config(
        page_title="Knowledge OS",
        page_icon="◎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Knowledge OS")

    mode = st.sidebar.radio("Mode", list(MODES), key="mode")
    st.sidebar.divider()
    MODES[mode]()


if __name__ == "__main__":
    main()
