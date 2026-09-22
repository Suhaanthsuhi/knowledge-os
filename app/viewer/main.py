"""Knowledge OS.

    uv run streamlit run app/viewer/main.py

Chat with your documents, build the knowledge base, or inspect exactly what the
parser made of any single file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.viewer.modes import chat, corpus, inspect  # noqa: E402
from app.viewer.theme import CSS  # noqa: E402

MODES = {
    "Chat": chat.render,
    "Corpus": corpus.render,
    "Inspect": inspect.render,
}


def main() -> None:
    st.set_page_config(
        page_title="Knowledge OS",
        page_icon="◎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.title("Knowledge OS")
    mode = st.sidebar.radio("Mode", list(MODES), key="mode", label_visibility="collapsed")
    st.sidebar.divider()

    MODES[mode]()


if __name__ == "__main__":
    main()
