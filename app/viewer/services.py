"""Shared, cached wiring for the viewer's three modes.

Connections are cached as resources; parsing and indexing as data. Every
external dependency degrades to `None` with a message rather than raising, so
the UI stays usable when Neo4j or Groq is unavailable.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.neo4j import Neo4jGraphStore  # noqa: E402
from app.graph.store import GraphStore  # noqa: E402
from app.ir.model import Document  # noqa: E402
from app.knowledge.llm import GroqLLM, StructuredLLM  # noqa: E402
from app.parsing.base import InMemoryBlobSink, parse_document  # noqa: E402
from app.parsing.layout import LayoutConfig  # noqa: E402
from app.parsing.pdf import PdfParser  # noqa: E402
from app.rag.bm25 import Bm25Index  # noqa: E402
from app.rag.retriever import (  # noqa: E402
    Bm25Retriever,
    GraphRetriever,
    HybridRetriever,
    QuestionAnalyzer,
)
from app.storage.filestore import FileDocumentStore  # noqa: E402

__all__ = [
    "ROOT",
    "STORE_DIR",
    "SUPPORTED",
    "document_store",
    "graph_store",
    "language_model",
    "bm25_index",
    "refresh_index",
    "build_retriever",
    "bundled_sources",
    "parse_source",
    "default_config_key",
]

STORE_DIR = ROOT / "store"
SUPPORTED = ["pdf", "md", "markdown", "png", "jpg", "jpeg", "xlsx"]


def document_store() -> FileDocumentStore:
    return FileDocumentStore(STORE_DIR)


@st.cache_resource(show_spinner=False)
def graph_store() -> tuple[GraphStore | None, str | None]:
    """Connect to Neo4j. Returns (store, error) — never raises."""
    try:
        from app.config import settings

        store = Neo4jGraphStore(
            settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password
        )
        store.verify_connectivity()
        store.ensure_schema()
        return store, None
    except Exception as error:
        return None, str(error)


@st.cache_resource(show_spinner=False)
def language_model() -> tuple[StructuredLLM | None, str | None]:
    try:
        return GroqLLM(), None
    except Exception as error:
        return None, str(error)


@st.cache_data(show_spinner=False)
def _index_for(doc_ids: tuple[str, ...]) -> Bm25Index:
    return Bm25Index.from_store(document_store())


def bm25_index() -> Bm25Index:
    """Lexical index over everything committed, rebuilt when the corpus changes."""
    return _index_for(tuple(document_store().list_ids()))


def refresh_index() -> None:
    """Drop the cached lexical index after the corpus changes."""
    _index_for.clear()


def build_retriever(graph: GraphStore, index: Bm25Index, hops: int = 2) -> HybridRetriever:
    model, _error = language_model()
    analyzer = QuestionAnalyzer(model)
    return HybridRetriever(GraphRetriever(graph, analyzer, hops=hops), Bm25Retriever(index))


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


def default_config_key() -> tuple:
    return tuple(sorted(asdict(LayoutConfig()).items()))


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
            document = PdfParser(LayoutConfig(**dict(config))).parse(temporary, blobs=sink)
        else:
            document = parse_document(temporary, blobs=sink)
    finally:
        temporary.unlink(missing_ok=True)

    document.source_name = name
    return document, sink.blobs
