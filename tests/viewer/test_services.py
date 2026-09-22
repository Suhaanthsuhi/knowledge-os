from __future__ import annotations

from pathlib import Path

from app.viewer import services


def test_bundled_sources_includes_the_project_asset_and_fixtures():
    found = services.bundled_sources()
    assert "payment_system.md" in found
    assert any(name.endswith(".pdf") for name in found)
    assert all(isinstance(path, Path) and path.is_file() for path in found.values())


def test_parse_source_returns_a_document_and_its_blobs():
    path = services.bundled_sources()["payment_system.md"]
    document, blobs = services.parse_source(
        path.name, path.read_bytes(), services.default_config_key()
    )

    assert document.title == "Payment System"
    assert blobs == {}


def test_parse_source_keeps_the_original_file_name():
    path = services.bundled_sources()["payment_system.md"]
    document, _ = services.parse_source(
        "renamed.md", path.read_bytes(), services.default_config_key()
    )
    assert document.source_name == "renamed.md"


def test_build_retriever_wires_graph_and_lexical_arms():
    from app.graph.memory import InMemoryGraphStore
    from app.rag.bm25 import Bm25Index

    retriever = services.build_retriever(InMemoryGraphStore(), Bm25Index([]), hops=2)
    result = retriever.retrieve("anything at all")

    assert result.is_empty
    assert result.trace["hops"] == 2
