from __future__ import annotations

import json

import pytest

from app.ir.model import BBox, Document, Element, ElementType, ImageRef, Page
from app.storage.filestore import FileDocumentStore


def _doc(doc_id: str = "abc123") -> Document:
    return Document(
        id=doc_id,
        source_name="payment.pdf",
        mime="application/pdf",
        checksum="f" * 64,
        title="Payment Architecture",
        meta={"page_count": 1, "parser": "pdf", "parser_version": "1"},
        pages=[
            Page(
                number=1,
                width=612,
                height=792,
                elements=[
                    Element(
                        id="p1e000",
                        type=ElementType.HEADING,
                        page_number=1,
                        bbox=BBox(x0=72, y0=100, x1=300, y1=120),
                        order=0,
                        level=1,
                        text="Payment Architecture",
                    ),
                    Element(
                        id="p1e001",
                        type=ElementType.IMAGE,
                        page_number=1,
                        bbox=BBox(x0=72, y0=140, x1=400, y1=340),
                        order=1,
                        parent_id="p1e000",
                        image=ImageRef(
                            blob_ref="p1_i000.png", width=640, height=400, format="png"
                        ),
                    ),
                ],
            )
        ],
    )


def test_put_then_get_round_trips(tmp_path):
    store = FileDocumentStore(tmp_path)
    doc = _doc()

    doc_id = store.put(doc, {"p1_i000.png": b"\x89PNG-bytes"})

    assert doc_id == "abc123"
    assert store.get("abc123") == doc


def test_put_writes_expected_layout(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {"p1_i000.png": b"bytes"})

    assert (tmp_path / "abc123" / "document.json").is_file()
    assert (tmp_path / "abc123" / "blobs" / "p1_i000.png").read_bytes() == b"bytes"


def test_serialized_json_is_sorted_and_deterministic(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {})
    first = (tmp_path / "abc123" / "document.json").read_text()

    store.put(_doc(), {})
    second = (tmp_path / "abc123" / "document.json").read_text()

    assert first == second
    top_level = list(json.loads(first).keys())
    assert top_level == sorted(top_level)


def test_get_blob_returns_bytes(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {"p1_i000.png": b"\x01\x02"})
    assert store.get_blob("abc123", "p1_i000.png") == b"\x01\x02"


def test_exists_and_list_ids(tmp_path):
    store = FileDocumentStore(tmp_path)
    assert not store.exists("abc123")
    store.put(_doc("abc123"), {})
    store.put(_doc("def456"), {})
    assert store.exists("abc123")
    assert sorted(store.list_ids()) == ["abc123", "def456"]


def test_get_missing_document_raises(tmp_path):
    store = FileDocumentStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("nope")


def test_blob_ref_cannot_escape_the_document_directory(tmp_path):
    store = FileDocumentStore(tmp_path)
    store.put(_doc(), {})
    with pytest.raises(ValueError, match="invalid blob ref"):
        store.get_blob("abc123", "../../etc/passwd")
