from __future__ import annotations

from pathlib import Path

import pytest

from app.ir.model import Document
from app.parsing import base
from app.parsing.base import (
    InMemoryBlobSink,
    ParseError,
    detect_extension,
    parse_document,
    register_parser,
    source_identity,
)


@pytest.fixture
def clean_registry():
    saved = dict(base._REGISTRY)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(saved)


class FakeParser:
    extensions = frozenset({".fake"})
    version = "1"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def parse(self, path: Path, *, blobs) -> Document:
        self.calls.append(path)
        blobs.add("p1_i000.png", b"x")
        short_id, checksum = source_identity(path)
        return Document(
            id=short_id, source_name=path.name, mime="application/fake", checksum=checksum
        )


def test_in_memory_blob_sink_collects_bytes():
    sink = InMemoryBlobSink()
    sink.add("a.png", b"1")
    sink.add("b.png", b"2")
    assert sink.blobs == {"a.png": b"1", "b.png": b"2"}


def test_in_memory_blob_sink_rejects_duplicate_refs():
    sink = InMemoryBlobSink()
    sink.add("a.png", b"1")
    with pytest.raises(ValueError, match="duplicate blob ref"):
        sink.add("a.png", b"2")


def test_source_identity_is_content_addressed(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    short_id, checksum = source_identity(path)
    assert len(short_id) == 16
    assert checksum.startswith(short_id)


def test_parse_document_dispatches_on_extension(tmp_path, clean_registry):
    parser = FakeParser()
    register_parser(".fake", parser)
    path = tmp_path / "doc.fake"
    path.write_bytes(b"content")

    doc = parse_document(path)

    assert parser.calls == [path]
    assert doc.source_name == "doc.fake"


def test_parse_document_collects_blobs_when_sink_supplied(tmp_path, clean_registry):
    register_parser(".fake", FakeParser())
    path = tmp_path / "doc.fake"
    path.write_bytes(b"content")

    sink = InMemoryBlobSink()
    parse_document(path, blobs=sink)

    assert sink.blobs == {"p1_i000.png": b"x"}


def test_parse_document_rejects_unknown_extension(tmp_path):
    path = tmp_path / "doc.unknown"
    path.write_bytes(b"not a known format")
    with pytest.raises(ParseError, match="no parser"):
        parse_document(path)


def test_parse_document_rejects_missing_file(tmp_path):
    with pytest.raises(ParseError, match="not a readable file"):
        parse_document(tmp_path / "absent.pdf")


@pytest.mark.parametrize(
    ("magic", "expected"),
    [
        (b"%PDF-1.7\n", ".pdf"),
        (b"\x89PNG\r\n\x1a\n", ".png"),
        (b"\xff\xd8\xff\xe0", ".jpg"),
        (b"PK\x03\x04", ".xlsx"),
    ],
)
def test_detect_extension_sniffs_magic_bytes(tmp_path, magic, expected):
    path = tmp_path / "mystery"
    path.write_bytes(magic + b"padding" * 10)
    assert detect_extension(path) == expected


def test_detect_extension_prefers_the_suffix(tmp_path):
    path = tmp_path / "doc.md"
    path.write_bytes(b"# heading")
    assert detect_extension(path) == ".md"
