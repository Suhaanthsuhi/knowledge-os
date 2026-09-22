from __future__ import annotations

import shutil

import pytest

from app.ingest import ingest, ingest_directory
from app.parsing.base import ParseError
from app.storage.filestore import FileDocumentStore


def test_ingest_parses_and_stores_in_one_call(tmp_path, fixtures):
    store = FileDocumentStore(tmp_path / "store")
    doc = ingest(fixtures["table_figure"], store)

    assert store.exists(doc.id)
    assert store.get(doc.id) == doc

    image = next(e for e in doc.iter_elements() if e.image and e.image.blob_ref)
    assert store.get_blob(doc.id, image.image.blob_ref)


def test_ingesting_the_same_file_twice_is_idempotent(tmp_path, fixtures):
    store = FileDocumentStore(tmp_path / "store")
    first = ingest(fixtures["simple"], store)
    second = ingest(fixtures["simple"], store)

    assert first.id == second.id
    assert list(store.list_ids()) == [first.id]


def test_ingest_directory_handles_mixed_formats(tmp_path, fixtures):
    source = tmp_path / "incoming"
    source.mkdir()
    for key in ("simple", "diagram_png", "sheet"):
        shutil.copy(fixtures[key], source / fixtures[key].name)
    shutil.copy("app/assets/payment_system.md", source / "payment_system.md")

    store = FileDocumentStore(tmp_path / "store")
    documents = ingest_directory(source, store)

    assert len(documents) == 4
    assert len(list(store.list_ids())) == 4
    assert {doc.mime for doc in documents} == {
        "application/pdf",
        "image/png",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/markdown",
    }


def test_ingest_directory_skips_unsupported_files_without_failing(tmp_path, fixtures):
    source = tmp_path / "incoming"
    source.mkdir()
    shutil.copy(fixtures["simple"], source / "doc.pdf")
    (source / "notes.rtf").write_text("unsupported")

    store = FileDocumentStore(tmp_path / "store")
    documents = ingest_directory(source, store)

    assert len(documents) == 1


def test_ingest_propagates_parse_errors_for_a_single_file(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7 nonsense")
    store = FileDocumentStore(tmp_path / "store")

    with pytest.raises(ParseError):
        ingest(broken, store)
