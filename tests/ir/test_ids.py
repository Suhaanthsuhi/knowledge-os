from __future__ import annotations

from app.ir.ids import blob_ref, document_id, element_id


def test_document_id_is_content_addressed_and_stable():
    short_a, full_a = document_id(b"hello world")
    short_b, full_b = document_id(b"hello world")
    assert short_a == short_b
    assert full_a == full_b
    assert len(short_a) == 16
    assert len(full_a) == 64
    assert full_a.startswith(short_a)


def test_document_id_differs_for_different_bytes():
    assert document_id(b"a")[0] != document_id(b"b")[0]


def test_element_id_is_zero_padded_and_page_scoped():
    assert element_id(12, 7) == "p12e007"
    assert element_id(1, 0) == "p1e000"


def test_blob_ref_includes_page_index_and_extension():
    assert blob_ref(3, 1, "png") == "p3_i001.png"
