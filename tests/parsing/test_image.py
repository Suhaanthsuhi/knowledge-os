from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, ParseError
from app.parsing.image import ImageParser


def test_a_standalone_image_becomes_one_page_one_element(fixtures):
    sink = InMemoryBlobSink()
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=sink)

    assert doc.page_count == 1
    assert len(doc.pages[0].elements) == 1

    element = doc.pages[0].elements[0]
    assert element.type is ElementType.IMAGE
    assert element.order == 0
    assert element.image.format == "png"
    assert (element.image.width, element.image.height) == (480, 300)


def test_the_bbox_covers_the_whole_page(fixtures):
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    page = doc.pages[0]
    element = page.elements[0]

    assert (element.bbox.x0, element.bbox.y0) == (0.0, 0.0)
    assert (element.bbox.x1, element.bbox.y1) == (page.width, page.height)


def test_the_original_bytes_are_stored_as_a_blob(fixtures):
    sink = InMemoryBlobSink()
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=sink)
    ref = doc.pages[0].elements[0].image.blob_ref

    assert sink.blobs[ref] == fixtures["diagram_png"].read_bytes()


def test_metadata_records_the_parser_and_mime(fixtures):
    doc = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    assert doc.mime == "image/png"
    assert doc.meta["parser"] == "image"
    assert doc.title == fixtures["diagram_png"].name


def test_parsing_is_deterministic(fixtures):
    from app.storage.base import serialize_document

    first = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    second = ImageParser().parse(fixtures["diagram_png"], blobs=InMemoryBlobSink())
    assert serialize_document(first) == serialize_document(second)


def test_an_undecodable_image_raises_parse_error(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnot really a png")
    with pytest.raises(ParseError, match="cannot decode"):
        ImageParser().parse(path, blobs=InMemoryBlobSink())
