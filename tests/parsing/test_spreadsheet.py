from __future__ import annotations

import pytest

from app.ir.model import ElementType
from app.parsing.base import InMemoryBlobSink, ParseError
from app.parsing.spreadsheet import SpreadsheetParser


@pytest.fixture(scope="module")
def workbook(fixtures):
    return SpreadsheetParser().parse(fixtures["sheet"], blobs=InMemoryBlobSink())


def test_each_sheet_becomes_a_page(workbook):
    assert workbook.page_count == 2
    assert [page.number for page in workbook.pages] == [1, 2]


def test_each_sheet_contributes_a_heading_and_a_table(workbook):
    first = workbook.pages[0].elements
    assert first[0].type is ElementType.HEADING
    assert first[0].text == "Services"
    assert first[1].type is ElementType.TABLE


def test_table_rows_and_headers_are_extracted(workbook):
    table = workbook.pages[1].elements[1].table
    assert table.headers == ["Incident", "Service", "Hours"]
    assert ["INC-2391", "Payment API", "4"] in table.rows


def test_a_merged_title_row_is_lifted_out_rather_than_read_as_headers(workbook):
    element = workbook.pages[0].elements[1]
    assert element.attrs["table_title"] == "Service Inventory"
    assert element.table.headers == ["Service", "Datastore", "Owner"]
    assert ["Payment API", "PostgreSQL", "Payments Team"] in element.table.rows
    assert "Service Inventory" not in [cell for row in element.table.rows for cell in row]


def test_formulas_are_recorded_alongside_their_cached_value(workbook):
    element = workbook.pages[1].elements[1]
    assert element.attrs["formulas"] == {"C4": "=SUM(C2:C3)"}


def test_sheet_names_are_recorded_in_metadata(workbook):
    assert workbook.meta["sheet_names"] == ["Services", "Incidents"]
    assert workbook.meta["parser"] == "spreadsheet"


def test_cells_carry_synthetic_grid_geometry(workbook):
    table_element = workbook.pages[0].elements[1]
    assert table_element.attrs["synthetic_bbox"] is True
    assert table_element.bbox.x1 > table_element.bbox.x0
    assert table_element.bbox.y1 > table_element.bbox.y0


def test_the_table_is_parented_to_its_sheet_heading(workbook):
    page = workbook.pages[0]
    assert page.elements[1].parent_id == page.elements[0].id


def test_parsing_is_deterministic(fixtures):
    from app.storage.base import serialize_document

    first = SpreadsheetParser().parse(fixtures["sheet"], blobs=InMemoryBlobSink())
    second = SpreadsheetParser().parse(fixtures["sheet"], blobs=InMemoryBlobSink())
    assert serialize_document(first) == serialize_document(second)


def test_a_non_workbook_raises_parse_error(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"PK\x03\x04 not a workbook")
    with pytest.raises(ParseError):
        SpreadsheetParser().parse(path, blobs=InMemoryBlobSink())
