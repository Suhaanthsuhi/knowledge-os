from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.base import InMemoryBlobSink, parse_document
from app.storage.base import serialize_document

GOLDEN_DIR = Path(__file__).parent / "golden"
NAMES = ["simple", "two_column", "table_figure", "diagram_png", "sheet", "asset"]


@pytest.mark.parametrize("name", NAMES)
def test_parsed_output_matches_the_golden_file(name, fixtures):
    golden = GOLDEN_DIR / f"{name}.json"
    assert golden.is_file(), (
        f"missing {golden}. Run: uv run python -m scripts.regenerate_golden"
    )

    source = Path("app/assets/payment_system.md") if name == "asset" else fixtures[name]
    actual = serialize_document(parse_document(source, blobs=InMemoryBlobSink()))

    assert actual == golden.read_text(encoding="utf-8"), (
        f"{name} output changed. If intended, run "
        "`uv run python -m scripts.regenerate_golden` and review the diff."
    )
