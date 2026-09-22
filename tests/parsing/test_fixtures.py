from __future__ import annotations

import hashlib


def test_all_fixtures_are_generated(fixtures):
    assert set(fixtures) == {"simple", "two_column", "table_figure", "diagram_png", "sheet"}
    for path in fixtures.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_generated_fixtures_are_byte_stable(tmp_path, fixtures):
    """Same bytes every time, in this run and across sessions.

    Comparing only two builds from one run would miss embedded timestamps,
    which agree within a session but drift between them - so the committed
    fixtures are part of the comparison.
    """
    from tests.fixtures.generate import build_all

    first = build_all(tmp_path / "a")
    second = build_all(tmp_path / "b")

    for key in first:
        digests = {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (first[key], second[key], fixtures[key])
        }
        assert len(digests) == 1, f"{key} is not byte-stable across builds"


def test_pdf_fixtures_start_with_the_pdf_magic(fixtures):
    for key in ("simple", "two_column", "table_figure"):
        assert fixtures[key].read_bytes().startswith(b"%PDF-")
