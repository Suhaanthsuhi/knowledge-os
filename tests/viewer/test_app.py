from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves a relative path against this test file, not the repo root.
APP = str(Path(__file__).resolve().parents[2] / "app" / "viewer" / "main.py")


def test_the_app_starts_without_exceptions():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception


def test_the_default_document_parses_and_reports_metrics():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert {"Pages", "Elements", "Headings", "Tables", "Images", "Warnings"} <= set(labels)


def test_every_bundled_document_renders_without_error():
    app = AppTest.from_file(APP, default_timeout=120).run()
    options = app.selectbox[0].options

    for option in options:
        app.selectbox[0].set_value(option).run()
        assert not app.exception, f"{option} raised {app.exception}"


def test_every_mode_renders_without_exception():
    for mode in ("Inspect", "Ingest", "Ask"):
        app = AppTest.from_file(APP, default_timeout=120).run()
        app.sidebar.radio[0].set_value(mode).run()
        assert not app.exception, f"{mode} raised {app.exception}"


def test_the_ask_mode_reports_how_much_is_indexed():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.radio[0].set_value("Ask").run()

    assert not app.exception
    assert any("passages indexed" in caption.value for caption in app.sidebar.caption)
