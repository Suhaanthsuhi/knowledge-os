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


def test_inspect_mode_parses_a_document_and_reports_metrics():
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.sidebar.radio[0].set_value("Inspect").run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert {"Pages", "Elements", "Headings", "Tables", "Images", "Warnings"} <= set(labels)


def test_every_bundled_document_renders_without_error():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.radio[0].set_value("Inspect").run()
    options = app.selectbox[0].options

    for option in options:
        app.selectbox[0].set_value(option).run()
        assert not app.exception, f"{option} raised {app.exception}"


def test_every_mode_renders_without_exception():
    for mode in ("Chat", "Corpus", "Inspect"):
        app = AppTest.from_file(APP, default_timeout=120).run()
        app.sidebar.radio[0].set_value(mode).run()
        assert not app.exception, f"{mode} raised {app.exception}"


def test_the_chat_mode_reports_how_much_is_indexed():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.radio[0].set_value("Chat").run()

    assert not app.exception
    assert any("passages" in caption.value for caption in app.sidebar.caption)


def test_chat_offers_starter_questions_when_there_is_no_history():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.radio[0].set_value("Chat").run()

    labels = [button.label for button in app.button]
    assert any("INC-2391" in label for label in labels)


def test_corpus_mode_shows_the_knowledge_base_overview():
    app = AppTest.from_file(APP, default_timeout=120).run()
    app.sidebar.radio[0].set_value("Corpus").run()

    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert {"Documents", "Entities", "Facts", "Passages indexed"} <= set(labels)
