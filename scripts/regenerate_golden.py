"""Regenerate the committed golden IR files.

Intentional parser changes become a reviewable diff:

    uv run python -m scripts.regenerate_golden
    git diff tests/golden
"""

from __future__ import annotations

from pathlib import Path

from app.parsing.base import InMemoryBlobSink, parse_document
from app.storage.base import serialize_document
from tests.fixtures.generate import build_all

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "tests" / "golden"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    sources = build_all()
    sources["asset"] = Path("app/assets/payment_system.md")

    for name, path in sorted(sources.items()):
        document = parse_document(path, blobs=InMemoryBlobSink())
        target = GOLDEN_DIR / f"{name}.json"
        target.write_text(serialize_document(document), encoding="utf-8")
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
