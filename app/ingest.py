"""Parse files into the IR and persist them.

    uv run python -m app.ingest app/assets/payment_system.md
    uv run python -m app.ingest ./corpus --store ./store
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ir.model import Document
from app.parsing.base import InMemoryBlobSink, ParseError, parse_document
from app.storage.base import DocumentStore
from app.storage.filestore import FileDocumentStore

__all__ = ["ingest", "ingest_directory", "DEFAULT_STORE_DIR"]

DEFAULT_STORE_DIR = Path("store")


def ingest(path: Path | str, store: DocumentStore) -> Document:
    """Parse one file and persist it. Re-ingesting identical bytes is a no-op."""
    sink = InMemoryBlobSink()
    document = parse_document(path, blobs=sink)
    store.put(document, sink.blobs)
    return document


def ingest_directory(directory: Path | str, store: DocumentStore) -> list[Document]:
    """Ingest every supported file in a directory tree, skipping the rest."""
    documents: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        try:
            documents.append(ingest(path, store))
        except ParseError:
            continue
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse documents into the IR.")
    parser.add_argument("path", type=Path, help="file or directory to ingest")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE_DIR)
    args = parser.parse_args(argv)

    store = FileDocumentStore(args.store)
    targets = (
        ingest_directory(args.path, store)
        if args.path.is_dir()
        else [ingest(args.path, store)]
    )

    for document in targets:
        print(
            f"{document.id}  {document.source_name}  "
            f"{document.page_count} page(s)  "
            f"{sum(len(p.elements) for p in document.pages)} elements"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
