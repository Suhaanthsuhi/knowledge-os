from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping

from app.ir.model import Document
from app.storage.base import deserialize_document, serialize_document

__all__ = ["FileDocumentStore"]

DOCUMENT_FILENAME = "document.json"
BLOB_DIRNAME = "blobs"


class FileDocumentStore:
    """Stores each document as `<root>/<doc_id>/document.json` plus a blob directory.

    Because `doc_id` is a content hash, re-ingesting an identical file simply
    overwrites identical bytes — idempotency needs no dedup pass.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    @staticmethod
    def _check_ref(ref: str) -> None:
        if "/" in ref or "\\" in ref or ref in {".", ".."} or ref.startswith("."):
            raise ValueError(f"invalid blob ref: {ref!r}")

    def put(self, doc: Document, blobs: Mapping[str, bytes]) -> str:
        target = self._dir(doc.id)
        target.mkdir(parents=True, exist_ok=True)
        if blobs:
            blob_dir = target / BLOB_DIRNAME
            blob_dir.mkdir(exist_ok=True)
            for ref, data in sorted(blobs.items()):
                self._check_ref(ref)
                (blob_dir / ref).write_bytes(data)
        (target / DOCUMENT_FILENAME).write_text(serialize_document(doc), encoding="utf-8")
        return doc.id

    def get(self, doc_id: str) -> Document:
        path = self._dir(doc_id) / DOCUMENT_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"no stored document with id {doc_id!r}")
        return deserialize_document(path.read_text(encoding="utf-8"))

    def get_blob(self, doc_id: str, ref: str) -> bytes:
        self._check_ref(ref)
        path = self._dir(doc_id) / BLOB_DIRNAME / ref
        if not path.is_file():
            raise FileNotFoundError(f"no blob {ref!r} for document {doc_id!r}")
        return path.read_bytes()

    def exists(self, doc_id: str) -> bool:
        return (self._dir(doc_id) / DOCUMENT_FILENAME).is_file()

    def list_ids(self) -> Iterator[str]:
        if not self.root.is_dir():
            return
        for child in sorted(self.root.iterdir()):
            if (child / DOCUMENT_FILENAME).is_file():
                yield child.name
