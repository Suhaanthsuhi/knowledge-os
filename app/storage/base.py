from __future__ import annotations

import json
from typing import Iterator, Mapping, Protocol, runtime_checkable

from app.ir.model import Document

__all__ = ["serialize_document", "deserialize_document", "DocumentStore"]


def serialize_document(doc: Document) -> str:
    """Canonical JSON: sorted keys, stable indentation, no ASCII escaping.

    BBox floats are already rounded at validation time, so the same document
    always produces the same bytes.
    """
    payload = doc.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def deserialize_document(raw: str) -> Document:
    return Document.model_validate_json(raw)


@runtime_checkable
class DocumentStore(Protocol):
    def put(self, doc: Document, blobs: Mapping[str, bytes]) -> str: ...
    def get(self, doc_id: str) -> Document: ...
    def get_blob(self, doc_id: str, ref: str) -> bytes: ...
    def exists(self, doc_id: str) -> bool: ...
    def list_ids(self) -> Iterator[str]: ...
