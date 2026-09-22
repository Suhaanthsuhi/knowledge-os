from __future__ import annotations

import hashlib

__all__ = ["document_id", "element_id", "blob_ref"]


def document_id(data: bytes) -> tuple[str, str]:
    """Return (short_id, full_checksum) derived from the file's bytes.

    Content addressing makes re-ingesting an identical file a no-op.
    """
    digest = hashlib.sha256(data).hexdigest()
    return digest[:16], digest


def element_id(page_number: int, index: int) -> str:
    """Stable element id, unique within a document."""
    return f"p{page_number}e{index:03d}"


def blob_ref(page_number: int, index: int, ext: str) -> str:
    """Stable filename for an extracted binary blob."""
    return f"p{page_number}_i{index:03d}.{ext.lstrip('.')}"
