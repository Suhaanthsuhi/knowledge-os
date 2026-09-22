from __future__ import annotations

import importlib
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from app.ir.ids import document_id
from app.ir.model import Document

__all__ = [
    "ParseError",
    "BlobSink",
    "InMemoryBlobSink",
    "DocumentParser",
    "register_parser",
    "detect_extension",
    "parse_document",
    "source_identity",
]


class ParseError(Exception):
    """Raised when a document cannot be parsed at all. Never a half-document."""


@runtime_checkable
class BlobSink(Protocol):
    def add(self, ref: str, data: bytes) -> None: ...


class InMemoryBlobSink:
    """Collects extracted binaries so parsers never touch the filesystem."""

    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def add(self, ref: str, data: bytes) -> None:
        if ref in self.blobs:
            raise ValueError(f"duplicate blob ref: {ref!r}")
        self.blobs[ref] = data


@runtime_checkable
class DocumentParser(Protocol):
    extensions: ClassVar[frozenset[str]]
    version: ClassVar[str]

    def parse(self, path: Path, *, blobs: BlobSink) -> Document: ...


# Extension -> dotted module path and class name. Imported lazily so that
# importing app.parsing.base does not pull in PyMuPDF, pdfplumber or openpyxl.
_BUILTIN: dict[str, tuple[str, str]] = {
    ".pdf": ("app.parsing.pdf", "PdfParser"),
    ".md": ("app.parsing.markdown", "MarkdownParser"),
    ".markdown": ("app.parsing.markdown", "MarkdownParser"),
    ".png": ("app.parsing.image", "ImageParser"),
    ".jpg": ("app.parsing.image", "ImageParser"),
    ".jpeg": ("app.parsing.image", "ImageParser"),
    ".xlsx": ("app.parsing.spreadsheet", "SpreadsheetParser"),
}

_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"PK\x03\x04", ".xlsx"),
)

_REGISTRY: dict[str, DocumentParser] = {}


def register_parser(extension: str, parser: DocumentParser) -> None:
    """Register (or override) the parser used for an extension."""
    _REGISTRY[extension.lower()] = parser


def _get_parser(extension: str) -> DocumentParser:
    extension = extension.lower()
    if extension in _REGISTRY:
        return _REGISTRY[extension]
    if extension not in _BUILTIN:
        raise ParseError(f"no parser registered for extension {extension!r}")
    module_name, class_name = _BUILTIN[extension]
    module = importlib.import_module(module_name)
    parser = getattr(module, class_name)()
    _REGISTRY[extension] = parser
    return parser


def detect_extension(path: Path) -> str:
    """Extension from the suffix, falling back to magic-byte sniffing."""
    suffix = path.suffix.lower()
    if suffix and (suffix in _REGISTRY or suffix in _BUILTIN):
        return suffix
    head = path.read_bytes()[:16]
    for magic, extension in _MAGIC:
        if head.startswith(magic):
            return extension
    return suffix


def source_identity(path: Path) -> tuple[str, str]:
    """(short document id, full sha256) for the file's bytes."""
    return document_id(Path(path).read_bytes())


def parse_document(path: Path | str, *, blobs: BlobSink | None = None) -> Document:
    """Parse any supported file into the IR. No network, no model calls."""
    path = Path(path)
    if not path.is_file():
        raise ParseError(f"not a readable file: {path}")
    sink = blobs if blobs is not None else InMemoryBlobSink()
    parser = _get_parser(detect_extension(path))
    return parser.parse(path, blobs=sink)
