from __future__ import annotations

import io
from pathlib import Path
from typing import ClassVar

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from app.ir.ids import blob_ref, element_id
from app.ir.model import BBox, Document, Element, ElementType, ImageRef, Page
from app.parsing.base import BlobSink, ParseError, source_identity

__all__ = ["ImageParser"]

_MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg"}


class ImageParser:
    """A standalone image becomes a one-page, one-element document.

    The uniform shape means a screenshot flows through the same retrieval and
    extraction pipeline as a PDF page, with no special-casing downstream.
    """

    extensions: ClassVar[frozenset[str]] = frozenset({".png", ".jpg", ".jpeg"})
    version: ClassVar[str] = "1"

    def parse(self, path: Path, *, blobs: BlobSink) -> Document:
        path = Path(path)
        data = path.read_bytes()
        short_id, checksum = source_identity(path)

        try:
            with PILImage.open(io.BytesIO(data)) as image:
                width, height = image.size
                image_format = (image.format or "PNG").upper()
        except (UnidentifiedImageError, OSError) as exc:
            raise ParseError(f"cannot decode {path} as an image: {exc}") from exc

        extension = "jpg" if image_format == "JPEG" else image_format.lower()
        ref = blob_ref(1, 0, extension)
        blobs.add(ref, data)

        element = Element(
            id=element_id(1, 0),
            type=ElementType.IMAGE,
            page_number=1,
            bbox=BBox(x0=0, y0=0, x1=float(width), y1=float(height)),
            order=0,
            image=ImageRef(blob_ref=ref, width=width, height=height, format=extension),
            attrs={"standalone": True},
        )

        return Document(
            id=short_id,
            source_name=path.name,
            mime=_MIME_BY_FORMAT.get(image_format, "application/octet-stream"),
            checksum=checksum,
            title=path.name,
            meta={
                "page_count": 1,
                "parser": "image",
                "parser_version": self.version,
                "image_format": image_format,
            },
            pages=[
                Page(number=1, width=float(width), height=float(height), elements=[element])
            ],
        )
