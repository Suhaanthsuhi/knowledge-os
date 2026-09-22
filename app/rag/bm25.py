from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from app.ir.model import Document
from app.knowledge.schema import ElementRef
from app.storage.base import DocumentStore

__all__ = ["Passage", "tokenize", "Bm25Index", "passage_from_element_ref"]

# A token is alphanumeric, optionally joined by - _ . so that INC-2391 and
# PAYMENT_502 survive tokenization intact.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")
_SPLIT = re.compile(r"[-_.]")

SKIPPED_TYPES = frozenset({"header", "footer", "page_number"})


@dataclass(frozen=True)
class Passage:
    doc_id: str
    element_id: str
    page_number: int
    element_type: str
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source_name: str = ""

    @property
    def global_id(self) -> str:
        return f"{self.doc_id}:{self.element_id}"


def passage_from_element_ref(ref: ElementRef, source_name: str = "") -> Passage:
    return Passage(
        doc_id=ref.doc_id,
        element_id=ref.id,
        page_number=ref.page_number,
        element_type=ref.type,
        text=ref.text,
        x0=ref.x0, y0=ref.y0, x1=ref.x1, y1=ref.y1,
        source_name=source_name,
    )


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, with compound identifiers kept whole and also split.

    Keeping only the whole token would lose a query for `2391`; keeping only the
    parts would lose the precision of the full identifier. Emitting both costs
    little and makes identifier search reliable.
    """
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        tokens.append(match)
        if _SPLIT.search(match):
            tokens.extend(part for part in _SPLIT.split(match) if part)
    return tokens


class Bm25Index:
    """Okapi BM25 over IR element text.

    Held in memory and rebuilt when the corpus changes. Fine for tens of
    documents; real indexing belongs to the serving layer.
    """

    def __init__(self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.passages = list(passages)
        self.passages_by_id = {passage.global_id: passage for passage in self.passages}

        self._tokens = [tokenize(passage.text) for passage in self.passages]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = (
            sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        )
        self._frequencies = [Counter(tokens) for tokens in self._tokens]

        document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())

        total = len(self.passages)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> Bm25Index:
        passages: list[Passage] = []
        for document in documents:
            for element in document.iter_elements():
                text = element.text_content().strip()
                if not text or element.type.value in SKIPPED_TYPES:
                    continue
                passages.append(
                    Passage(
                        doc_id=document.id,
                        element_id=element.id,
                        page_number=element.page_number,
                        element_type=element.type.value,
                        text=text,
                        x0=element.bbox.x0,
                        y0=element.bbox.y0,
                        x1=element.bbox.x1,
                        y1=element.bbox.y1,
                        source_name=document.source_name,
                    )
                )
        return cls(passages)

    @classmethod
    def from_store(cls, store: DocumentStore) -> Bm25Index:
        return cls.from_documents(store.get(doc_id) for doc_id in store.list_ids())

    def search(self, query: str, limit: int = 10) -> list[tuple[Passage, float]]:
        terms = tokenize(query)
        if not terms or not self.passages:
            return []

        scored: list[tuple[Passage, float]] = []
        for index, frequencies in enumerate(self._frequencies):
            length = self._lengths[index]
            if not length:
                continue
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self._average_length or 1)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((self.passages[index], score))

        scored.sort(key=lambda pair: (-pair[1], pair[0].global_id))
        return scored[:limit]
