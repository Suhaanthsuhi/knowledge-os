"""Batch ingestion: many files in, knowledge graph out.

Deliberately free of Streamlit so the sequencing and failure handling can be
tested directly. Progress reaches the UI through a callback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Iterable, Protocol, Sequence

from app.graph.store import GraphStats, GraphStore
from app.ir.model import Document
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.schema import GraphUpsert
from app.parsing.base import ParseError
from app.storage.base import DocumentStore

__all__ = ["Stage", "SourceFile", "FileOutcome", "BatchResult", "ingest_batch"]


class Stage(StrEnum):
    QUEUED = "queued"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    COMMITTING = "committing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceFile:
    name: str
    data: bytes


@dataclass
class FileOutcome:
    name: str
    stage: Stage = Stage.QUEUED
    document: Document | None = None
    upsert: GraphUpsert | None = None
    stats: GraphStats | None = None
    error: str | None = None
    chunks_done: int = 0
    chunks_total: int = 0

    @property
    def succeeded(self) -> bool:
        return self.stage is Stage.DONE

    @property
    def fact_count(self) -> int:
        return len(self.upsert.facts) if self.upsert else 0

    @property
    def entity_count(self) -> int:
        return len(self.upsert.entities) if self.upsert else 0

    @property
    def warnings(self) -> list[dict]:
        return list(self.upsert.warnings) if self.upsert else []


@dataclass
class BatchResult:
    outcomes: list[FileOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> list[FileOutcome]:
        return [outcome for outcome in self.outcomes if outcome.succeeded]

    @property
    def failed(self) -> list[FileOutcome]:
        return [outcome for outcome in self.outcomes if outcome.stage is Stage.FAILED]

    def totals(self) -> GraphStats:
        total = GraphStats()
        for outcome in self.succeeded:
            if outcome.stats:
                total = total + outcome.stats
        return total


class Parser(Protocol):
    def __call__(self, name: str, data: bytes) -> tuple[Document, dict]: ...


def ingest_batch(
    files: Sequence[SourceFile],
    *,
    parse: Parser,
    pipeline: KnowledgePipeline,
    documents: DocumentStore,
    graph: GraphStore,
    commit: bool = True,
    on_update: Callable[[FileOutcome], None] | None = None,
) -> BatchResult:
    """Parse, extract and commit each file in turn.

    Files are processed sequentially even though each file's chunks extract in
    parallel: the entity registry grows as documents are ingested, so resolving
    two documents at once would make deduplication depend on timing.

    A failure is confined to its own file. One unreadable PDF in a folder of
    twenty must not cost the other nineteen.
    """
    result = BatchResult()

    for source in files:
        outcome = FileOutcome(name=source.name)
        result.outcomes.append(outcome)

        def publish() -> None:
            if on_update:
                on_update(outcome)

        try:
            outcome.stage = Stage.PARSING
            publish()
            document, blobs = parse(source.name, source.data)
            outcome.document = document

            outcome.stage = Stage.EXTRACTING
            publish()

            def report(done: int, total: int) -> None:
                outcome.chunks_done, outcome.chunks_total = done, total
                publish()

            known = graph.known_entities()
            outcome.upsert = pipeline.run(document, known, on_progress=report)

            if not commit:
                outcome.stage = Stage.DONE
                publish()
                continue

            outcome.stage = Stage.COMMITTING
            publish()
            # The store is written first: a document searchable without its graph
            # edges is recoverable; edges citing elements no store can resolve
            # are dead citations.
            documents.put(document, blobs)
            outcome.stats = graph.apply(outcome.upsert)

            outcome.stage = Stage.DONE
            publish()

        except ParseError as error:
            outcome.stage = Stage.FAILED
            outcome.error = f"could not parse: {error}"
            publish()
        except Exception as error:
            outcome.stage = Stage.FAILED
            outcome.error = str(error)
            publish()

    return result
