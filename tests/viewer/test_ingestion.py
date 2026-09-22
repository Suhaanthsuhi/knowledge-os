from __future__ import annotations

import pytest

from app.graph.memory import InMemoryGraphStore
from app.knowledge.extractor import KnowledgeExtractor
from app.knowledge.llm import FakeLLM
from app.knowledge.pipeline import KnowledgePipeline
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import (
    ChunkExtraction,
    EntityType,
    ExtractedEntity,
    ExtractedFact,
    RelationType,
)
from app.parsing.base import InMemoryBlobSink, ParseError, parse_document
from app.storage.filestore import FileDocumentStore
from app.viewer.ingestion import SourceFile, Stage, ingest_batch

ASSET = "app/assets/payment_system.md"


def _extraction(prompt: str, schema):
    if "Redis" not in prompt:
        return ChunkExtraction()
    return ChunkExtraction(
        entities=[
            ExtractedEntity(name="Payment API", type=EntityType.SERVICE),
            ExtractedEntity(name="Redis", type=EntityType.DATABASE),
        ],
        facts=[
            ExtractedFact(
                source="Payment API", relation=RelationType.DEPENDS_ON, target="Redis",
                evidence="The Payment API depends on Redis for caching and PostgreSQL",
            )
        ],
    )


def _pipeline() -> KnowledgePipeline:
    llm = FakeLLM(handler=_extraction)
    return KnowledgePipeline(KnowledgeExtractor(llm), EntityResolver(None))


def _parse(name: str, data: bytes):
    import tempfile
    from pathlib import Path

    sink = InMemoryBlobSink()
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix, delete=False) as handle:
        handle.write(data)
        path = Path(handle.name)
    try:
        document = parse_document(path, blobs=sink)
    finally:
        path.unlink(missing_ok=True)
    document.source_name = name
    return document, sink.blobs


def _files(count: int = 1) -> list[SourceFile]:
    data = open(ASSET, "rb").read()
    return [SourceFile(name=f"doc{index}.md", data=data) for index in range(count)]


def test_a_single_file_is_parsed_extracted_and_committed(tmp_path):
    graph = InMemoryGraphStore()
    store = FileDocumentStore(tmp_path)

    result = ingest_batch(_files(1), parse=_parse, pipeline=_pipeline(),
                          documents=store, graph=graph)

    assert len(result.succeeded) == 1
    assert result.outcomes[0].stage is Stage.DONE
    assert result.outcomes[0].fact_count == 1
    assert list(store.list_ids())
    assert graph.stats().facts == 1


def test_identical_files_deduplicate_by_content(tmp_path):
    """Content addressing means the same bytes twice is one document."""
    graph = InMemoryGraphStore()
    store = FileDocumentStore(tmp_path)

    result = ingest_batch(_files(3), parse=_parse, pipeline=_pipeline(),
                          documents=store, graph=graph)

    assert len(result.succeeded) == 3
    assert len(list(store.list_ids())) == 1
    assert graph.stats().facts == 1


def test_one_bad_file_does_not_abort_the_batch(tmp_path):
    files = [
        SourceFile(name="broken.pdf", data=b"%PDF-1.7 not really a pdf"),
        *_files(1),
    ]
    graph = InMemoryGraphStore()

    result = ingest_batch(files, parse=_parse, pipeline=_pipeline(),
                          documents=FileDocumentStore(tmp_path), graph=graph)

    assert len(result.failed) == 1
    assert len(result.succeeded) == 1
    assert result.failed[0].stage is Stage.FAILED
    assert "could not parse" in result.failed[0].error


def test_stage_updates_are_published_in_order(tmp_path):
    seen: list[tuple[str, Stage]] = []

    ingest_batch(
        _files(1), parse=_parse, pipeline=_pipeline(),
        documents=FileDocumentStore(tmp_path), graph=InMemoryGraphStore(),
        on_update=lambda outcome: seen.append((outcome.name, outcome.stage)),
    )

    stages = [stage for _name, stage in seen]
    assert stages[0] is Stage.PARSING
    assert Stage.EXTRACTING in stages
    assert Stage.COMMITTING in stages
    assert stages[-1] is Stage.DONE


def test_commit_can_be_withheld_for_review(tmp_path):
    graph = InMemoryGraphStore()
    store = FileDocumentStore(tmp_path)

    result = ingest_batch(_files(1), parse=_parse, pipeline=_pipeline(),
                          documents=store, graph=graph, commit=False)

    assert result.outcomes[0].stage is Stage.DONE
    assert result.outcomes[0].upsert is not None
    assert graph.stats().facts == 0
    assert list(store.list_ids()) == []


def test_totals_sum_across_committed_files(tmp_path):
    result = ingest_batch(_files(2), parse=_parse, pipeline=_pipeline(),
                          documents=FileDocumentStore(tmp_path), graph=InMemoryGraphStore())
    assert result.totals().facts == 2


def test_an_empty_batch_is_a_no_op(tmp_path):
    result = ingest_batch([], parse=_parse, pipeline=_pipeline(),
                          documents=FileDocumentStore(tmp_path), graph=InMemoryGraphStore())
    assert result.outcomes == []
    assert result.totals().facts == 0


def test_a_graph_failure_is_confined_to_its_file(tmp_path):
    class BrokenGraph(InMemoryGraphStore):
        def apply(self, upsert):
            raise RuntimeError("neo4j write refused")

    result = ingest_batch(_files(1), parse=_parse, pipeline=_pipeline(),
                          documents=FileDocumentStore(tmp_path), graph=BrokenGraph())

    assert result.outcomes[0].stage is Stage.FAILED
    assert "neo4j write refused" in result.outcomes[0].error
