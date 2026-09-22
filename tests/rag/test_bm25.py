from __future__ import annotations

from app.parsing.base import InMemoryBlobSink, parse_document
from app.rag.bm25 import Bm25Index, Passage, tokenize


def _passage(element_id: str, text: str, doc_id: str = "d") -> Passage:
    return Passage(
        doc_id=doc_id, element_id=element_id, page_number=1, element_type="paragraph",
        text=text, x0=0, y0=0, x1=10, y1=10, source_name="doc.md",
    )


CORPUS = [
    _passage("p1e001", "The Payment API depends on Redis for caching."),
    _passage("p1e002", "The Payments Team owns the Payment API service."),
    _passage("p1e003", "Incident INC-2391 affected the Payment API in production."),
    _passage("p1e004", "The ledger is stored in PostgreSQL for durability."),
    _passage("p1e005", "Error code PAYMENT_502 indicates an upstream timeout."),
]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("The Payment API, really!") == ["the", "payment", "api", "really"]


def test_tokenize_keeps_an_identifier_whole_and_in_parts():
    tokens = tokenize("Incident INC-2391 happened")
    assert "inc-2391" in tokens
    assert "inc" in tokens
    assert "2391" in tokens


def test_tokenize_handles_underscored_error_codes():
    tokens = tokenize("code PAYMENT_502 raised")
    assert "payment_502" in tokens
    assert "502" in tokens


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("   ") == []


def test_an_exact_identifier_ranks_its_own_element_first():
    top, score = Bm25Index(CORPUS).search("INC-2391")[0]
    assert top.element_id == "p1e003"
    assert score > 0


def test_an_underscored_error_code_is_found():
    assert Bm25Index(CORPUS).search("PAYMENT_502")[0][0].element_id == "p1e005"


def test_a_common_term_ranks_below_a_rare_one():
    ranked = [p.element_id for p, _ in Bm25Index(CORPUS).search("Payment API Redis")]
    assert ranked[0] == "p1e001"


def test_search_respects_the_limit():
    assert len(Bm25Index(CORPUS).search("payment", limit=2)) == 2


def test_a_query_matching_nothing_returns_nothing():
    assert Bm25Index(CORPUS).search("kubernetes helm chart") == []


def test_an_empty_query_returns_nothing():
    assert Bm25Index(CORPUS).search("   ") == []


def test_an_empty_index_searches_safely():
    assert Bm25Index([]).search("anything") == []


def test_passages_are_addressable_by_global_id():
    assert Bm25Index(CORPUS).passages_by_id["d:p1e003"].element_id == "p1e003"


def test_global_id_is_document_scoped():
    assert _passage("p1e001", "x", doc_id="abc").global_id == "abc:p1e001"


def test_scores_are_ordered_descending():
    scores = [score for _, score in Bm25Index(CORPUS).search("Payment API")]
    assert scores == sorted(scores, reverse=True)


def test_an_index_can_be_built_from_parsed_documents():
    doc = parse_document("app/assets/payment_system.md", blobs=InMemoryBlobSink())
    index = Bm25Index.from_documents([doc])

    hit, _score = index.search("Redis connection timeouts")[0]
    assert hit.doc_id == doc.id
    assert "Redis" in hit.text


def test_an_index_built_from_a_store_round_trips(tmp_path):
    from app.ingest import ingest
    from app.storage.filestore import FileDocumentStore

    store = FileDocumentStore(tmp_path)
    ingest("app/assets/payment_system.md", store)

    assert Bm25Index.from_store(store).search("Payments Team")
