import pytest

from app.retrieval.bm25 import tokenize
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.hybrid import QueryValidationError, preprocess_query
from app.retrieval.vector_store import ChromaIndex, EmbeddingModelMismatchError
from app.schemas import SearchFilters
from tests.conftest import CHARTER_MD, CHARTER_META, CHRONICLE_HTML, CHRONICLE_META, write_doc


@pytest.fixture
def indexed(settings, services):
    write_doc(settings.sources_dir, "charter.md", CHARTER_MD, CHARTER_META)
    write_doc(settings.sources_dir, "chronicle.html", CHRONICLE_HTML, CHRONICLE_META)
    services.pipeline().run()
    return services


def test_hybrid_finds_relevant_chunk(indexed):
    res = indexed.retriever.retrieve("Who abolished the salt levy?", top_k=3)
    assert res[0].document_id == "testland_chronicle"
    assert "salt levy" in res[0].text
    assert res[0].fused_score > 0 and res[0].bm25_rank is not None


def test_keyword_and_vector_modes(indexed):
    kw = indexed.retriever.retrieve("High Court review", top_k=3, mode="keyword")
    vec = indexed.retriever.retrieve("High Court review", top_k=3, mode="vector")
    assert kw and all(c.vector_rank is None for c in kw)
    assert vec and all(c.bm25_rank is None for c in vec)
    assert kw[0].document_id == "testland_charter"


def test_period_and_source_type_filters(indexed):
    r = indexed.retriever
    only_ancient = r.retrieve("court reform levy", SearchFilters(periods=["ancient"]), top_k=10)
    assert only_ancient and {c.document_id for c in only_ancient} == {"testland_charter"}
    only_scholar = r.retrieve("court reform levy", SearchFilters(source_types=["scholarship"]), top_k=10)
    assert {c.document_id for c in only_scholar} == {"testland_chronicle"}


def test_year_range_filter_overlap(indexed):
    r = indexed.retriever
    hits = r.retrieve("court reform levy", SearchFilters(year_from=1233, year_to=1235), top_k=10)
    assert {c.document_id for c in hits} == {"testland_chronicle"}
    none = r.retrieve("court reform levy", SearchFilters(year_from=1500), top_k=10)
    assert none == []


def test_python_only_filters_person_topic(indexed):
    r = indexed.retriever
    hits = r.retrieve("reform", SearchFilters(person="belan"), top_k=10)
    assert hits and {c.document_id for c in hits} == {"testland_chronicle"}
    hits = r.retrieve("assembly", SearchFilters(topic="courts"), top_k=10)
    assert {c.document_id for c in hits} == {"testland_charter"}


def test_published_date_filter(indexed):
    hits = indexed.retriever.retrieve("reform assembly", SearchFilters(published_from="1235"), top_k=10)
    assert {c.document_id for c in hits} == {"testland_chronicle"}


def test_results_have_no_duplicate_chunks(indexed):
    res = indexed.retriever.retrieve("Testland", top_k=20)
    assert len({c.chunk_id for c in res}) == len(res)
    assert len({c.text for c in res}) == len(res)


def test_query_validation():
    assert preprocess_query("  What   is\tthis? ") == "What is this?"
    for bad in ["", "ab", "???", "x" * 1001]:
        with pytest.raises(QueryValidationError):
            preprocess_query(bad)


def test_tokenizer_keeps_devanagari_words_intact():
    assert tokenize("भारत का संविधान") == ["भारत", "का", "संविधान"]
    assert tokenize("The Constitution of India") == ["constitution", "india"]


def test_embedding_model_mismatch_detected(settings, indexed):
    with pytest.raises(EmbeddingModelMismatchError):
        ChromaIndex(settings.index_dir, settings.collection_name, HashingEmbedder(name="other-model"))


def test_empty_index_returns_nothing(services):
    assert services.retriever.retrieve("anything at all") == []
