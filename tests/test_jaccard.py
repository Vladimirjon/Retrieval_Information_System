import pytest

from ris.data import Doc
from ris.index import InvertedIndex
from ris.retrievers import JaccardRetriever, Result


@pytest.fixture
def simple_preprocess(monkeypatch):
    """Sustituye el preprocesamiento para calcular los scores a mano."""

    def preprocess(text, *, stem=False):
        return text.lower().split()

    monkeypatch.setattr("ris.index.preprocess", preprocess)
    monkeypatch.setattr("ris.retrievers.jaccard.preprocess", preprocess)


@pytest.fixture
def retriever(simple_preprocess):
    corpus = {
        "d1": Doc(title="", text="gato perro"),
        "d2": Doc(title="gato", text="gato raton"),
        "d3": Doc(title="", text="perro pez loro casa"),
        "d4": Doc(title="", text=""),
    }
    return JaccardRetriever(InvertedIndex.build(corpus), stem=False)


def scores(results):
    return {result.doc_id: result.score for result in results}


def test_hand_calculated_scores(retriever):
    # Q = {gato, perro}
    # d1 = {gato, perro}:            2 / 2
    # d2 = {gato, raton}:            1 / (2 + 2 - 1) = 1/3
    # d3 = {perro, pez, loro, casa}: 1 / (2 + 4 - 1) = 1/5
    results = retriever.search("gato perro")

    assert [result.doc_id for result in results] == ["d1", "d2", "d3"]
    assert results[0] == Result(doc_id="d1", score=1.0)
    assert results[1].score == pytest.approx(1 / 3)
    assert results[2].score == pytest.approx(1 / 5)


def test_document_repetitions_do_not_change_its_size(retriever):
    # d2 tiene 3 tokens pero solo 2 términos únicos: 1 / (1 + 2 - 1)
    assert scores(retriever.search("raton"))["d2"] == pytest.approx(1 / 2)


def test_query_repetitions_have_no_effect(retriever):
    assert retriever.search("gato gato") == retriever.search("gato")
    assert scores(retriever.search("gato gato")) == {
        "d1": pytest.approx(1 / 2),
        "d2": pytest.approx(1 / 2),
    }


def test_unknown_terms_only_increase_query_size(retriever):
    # Q = {gato, dragon}: d1 y d2 obtienen 1 / (2 + 2 - 1)
    assert scores(retriever.search("gato dragon")) == {
        "d1": pytest.approx(1 / 3),
        "d2": pytest.approx(1 / 3),
    }
    assert retriever.search("dragon unicornio") == []


def test_empty_query_and_empty_documents(retriever):
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert "d4" not in scores(retriever.search("gato perro pez loro casa raton"))


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(retriever, k):
    with pytest.raises(ValueError):
        retriever.search("gato", k=k)


def test_k_limits_results(retriever):
    assert [r.doc_id for r in retriever.search("gato perro", k=1)] == ["d1"]
    assert [r.doc_id for r in retriever.search("gato perro", k=2)] == ["d1", "d2"]
    assert len(retriever.search("gato perro", k=50)) == 3


def test_ties_are_sorted_by_doc_id(simple_preprocess):
    corpus = {
        "doc_b": Doc(title="", text="sol luna"),
        "doc_c": Doc(title="", text="sol"),
        "doc_a": Doc(title="", text="sol mar"),
    }
    results = JaccardRetriever(InvertedIndex.build(corpus)).search("sol")

    assert results == [
        Result(doc_id="doc_c", score=1.0),
        Result(doc_id="doc_a", score=0.5),
        Result(doc_id="doc_b", score=0.5),
    ]


def test_results_are_unique(retriever):
    doc_ids = [r.doc_id for r in retriever.search("gato perro raton pez")]
    assert len(doc_ids) == len(set(doc_ids))


def test_rejects_index_with_different_stemming(simple_preprocess):
    index = InvertedIndex.build({"d1": Doc(title="", text="gato")}, stem=True)

    with pytest.raises(ValueError, match="stemming distinta"):
        JaccardRetriever(index, stem=False)


def test_uses_real_preprocess_with_stemming():
    corpus = {
        "d1": Doc(title="", text="The banks are lending"),
        "d2": Doc(title="", text="A river bank"),
    }
    retriever = JaccardRetriever(InvertedIndex.build(corpus, stem=True), stem=True)

    # Con Porter: Q = {bank}, d1 = {bank, lend}, d2 = {river, bank}
    assert scores(retriever.search("bank")) == {
        "d1": pytest.approx(1 / 2),
        "d2": pytest.approx(1 / 2),
    }
    assert retriever.search("the and of") == []
