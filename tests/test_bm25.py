from math import log

import pytest

from ris.data import Doc
from ris.index import InvertedIndex
from ris.retrievers import BM25Retriever, Result, Retriever


@pytest.fixture
def simple_preprocess(monkeypatch):
    """Aísla la fórmula del tokenizador; Porter se comprueba aparte."""

    def preprocess(text, *, stem=False):
        return text.lower().split()

    monkeypatch.setattr("ris.index.preprocess", preprocess)
    monkeypatch.setattr("ris.retrievers.bm25.preprocess", preprocess)


@pytest.fixture
def retriever(simple_preprocess):
    corpus = {
        "d1": Doc(title="apple", text="apple banana"),
        "d2": Doc(title="", text="apple banana banana banana"),
        "d3": Doc(title="", text="banana carrot"),
        "d4": Doc(title="", text=""),
    }
    return BM25Retriever(InvertedIndex.build(corpus))


def test_hand_calculated_scores(retriever):
    # N=4, longitudes=(3,4,2,0), avgdl=9/4 (incluye el documento vacío).
    # df(apple)=2, df(banana)=3. IDF: ln(2) y ln(10/7).
    # K(d)=1.5*(.25+.75*dl/(9/4)) = 3/8+dl/2.
    # K(d1)=15/8; K(d2)=19/8; K(d3)=11/8.
    # Para apple: factores d1=40/31, d2=20/27.
    # Para banana: factores d1=20/23, d2=60/43, d3=20/19.
    expected = {
        "d1": log(2) * 40 / 31 + log(10 / 7) * 20 / 23,
        "d2": log(2) * 20 / 27 + log(10 / 7) * 60 / 43,
        "d3": log(10 / 7) * 20 / 19,
    }
    model: Retriever = retriever
    results = model.search("apple banana")
    assert [r.doc_id for r in results] == ["d1", "d2", "d3"]
    assert {r.doc_id: r.score for r in results} == pytest.approx(expected)
    assert retriever.avgdl == pytest.approx(9 / 4)


def test_unique_query_terms_and_query_permutation(retriever):
    assert retriever.search("apple apple banana banana") == retriever.search(
        "banana apple"
    )
    assert retriever.search("apple apple apple") == retriever.search("apple")


def test_unknown_terms_are_ignored(retriever):
    assert retriever.search("apple dragon dragon") == retriever.search("apple")
    assert retriever.search("dragon unicorn") == []


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query(retriever, query):
    assert retriever.search(query) == []


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises_even_for_empty_query(retriever, k):
    with pytest.raises(ValueError, match="positivo"):
        retriever.search("", k=k)


def test_unique_descending_ranking_and_top_k(retriever):
    full = retriever.search("apple banana carrot", k=50)
    assert len(full) == 3
    assert len({r.doc_id for r in full}) == 3
    assert all(r.score > 0 for r in full)
    assert full == sorted(full, key=lambda r: (-r.score, r.doc_id))
    assert retriever.search("apple banana carrot", k=2) == full[:2]


def test_exact_ties_use_lexicographic_doc_id(simple_preprocess):
    corpus = {
        "d2": Doc(title="", text="apple banana"),
        "d10": Doc(title="", text="apple banana"),
        "empty": Doc(title="", text=""),
    }
    results = BM25Retriever(InvertedIndex.build(corpus)).search("apple")
    # N=3, df=2, avgdl=4/3, dl=2; tf normalizado=40/49.
    assert results == [
        Result("d10", pytest.approx(log(8 / 5) * 40 / 49)),
        Result("d2", pytest.approx(log(8 / 5) * 40 / 49)),
    ]


def test_term_in_every_document_keeps_positive_idf(simple_preprocess):
    corpus = {
        "a": Doc(title="", text="common"),
        "b": Doc(title="", text="common"),
    }
    # df=N=2: ln(1+.5/2.5)=ln(6/5), no se anula ni es negativo.
    results = BM25Retriever(InvertedIndex.build(corpus)).search("common")
    assert [r.doc_id for r in results] == ["a", "b"]
    assert [r.score for r in results] == pytest.approx([log(6 / 5)] * 2)


def test_length_normalization_favors_shorter_document(simple_preprocess):
    corpus = {
        "short": Doc(title="", text="apple"),
        "long": Doc(title="", text="apple banana banana banana banana"),
    }
    results = BM25Retriever(InvertedIndex.build(corpus)).search("apple")
    # N=2, avgdl=3. tf=1: factor corto=10/7; largo=10/13.
    assert [r.doc_id for r in results] == ["short", "long"]
    assert [r.score for r in results] == pytest.approx(
        [log(6 / 5) * 10 / 7, log(6 / 5) * 10 / 13]
    )


def test_tf_saturates_for_equal_lengths(simple_preprocess):
    corpus = {
        "once": Doc(title="", text="apple banana banana"),
        "twice": Doc(title="", text="apple apple banana"),
    }
    results = BM25Retriever(InvertedIndex.build(corpus)).search("apple")
    # dl=avgdl: factores tf=2 -> 10/7; tf=1 -> 1.
    assert [r.doc_id for r in results] == ["twice", "once"]
    assert [r.score for r in results] == pytest.approx(
        [log(6 / 5) * 10 / 7, log(6 / 5)]
    )


@pytest.mark.parametrize("with_document", [False, True])
def test_empty_index_and_all_empty_documents(simple_preprocess, with_document):
    corpus = {"empty": Doc(title="", text="")} if with_document else {}
    model = BM25Retriever(InvertedIndex.build(corpus))
    assert model.avgdl == 0
    assert model.search("apple") == []
    with pytest.raises(ValueError, match="positivo"):
        model.search("apple", k=0)


def test_uses_existing_persisted_index(retriever, tmp_path):
    path = tmp_path / "index.json"
    retriever.index.save(path)
    restored = BM25Retriever(InvertedIndex.load(path, expected_stem=False))
    assert restored.search("apple banana") == retriever.search("apple banana")


@pytest.mark.parametrize("index_stem,query_stem", [(True, False), (False, True)])
def test_rejects_mismatched_stemming(simple_preprocess, index_stem, query_stem):
    index = InvertedIndex.build({}, stem=index_stem)
    with pytest.raises(ValueError, match="stemming distinta"):
        BM25Retriever(index, stem=query_stem)


@pytest.mark.parametrize("stem", [False, True])
def test_real_preprocess_with_and_without_porter(stem):
    corpus = {
        "banks": Doc(title="The banks", text="are lending"),
        "bank": Doc(title="", text="A river bank"),
        "empty": Doc(title="", text=""),
    }
    model = BM25Retriever(InvertedIndex.build(corpus, stem=stem), stem=stem)
    results = model.search("The BANKS")

    # N=3, dl=2 para ambos, avgdl=4/3, factor=40/49.
    if stem:
        assert [r.doc_id for r in results] == ["bank", "banks"]
        assert [r.score for r in results] == pytest.approx([log(8 / 5) * 40 / 49] * 2)
    else:
        assert [r.doc_id for r in results] == ["banks"]
        assert results[0].score == pytest.approx(log(8 / 3) * 40 / 49)

    assert model.search("the and of") == []