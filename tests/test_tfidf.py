from math import log, sqrt

import pytest

from ris.data import Doc
from ris.index import InvertedIndex
from ris.retrievers import Result, Retriever, TfidfRetriever


@pytest.fixture
def simple_preprocess(monkeypatch):
    """Aísla las fórmulas del tokenizador; Porter se comprueba aparte."""

    def preprocess(text, *, stem=False):
        return text.lower().split()

    monkeypatch.setattr("ris.index.preprocess", preprocess)
    monkeypatch.setattr("ris.retrievers.tfidf.preprocess", preprocess)


@pytest.fixture
def retriever(simple_preprocess):
    corpus = {
        "d1": Doc(title="apple", text="apple banana"),
        "d2": Doc(title="", text="apple banana banana banana"),
        "d3": Doc(title="", text="banana carrot"),
        "d4": Doc(title="", text=""),
    }
    return TfidfRetriever(InvertedIndex.build(corpus))


def test_hand_calculated_cosines_with_log_tf_and_unsmoothed_idf(retriever):
    # N=4 (incluye d4 vacío); df(apple)=2, df(banana)=3, df(carrot)=1.
    a, b, c = log(2), log(4 / 3), log(4)
    # Q="apple apple apple banana": ((1+ln(3))*a, b, 0).
    q_apple = (1 + log(3)) * a
    q_norm = sqrt(q_apple**2 + b**2)
    # d1=((1+ln(2))*a,b,0); d2=(a,(1+ln(3))*b,0); d3=(0,b,c).
    d1_apple = (1 + log(2)) * a
    d2_banana = (1 + log(3)) * b
    expected = {
        "d1": (q_apple * d1_apple + b**2)
        / (q_norm * sqrt(d1_apple**2 + b**2)),
        "d2": (q_apple * a + b * d2_banana)
        / (q_norm * sqrt(a**2 + d2_banana**2)),
        "d3": b**2 / (q_norm * sqrt(b**2 + c**2)),
    }

    model: Retriever = retriever
    results = model.search("apple apple apple banana")
    assert [r.doc_id for r in results] == ["d1", "d2", "d3"]
    assert {r.doc_id: r.score for r in results} == pytest.approx(expected)


def test_document_norm_includes_terms_absent_from_query(retriever):
    # Consulta unidimensional: d3 conserva carrot en su norma completa.
    b, c = log(4 / 3), log(4)
    scores = {r.doc_id: r.score for r in retriever.search("banana")}
    assert scores["d3"] == pytest.approx(b / sqrt(b**2 + c**2))


def test_repeated_query_terms_change_relative_weights(retriever):
    single = {r.doc_id: r.score for r in retriever.search("apple banana")}
    repeated = {r.doc_id: r.score for r in retriever.search("apple apple apple banana")}
    assert repeated["d1"] != pytest.approx(single["d1"])
    repeated_single = retriever.search("apple apple")
    single_term = retriever.search("apple")
    assert [r.doc_id for r in repeated_single] == [r.doc_id for r in single_term]
    assert [r.score for r in repeated_single] == pytest.approx(
        [r.score for r in single_term]
    )


def test_unknown_terms_have_no_coordinates_in_corpus_vocabulary(retriever):
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
    assert all(0 < r.score <= 1 for r in full)
    assert full == sorted(full, key=lambda r: (-r.score, r.doc_id))
    assert retriever.search("apple banana carrot", k=2) == full[:2]


def test_exact_ties_use_lexicographic_doc_id(simple_preprocess):
    corpus = {
        "d2": Doc(title="", text="apple banana"),
        "d10": Doc(title="", text="apple banana"),
        "empty": Doc(title="", text=""),
    }
    results = TfidfRetriever(InvertedIndex.build(corpus)).search("apple")
    assert [r.doc_id for r in results] == ["d10", "d2"]
    assert all(isinstance(r, Result) for r in results)
    assert [r.score for r in results] == pytest.approx([1 / sqrt(2)] * 2)


def test_common_terms_have_zero_idf_and_zero_norm_documents_are_omitted(
    simple_preprocess,
):
    corpus = {
        "zero": Doc(title="", text="common common"),
        "match": Doc(title="", text="common rare rare"),
    }
    model = TfidfRetriever(InvertedIndex.build(corpus))
    assert model.search("common") == []
    assert model.search("common unknown") == []
    assert model.search("rare common") == [Result(doc_id="match", score=1.0)]


def test_empty_index_and_all_empty_documents(simple_preprocess):
    for corpus in ({}, {"empty": Doc(title="", text="")}):
        assert TfidfRetriever(InvertedIndex.build(corpus)).search("apple") == []


def test_classroom_corpus_with_project_weighting(simple_preprocess):
    # Corpus docente: Modelo_vectorial (1).pdf, pp. 4 y 20.
    # Scores recalculados con ln y TF sublineal del acuerdo del equipo;
    # no son los scores de la variante numérica del ejemplo del profesor.
    texts = {
        "D1": "perro duerme casa",
        "D2": "gato duerme casa",
        "D3": "perro juega gato",
        "D4": "gato juega juega jardín",
        "D5": "pájaro canta jardín",
    }
    corpus = {doc_id: Doc(title="", text=text) for doc_id, text in texts.items()}
    results = TfidfRetriever(InvertedIndex.build(corpus)).search("gato juega")
    assert [r.doc_id for r in results] == ["D4", "D3", "D2"]
    assert [r.score for r in results] == pytest.approx(
        [0.8563619524082989, 0.7531594359224856, 0.17857875162038853]
    )


def test_uses_existing_persisted_index(retriever, tmp_path):
    path = tmp_path / "index.json"
    retriever.index.save(path)
    restored = TfidfRetriever(InvertedIndex.load(path, expected_stem=False))
    assert restored.search("apple banana") == retriever.search("apple banana")


@pytest.mark.parametrize("index_stem,query_stem", [(True, False), (False, True)])
def test_rejects_mismatched_stemming(simple_preprocess, index_stem, query_stem):
    index = InvertedIndex.build({}, stem=index_stem)
    with pytest.raises(ValueError, match="stemming distinta"):
        TfidfRetriever(index, stem=query_stem)


@pytest.mark.parametrize("stem", [False, True])
def test_real_preprocess_with_and_without_porter(stem):
    corpus = {
        "banks": Doc(title="The banks", text="are lending"),
        "bank": Doc(title="", text="A river bank"),
        "empty": Doc(title="", text=""),
    }
    model = TfidfRetriever(InvertedIndex.build(corpus, stem=stem), stem=stem)
    results = model.search("The BANKS")
    if stem:
        # bank aparece en 2/3 documentos; lend y river solo en 1/3.
        bank_idf, other_idf = log(3 / 2), log(3)
        expected = bank_idf / sqrt(bank_idf**2 + other_idf**2)
        assert [r.doc_id for r in results] == ["bank", "banks"]
        assert [r.score for r in results] == pytest.approx([expected] * 2)
    else:
        assert [r.doc_id for r in results] == ["banks"]
        assert results[0].score == pytest.approx(1 / sqrt(2))
    assert model.search("the and of") == []
