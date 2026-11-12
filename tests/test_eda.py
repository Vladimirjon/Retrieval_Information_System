"""Cálculos de #4 con ejemplos manuales; sin descargar FiQA ni recursos NLTK."""

import json

import pytest

from ris import eda
from ris.data import Doc


def test_population_statistics_and_linear_percentiles():
    stats = eda.describe([0, 1, 1, 6])
    assert stats["count"] == 4
    assert stats["total"] == 8
    assert stats["mean"] == 2
    assert stats["median"] == 1
    assert stats["std_population"] == pytest.approx((4 + 1 + 1 + 16) ** 0.5 / 2)
    assert stats["percentiles"]["25"] == 0.75
    assert stats["percentiles"]["75"] == 2.25
    assert stats["distribution"] == {"0": 1, "1": 2, "6": 1}
    assert stats["zeros"] == 1
    with pytest.raises(ValueError):
        eda.describe([-1])


def test_empty_statistics_and_json(tmp_path):
    summary = eda.describe([])
    assert summary["mean"] is None
    assert summary["min"] is None
    assert summary["distribution"] == {}
    assert eda.percentage(0, 0) == 0
    path = tmp_path / "nested" / "summary.json"
    eda.write_summary(summary, path)
    assert json.loads(path.read_text(encoding="utf-8")) == summary


def test_tf_collection_frequency_and_df_are_distinct():
    profile = eda.Profile()
    profile.add(["a", "a", "b"])
    profile.add(["a", "c"])
    profile.add([])
    stats = profile.summarize(["d1", "d2", "d3"])
    assert profile.cf == {"a": 3, "b": 1, "c": 1}
    assert profile.df == {"a": 2, "b": 1, "c": 1}
    assert stats["collection_frequency_total"] == stats["length"]["total"] == 5
    assert stats["vocabulary_size"] == 3
    assert stats["hapax_cf1"] == stats["single_document_df1"] == 2
    assert stats["top_by_cf"][0] == {"term": "a", "cf": 3, "df": 2}
    assert stats["top_by_cf"][1]["term"] == "b"  # desempate estable
    assert stats["length"]["zeros"] == 1


@pytest.fixture
def small_analysis(monkeypatch):
    monkeypatch.setattr(eda, "word_tokenize", str.split)

    def fixture_preprocess(text, *, stem=False):
        tokens = [t for t in text.lower().split() if t.isalnum() and t != "the"]
        return (
            ["invest" if t in {"invest", "invests"} else t for t in tokens]
            if stem
            else tokens
        )

    monkeypatch.setattr(eda, "preprocess", fixture_preprocess)
    corpus = {
        "a": Doc("", "The invest invests"),
        "b": Doc("", "invest"),
        "c": Doc("", ""),
    }
    queries = {"q1": "The invest unknown", "q2": "", "q3": "invests"}
    qrels = {"q1": {"a": 1, "b": 1}, "q2": {"c": 0}, "q3": {"c": 1}}
    return corpus, queries, qrels


def test_analysis_empty_text_oov_stemming_and_oracle_bounds(small_analysis):
    summary = eda.analyze(*small_analysis)
    assert summary["counts"] == {"documents": 3, "test_queries": 3}
    assert summary["empty_fields"]["title"]["empty_exact_pct_documents"] == 100
    assert summary["documents"]["lexical"]["vocabulary_size"] == 2
    assert summary["documents"]["porter"]["vocabulary_size"] == 1
    assert summary["reductions_documents"]["vocabulary_porter_pct"] == 50
    assert summary["queries_test"]["lexical"]["oov_tokens"] == 1
    assert summary["queries_test"]["lexical"]["length"]["zeros"] == 1
    rel = summary["relevance_test"]
    assert rel["distribution"] == {"0": 1, "1": 1, "2": 1}
    assert rel["judgment_pairs"] == 4
    assert rel["total"] == rel["unique_relevant_documents"] == 3
    assert rel["macro_oracle_ceiling_p10"] == pytest.approx(0.1)
    assert rel["macro_oracle_ceiling_r10"] == pytest.approx(2 / 3)
    assert rel["macro_oracle_ceiling_ap1000"] == pytest.approx(2 / 3)
    assert rel["relevant_documents_empty_original"] == 1
    assert rel["empty_relevant_examples"] == [{"doc_id": "c", "qids": ["q3"]}]


def test_oracle_with_more_than_k_relevant_documents(monkeypatch):
    monkeypatch.setattr(eda, "word_tokenize", str.split)
    monkeypatch.setattr(eda, "preprocess", lambda text, *, stem=False: text.split())
    corpus = {str(i): Doc("", "a") for i in range(12)}
    summary = eda.analyze(corpus, {"q": "a"}, {"q": dict.fromkeys(corpus, 1)})
    assert summary["relevance_test"]["macro_oracle_ceiling_r10"] == pytest.approx(
        10 / 12
    )
    assert summary["relevance_test"]["macro_oracle_ceiling_p10"] == 1


def test_relevance_pairs_differ_from_unique_documents(small_analysis):
    corpus, queries, qrels = small_analysis
    summary = eda.analyze(corpus, queries, {**qrels, "q3": {"a": 1, "c": 1}})
    rel = summary["relevance_test"]
    assert rel["judgment_pairs"] == 5  # Incluye un juicio con score cero.
    assert rel["total"] == 4  # Pares relevantes: a cuenta para dos consultas.
    assert rel["unique_relevant_documents"] == 3


def test_reject_inconsistent_ids_and_references(small_analysis):
    corpus, queries, qrels = small_analysis
    with pytest.raises(ValueError, match="coincidir"):
        eda.analyze(corpus, {}, qrels)
    with pytest.raises(ValueError, match="ausentes"):
        eda.analyze(corpus, queries, {**qrels, "q1": {"missing": 1}})
    with pytest.raises(ValueError, match="str"):
        eda.analyze({1: Doc("", "")}, {}, {})
    with pytest.raises(ValueError, match="enteros"):
        eda.analyze(corpus, queries, {**qrels, "q1": {"a": "1"}})


def test_empty_analysis():
    summary = eda.analyze({}, {}, {})
    assert summary["relevance_test"]["macro_oracle_ceiling_p10"] == 0
    assert summary["documents"]["lexical"]["vocabulary_size"] == 0
