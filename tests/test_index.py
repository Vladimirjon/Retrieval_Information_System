import json

import pytest

from ris.data import Doc
from ris.index import InvertedIndex


@pytest.fixture
def corpus():
    return {
        "d1": Doc(title="bank", text="bank loan"),
        "d2": Doc(title="", text="bank"),
        "d3": Doc(title="", text=""),
    }


@pytest.fixture
def simple_preprocess(monkeypatch):
    """Sustituye el preprocesamiento para probar el índice de forma aislada."""

    def preprocess(text, *, stem=False):
        return text.lower().split()

    monkeypatch.setattr("ris.index.preprocess", preprocess)


def test_build_counts_frequencies_and_empty_documents(corpus, simple_preprocess):
    index = InvertedIndex.build(corpus)

    assert index.postings == {
        "bank": {"d1": 2, "d2": 1},
        "loan": {"d1": 1},
    }
    assert index.df == {"bank": 2, "loan": 1}
    assert index.doc_lengths == {"d1": 3, "d2": 1, "d3": 0}
    assert index.N == 3
    assert index.stem is False


def test_build_reports_progress(corpus, simple_preprocess):
    progress = []

    InvertedIndex.build(
        corpus,
        progress_callback=lambda processed, total, vocabulary: progress.append(
            (processed, total, vocabulary)
        ),
    )

    assert progress == [(1, 3, 2), (2, 3, 2), (3, 3, 2)]


def test_save_and_load_preserve_index(corpus, simple_preprocess, tmp_path):
    index = InvertedIndex.build(corpus, stem=True)
    path = tmp_path / "index.json"

    index.save(path)
    loaded = InvertedIndex.load(path, expected_stem=True)

    assert loaded.postings == index.postings
    assert loaded.df == index.df
    assert loaded.doc_lengths == index.doc_lengths
    assert loaded.N == index.N
    assert loaded.stem == index.stem


def test_load_rejects_different_stemming(corpus, simple_preprocess, tmp_path):
    index = InvertedIndex.build(corpus, stem=False)
    path = tmp_path / "index.json"
    index.save(path)

    with pytest.raises(ValueError, match="stemming distinta"):
        InvertedIndex.load(path, expected_stem=True)


def test_load_rejects_unknown_format(corpus, simple_preprocess, tmp_path):
    index = InvertedIndex.build(corpus)
    path = tmp_path / "index.json"
    index.save(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["format_version"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Versión de formato"):
        InvertedIndex.load(path)