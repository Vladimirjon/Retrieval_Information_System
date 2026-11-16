import threading

import pytest

from ris.data import Doc
from ris.engine import (
    MODELS,
    ArtifactsMissingError,
    EmptyQueryError,
    ModelUnavailableError,
    SearchEngine,
)
from ris.index import InvertedIndex
from ris.retrievers import Result

CORPUS = {
    "d1": Doc(title="", text="gato perro"),
    "d2": Doc(title="Gatos", text="gato raton"),
    "d3": Doc(title="", text="perro pez loro casa"),
    "d4": Doc(title="", text=""),
}


@pytest.fixture(autouse=True)
def simple_preprocess(monkeypatch):
    """Sustituye el preprocesamiento: la prueba no depende de recursos NLTK."""

    def preprocess(text, *, stem=False):
        return text.lower().split()

    for module in ("index", "retrievers.jaccard", "retrievers.tfidf", "retrievers.bm25"):
        monkeypatch.setattr(f"ris.{module}.preprocess", preprocess, raising=False)


class FakeDense:
    """Doble del recuperador denso: ranking fijo y contador de cargas."""

    def __init__(self, doc_ids=("d1", "d2", "d3", "d4")):
        self.doc_ids = tuple(doc_ids)
        self.queries = []

    def search(self, query, k=10):
        self.queries.append(query)
        if not query.strip():
            return []
        order = sorted(self.doc_ids, reverse=True)
        return [Result(doc_id, 1.0 - 0.1 * i) for i, doc_id in enumerate(order[:k])]


def write_artifacts(directory, *, stem_too=False, dense=True):
    directory.mkdir(parents=True, exist_ok=True)
    InvertedIndex.build(CORPUS, stem=False).save(directory / "index_no_stem.json")
    if stem_too:
        InvertedIndex.build(CORPUS, stem=True).save(directory / "index_stem.json")
    if dense:
        (directory / "dense").mkdir()
        for name in ("metadata.json", "embeddings.npy", "index.faiss"):
            (directory / "dense" / name).write_bytes(b"x")


@pytest.fixture
def engine(tmp_path):
    write_artifacts(tmp_path, stem_too=True)
    loads = {"corpus": 0, "dense": 0}

    def corpus_loader():
        loads["corpus"] += 1
        return CORPUS

    def dense_loader(path):
        loads["dense"] += 1
        return FakeDense()

    engine = SearchEngine(tmp_path, corpus_loader=corpus_loader, dense_loader=dense_loader)
    engine.loads = loads
    return engine


def test_all_four_models_return_hits_joined_with_corpus(engine):
    for model in MODELS:
        response = engine.search(model, "gato perro", 2)
        assert response.model == model
        assert 1 <= len(response.hits) <= 2
        assert [hit.rank for hit in response.hits] == list(range(1, len(response.hits) + 1))
        for hit in response.hits:
            assert hit.title == CORPUS[hit.doc_id].title
            assert hit.text == CORPUS[hit.doc_id].text
        assert response.elapsed_ms >= 0


def test_jaccard_matches_hand_calculation(engine):
    # Q = {gato, perro}; el documento es título + texto.
    # d1 = {gato, perro}:        2 / 2
    # d2 = {gatos, gato, raton}: 1 / (2 + 3 - 1) = 1/4
    # d3 = {perro, pez, loro, casa}: 1 / (2 + 4 - 1) = 1/5
    response = engine.search("jaccard", "gato perro", 3)
    assert [(hit.doc_id, hit.score) for hit in response.hits] == [
        ("d1", 1.0), ("d2", pytest.approx(1 / 4)), ("d3", pytest.approx(1 / 5)),
    ]


def test_models_are_loaded_once_and_reused(engine):
    for _ in range(3):
        engine.search("dense", "gato", 1)
        engine.search("bm25", "gato", 1)
        engine.search("tfidf", "gato", 1)
    assert engine.loads == {"corpus": 1, "dense": 1}
    steps = [event.step for event in engine.load_events]
    assert steps.count("index_no_stem") == 1
    assert steps.count("retriever_dense") == 1
    assert steps.count("retriever_bm25") == 1


def test_stem_flag_selects_the_matching_index_and_is_ignored_by_dense(engine):
    assert engine.search("bm25", "gato", 1, stem=True).stem is True
    assert engine.search("bm25", "gato", 1).stem is False
    assert engine.search("dense", "gato", 1, stem=True).stem is None
    steps = [event.step for event in engine.load_events]
    assert "index_stem" in steps and "retriever_bm25_stem" in steps


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_empty_query_has_a_readable_message_and_does_not_load_models(engine, query):
    with pytest.raises(EmptyQueryError, match="Escribe una consulta"):
        engine.search("bm25", query)
    assert engine.load_events == []


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k_is_rejected(engine, k):
    with pytest.raises(ValueError, match="entero positivo"):
        engine.search("bm25", "gato", k)


def test_unknown_model_is_rejected(engine):
    with pytest.raises(ValueError, match="Modelo desconocido"):
        engine.search("lsi", "gato")


def test_query_without_matches_returns_no_hits(engine):
    assert engine.search("bm25", "zzz", 5).hits == ()


def test_missing_artifacts_report_what_and_how_to_fix_it(tmp_path):
    write_artifacts(tmp_path, dense=False)
    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS)
    with pytest.raises(ArtifactsMissingError, match=r"build_artifacts\.py --dense"):
        engine.search("dense", "gato")
    with pytest.raises(ArtifactsMissingError, match=r"build_artifacts\.py --stem"):
        engine.search("bm25", "gato", stem=True)
    assert engine.search("bm25", "gato").hits  # el resto sigue funcionando


def test_availability_lists_each_model_without_loading_anything(tmp_path):
    write_artifacts(tmp_path, dense=False)
    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS)
    report = engine.availability()
    assert report["jaccard"] is None and report["bm25"] is None
    assert "Faltan los archivos" in report["dense"]
    assert engine.load_events == []


def test_missing_message_is_none_when_files_exist_and_text_when_not(tmp_path):
    write_artifacts(tmp_path, dense=False)
    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS)
    assert engine.missing_message("bm25") is None
    assert "index_stem.json" in engine.missing_message("bm25", stem=True)
    assert "--dense" in engine.missing_message("dense")


def test_corrupt_dense_artifact_becomes_model_unavailable(tmp_path):
    write_artifacts(tmp_path)

    def broken(path):
        raise ValueError("SHA-256 incorrecto: index.faiss")

    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS, dense_loader=broken)
    with pytest.raises(ModelUnavailableError, match="SHA-256 incorrecto"):
        engine.search("dense", "gato")
    assert "SHA-256" in engine.availability()["dense"]
    assert engine.search("bm25", "gato").hits


def test_dense_ids_missing_from_corpus_are_rejected(tmp_path):
    write_artifacts(tmp_path)
    engine = SearchEngine(
        tmp_path, corpus_loader=lambda: CORPUS,
        dense_loader=lambda path: FakeDense(("d1", "zzz")),
    )
    with pytest.raises(ModelUnavailableError, match="no están en el corpus"):
        engine.search("dense", "gato")


def test_index_from_another_corpus_is_rejected(tmp_path):
    write_artifacts(tmp_path)
    smaller = {key: CORPUS[key] for key in ("d1", "d2")}
    engine = SearchEngine(tmp_path, corpus_loader=lambda: smaller)
    with pytest.raises(ModelUnavailableError, match="4 documentos"):
        engine.search("bm25", "gato")


def test_corpus_failure_is_readable(tmp_path):
    write_artifacts(tmp_path)

    def broken():
        raise OSError("sin red")

    engine = SearchEngine(tmp_path, corpus_loader=broken)
    with pytest.raises(ModelUnavailableError, match="No se pudo cargar el corpus"):
        engine.search("bm25", "gato")


def test_warm_up_loads_the_dense_encoder_before_the_first_query(engine):
    engine.warm_up()
    steps = [event.step for event in engine.load_events]
    assert "dense_encoder" in steps
    assert engine.loads["dense"] == 1


def test_concurrent_first_searches_load_each_model_once(engine):
    errors = []

    def work():
        try:
            engine.search("dense", "gato", 1)
            engine.search("tfidf", "gato", 1)
        except Exception as error:  # noqa: BLE001 - se recoge para afirmarlo después
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert engine.loads == {"corpus": 1, "dense": 1}


def test_index_with_other_ids_of_the_same_size_is_rejected(tmp_path):
    write_artifacts(tmp_path)
    renamed = {f"x{i}": doc for i, doc in enumerate(CORPUS.values())}
    engine = SearchEngine(tmp_path, corpus_loader=lambda: renamed)
    with pytest.raises(ModelUnavailableError, match="identificadores"):
        engine.search("bm25", "gato")


class BrokenEncoder(FakeDense):
    def search(self, query, k=10):
        raise OSError("sin red para descargar MiniLM")


def test_encoder_failure_does_not_break_startup_or_other_models(tmp_path):
    write_artifacts(tmp_path)
    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS,
                          dense_loader=lambda path: BrokenEncoder())
    with pytest.raises(ModelUnavailableError, match="sin red"):
        engine.warm_up(("dense",))
    with pytest.raises(ModelUnavailableError, match="sin red"):
        engine.search("dense", "gato")
    assert "sin red" in engine.availability()["dense"]
    assert engine.search("bm25", "gato").hits


def test_default_dense_loader_checks_the_corpus_hash(tmp_path, monkeypatch):
    from ris.engine import corpus_sha256

    write_artifacts(tmp_path)
    seen = {}

    def fake_load(path, *, encoder=None, expected_corpus_sha256=None):
        seen["sha"] = expected_corpus_sha256
        return FakeDense()

    monkeypatch.setattr("ris.engine.DenseRetriever.load", fake_load)
    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS)
    engine.search("dense", "gato")
    assert seen["sha"] == corpus_sha256(CORPUS)
