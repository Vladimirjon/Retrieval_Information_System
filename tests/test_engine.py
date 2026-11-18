import threading

import pytest

from ris.data import Doc
from ris.engine import (
    MODELS,
    ArtifactsMissingError,
    EmptyQueryError,
    ModelUnavailableError,
    SearchEngine,
    _edit_distance,
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

    for module in ("index", "engine", "retrievers.jaccard", "retrievers.tfidf", "retrievers.bm25"):
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


def test_query_terms_are_unique_and_keep_order(engine):
    assert engine.query_terms("Perro gato perro") == ("perro", "gato")
    assert engine.query_terms("   ") == ()


def test_matched_terms_lists_query_terms_present_in_the_document(engine):
    terms = engine.query_terms("gato loro ballena")
    assert engine.matched_terms("d1", terms) == ("gato",)
    assert engine.matched_terms("d3", terms) == ("loro",)
    assert engine.matched_terms("d4", terms) == ()


def test_skip_empty_is_opt_in_and_fills_requested_dense_ranking(engine):
    normal = engine.search("dense", "gato", 2)
    assert [hit.doc_id for hit in normal.hits] == ["d4", "d3"]
    filtered = engine.search("dense", "gato", 2, skip_empty=True)
    assert [hit.doc_id for hit in filtered.hits] == ["d3", "d2"]
    assert [hit.rank for hit in filtered.hits] == [1, 2]
    assert filtered.k == 2


def test_skip_empty_returns_fewer_hits_when_no_more_nonempty_docs(engine):
    result = engine.search("dense", "gato", 4, skip_empty=True)
    assert [hit.doc_id for hit in result.hits] == ["d3", "d2", "d1"]


def test_skip_empty_uses_text_only_and_requests_extra_only_for_dense(engine, monkeypatch):
    requested = []
    original_run = engine._run

    def track_run(model, stem, retriever, query, k):
        requested.append((model, k))
        return original_run(model, stem, retriever, query, k)

    monkeypatch.setattr(engine, "_run", track_run)
    assert engine.empty_doc_ids == frozenset({"d4"})
    assert [hit.doc_id for hit in engine.search("bm25", "gato", 2, skip_empty=True).hits] == ["d1", "d2"]
    assert [hit.doc_id for hit in engine.search("dense", "gato", 2, skip_empty=True).hits] == ["d3", "d2"]
    assert requested == [("bm25", 2), ("dense", 3)]


def test_title_without_visible_text_is_empty(tmp_path):
    corpus = {"d1": Doc(title="Only title", text="  "), "d2": Doc(title="", text="visible")}
    engine = SearchEngine(tmp_path, corpus_loader=lambda: corpus)
    assert engine.empty_doc_ids == frozenset({"d1"})


GRAPH_VECTORS = {"d1": [1.0, 0.0], "d2": [0.0, 1.0], "d3": [-1.0, 0.0], "d4": [0.6, 0.8]}


class GraphDense(FakeDense):
    """Doble con vectores: d1 coincide con la consulta, d2 es ortogonal y d3 opuesto."""

    def encode_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0]

    def vectors(self, doc_ids):
        unknown = [doc_id for doc_id in doc_ids if doc_id not in GRAPH_VECTORS]
        if unknown:
            raise ValueError(f"Documento sin vector: {unknown[0]}")
        return [GRAPH_VECTORS[doc_id] for doc_id in doc_ids]


def graph_engine(tmp_path, dense=GraphDense):
    write_artifacts(tmp_path)
    return SearchEngine(tmp_path, corpus_loader=lambda: CORPUS, dense_loader=lambda path: dense())


def test_query_graph_places_the_given_documents_by_meaning(tmp_path):
    engine = graph_engine(tmp_path)
    graph = engine.query_graph("gato", ["d2", "d1", "d3"], neighbors=1)
    assert [node.doc_id for node in graph.nodes] == ["d2", "d1", "d3"]
    assert [node.similarity for node in graph.nodes] == pytest.approx([0.0, 1.0, -1.0])
    assert graph.nodes[1].position == pytest.approx((0.0, 0.0, 0.0))  # igual que la consulta
    assert max(sum(v * v for v in node.position) for node in graph.nodes) == pytest.approx(1.0)
    assert [edge[:2] for edge in graph.edges] == [(0, 1), (0, 2)]  # d2 es el más cercano de d1 y d3


def test_query_graph_uses_the_dense_model_whatever_ranked_the_documents(tmp_path):
    engine = graph_engine(tmp_path)
    ranked = [hit.doc_id for hit in engine.search("bm25", "gato perro", k=3).hits]
    graph = engine.query_graph("gato perro", ranked)
    assert [node.doc_id for node in graph.nodes] == ranked
    assert engine.get_retriever("dense").queries == ["gato perro"]


@pytest.mark.parametrize("query", ["", "  "])
def test_query_graph_needs_a_query(tmp_path, query):
    with pytest.raises(EmptyQueryError):
        graph_engine(tmp_path).query_graph(query, ["d1"])


def test_query_graph_without_dense_artifacts_says_what_is_missing(tmp_path):
    write_artifacts(tmp_path, dense=False)
    engine = SearchEngine(tmp_path, corpus_loader=lambda: CORPUS)
    with pytest.raises(ArtifactsMissingError, match="Embeddings"):
        engine.query_graph("gato", ["d1"])
    assert engine.search("bm25", "gato", k=1).hits  # la búsqueda no depende del grafo


def test_query_graph_failure_is_readable_and_does_not_disable_dense(tmp_path):
    engine = graph_engine(tmp_path, dense=FakeDense)  # sin encode_query ni vectors
    with pytest.raises(ModelUnavailableError, match="No se pudo calcular el grafo"):
        engine.query_graph("gato", ["d1"])
    with pytest.raises(ModelUnavailableError, match="zz"):
        graph_engine(tmp_path / "otro").query_graph("gato", ["zz"])
    assert engine.availability()["dense"] is None
    assert engine.search("dense", "gato", k=1).hits


def test_query_graph_passes_linked_through(tmp_path):
    engine = graph_engine(tmp_path)
    graph = engine.query_graph("gato", ["d1", "d2", "d3"], neighbors=1, linked=2)
    assert [edge[:2] for edge in graph.edges] == [(0, 1)]


@pytest.mark.parametrize(("a", "b", "limit", "expected"), [
    ("stok", "stock", 1, 1), ("morgage", "mortgage", 1, 1), ("abc", "abc", 0, 0),
    ("kitten", "sitting", 3, 3), ("kitten", "sitting", 2, None), ("ab", "abcd", 1, None),
])
def test_edit_distance_is_levenshtein_up_to_the_limit(a, b, limit, expected):
    assert _edit_distance(a, b, limit) == expected


SPELLING = {
    # stocks en 40 documentos, stock en 30, "stoks" mal escrito en uno solo, etc.
    **{f"s{i}": Doc(title="", text="stocks stock") for i in range(30)},
    **{f"t{i}": Doc(title="", text="stocks") for i in range(10)},
    "u1": Doc(title="", text="stoks morgage"),
    **{f"m{i}": Doc(title="", text="mortgage") for i in range(25)},
    **{f"r{i}": Doc(title="", text="retirement planning") for i in range(3)},
    "x1": Doc(title="", text="dalbar okpay"),
}


def spelling_engine(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    InvertedIndex.build(SPELLING, stem=False).save(tmp_path / "index_no_stem.json")
    return SearchEngine(tmp_path, corpus_loader=lambda: SPELLING)


@pytest.mark.parametrize(("query", "expected"), [
    ("stoks", "stocks"),                        # rara (1 doc) y falta una letra: stocks (40 docs)
    ("what is a morgage", "what is a mortgage"),  # en el corpus, pero 25 veces menos frecuente
    ("retirment planning", "retirement planning"),  # desconocida, distancia 1
    ("retir", "retirement"),                    # palabra empezada: se completa
    ("stocks and mortgage", None),             # todo correcto
    ("Dalbar stoks", "Dalbar stocks"),         # con mayúsculas no se toca (siglas, nombres)
    ("STOKS", None),
    ("zzzzqqq", None),                          # nada parecido
    ("", None),
])
def test_correct_query_fixes_misspelled_lowercase_words(tmp_path, query, expected):
    assert spelling_engine(tmp_path).correct_query(query) == expected


def test_correct_query_needs_the_rare_word_to_be_much_less_frequent(tmp_path):
    engine = spelling_engine(tmp_path)
    # "stock" (30 docs) está en el corpus y no es rara: no se cambia por "stocks".
    assert engine.correct_query("stock") is None


def test_correct_query_without_index_returns_none(tmp_path):
    engine = SearchEngine(tmp_path, corpus_loader=lambda: SPELLING)
    assert engine.correct_query("stoks") is None


def test_top_terms_are_the_most_frequent_alphabetic_terms(tmp_path):
    engine = spelling_engine(tmp_path)
    assert engine.top_terms(3) == ("stocks", "stock", "mortgage")
    assert len(engine.top_terms(100)) == len(set(engine.top_terms(100)))
