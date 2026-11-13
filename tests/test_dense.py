import json
import sys
from hashlib import sha256
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from ris.data import Doc
from ris.retrievers import DenseRetriever, Retriever
from ris.retrievers.dense import DIMENSION, MODEL_NAME, MODEL_REVISION


def vector(*values):
    result = np.zeros(DIMENSION, dtype=np.float32)
    result[:len(values)] = values
    return result


class Encoder:
    """Doble sin red; mantiene textos y parámetros para comprobar el contrato."""

    def __init__(self):
        self.calls = []
        self.vectors = {
            "Title Original, $3.50!": vector(2, 3),
            " second": vector(3, 0),
            " ": vector(-1, -2),
            " Query ORIGINAL! ": vector(1, 2),
        }

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.array([self.vectors[text] for text in texts])


@pytest.fixture
def corpus():
    return {
        "d3": Doc(title="", text=""),
        "d1": Doc(title="Title", text="Original, $3.50!"),
        "d2": Doc(title="", text="second"),
    }


@pytest.fixture
def built(corpus):
    encoder = Encoder()
    return DenseRetriever.build(corpus, encoder=encoder, batch_size=2), encoder


def test_exact_cosine_original_texts_and_batches(built):
    retriever, encoder = built
    model: Retriever = retriever
    results = model.search(" Query ORIGINAL! ", k=20)
    # Ejemplo vectorial docente: q=(1,2), d1=(2,3), d2=(3,0), d3=(-1,-2).
    assert [r.doc_id for r in results] == ["d1", "d2", "d3"]
    assert [r.score for r in results] == pytest.approx(
        [8 / np.sqrt(65), 1 / np.sqrt(5), -1], abs=1e-6
    )
    assert [call[0] for call in encoder.calls] == [
        ["Title Original, $3.50!", " second"], [" "], [" Query ORIGINAL! "]
    ]
    assert encoder.max_seq_length == 256
    assert encoder.calls[0][1]["convert_to_numpy"] is True
    assert len({r.doc_id for r in results}) == 3


def test_top_k_ties_at_boundary_are_globally_sorted(corpus):
    encoder = Encoder()
    encoder.vectors = dict.fromkeys(encoder.vectors, vector(1))
    retriever = DenseRetriever.build(corpus, encoder=encoder)
    assert [r.doc_id for r in retriever.search(" Query ORIGINAL! ", k=2)] == [
        "d1", "d2"
    ]


@pytest.mark.parametrize("query", ["", "  \n\t"])
def test_empty_query_does_not_encode(built, query):
    retriever, encoder = built
    count = len(encoder.calls)
    assert retriever.search(query) == []
    assert len(encoder.calls) == count


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_precedes_empty_query(built, k):
    with pytest.raises(ValueError, match="positivo"):
        built[0].search("", k=k)


def test_save_load_preserves_vectors_ids_and_query_without_rebuild(built, tmp_path):
    retriever, _ = built
    retriever.save(tmp_path)
    encoder = Encoder()
    loaded = DenseRetriever.load(tmp_path, encoder=encoder)
    assert encoder.calls == []
    assert loaded.search(" Query ORIGINAL! ") == retriever.search(" Query ORIGINAL! ")
    assert [call[0] for call in encoder.calls] == [[" Query ORIGINAL! "]]
    embeddings = np.load(tmp_path / "embeddings.npy", allow_pickle=False)
    index = faiss.read_index(str(tmp_path / "index.faiss"))
    assert isinstance(index, faiss.IndexFlatIP)
    assert embeddings.dtype == np.float32
    assert embeddings.shape == (3, DIMENSION)
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1, 1, 1])
    assert np.array_equal(index.reconstruct_n(0, 3), embeddings)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["doc_ids"] == ["d1", "d2", "d3"]
    assert metadata["model_name"] == MODEL_NAME
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["max_seq_length"] == 256
    assert metadata["batch_size"] == 2
    assert metadata["normalization"] == "l2"
    assert metadata["dimension"] == DIMENSION
    for name, digest in metadata["file_sha256"].items():
        assert sha256((tmp_path / name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", ["index.faiss", "embeddings.npy"])
def test_corrupted_file_is_rejected_before_loading(built, tmp_path, name):
    built[0].save(tmp_path)
    with (tmp_path / name).open("ab") as file:
        file.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256"):
        DenseRetriever.load(tmp_path)


@pytest.mark.parametrize("field,value", [
    ("format_version", 99), ("model_revision", "wrong"),
    ("dimension", 768), ("normalization", "none"),
    ("max_seq_length", 512), ("doc_ids", ["d1", "d1", "d3"]),
    ("doc_ids", ["d2", "d1", "d3"]), ("batch_size", 0),
])
def test_incompatible_metadata_is_rejected(built, tmp_path, field, value):
    built[0].save(tmp_path)
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        DenseRetriever.load(tmp_path)


@pytest.mark.parametrize("bad", [vector(0), vector(float("nan")), np.ones(12)])
def test_invalid_encoder_vectors_are_rejected(corpus, bad):
    encoder = Encoder()
    encoder.vectors = dict.fromkeys(encoder.vectors, bad)
    with pytest.raises(ValueError):
        DenseRetriever.build(corpus, encoder=encoder)


def test_empty_corpus_saves_loads_without_encoder(tmp_path):
    retriever = DenseRetriever.build({})
    retriever.save(tmp_path)
    loaded = DenseRetriever.load(tmp_path)
    assert loaded.search("query") == []


def test_expected_corpus_hash_is_checked(built, tmp_path):
    built[0].save(tmp_path)
    with pytest.raises(ValueError, match="otro corpus"):
        DenseRetriever.load(tmp_path, expected_corpus_sha256="0" * 64)


def test_query_vectors_are_validated(built):
    retriever, encoder = built
    encoder.vectors[" Query ORIGINAL! "] = vector(float("inf"))
    with pytest.raises(ValueError, match="no finitos"):
        retriever.search(" Query ORIGINAL! ")


@pytest.mark.parametrize("field,value", [("batch_size", 0), ("threads", 0)])
def test_invalid_build_options(corpus, field, value):
    with pytest.raises(ValueError, match="positivo"):
        DenseRetriever.build(corpus, **{field: value})


def test_progress_and_stable_ids_do_not_depend_on_input_order(corpus):
    reports = []
    forward = DenseRetriever.build(
        corpus, encoder=Encoder(), batch_size=2,
        progress_callback=lambda done, total: reports.append((done, total)),
    )
    backward = DenseRetriever.build(dict(reversed(list(corpus.items()))), encoder=Encoder())
    assert reports == [(2, 3), (3, 3)]
    assert forward.doc_ids == backward.doc_ids
    assert forward.metadata["corpus_sha256"] == backward.metadata["corpus_sha256"]
    assert forward.search(" Query ORIGINAL! ") == backward.search(" Query ORIGINAL! ")


def test_correct_hash_cannot_hide_wrong_vector_index_correspondence(built, tmp_path):
    built[0].save(tmp_path)
    path = tmp_path / "embeddings.npy"
    vectors = np.load(path)
    vectors[[0, 1]] = vectors[[1, 0]]
    np.save(path, vectors)
    manifest = tmp_path / "metadata.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["file_sha256"]["embeddings.npy"] = sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="correspondencia"):
        DenseRetriever.load(tmp_path)


def test_correct_hash_cannot_hide_wrong_index_type(built, tmp_path):
    built[0].save(tmp_path)
    path = tmp_path / "index.faiss"
    index = faiss.IndexFlatL2(DIMENSION)
    index.add(np.load(tmp_path / "embeddings.npy"))
    faiss.write_index(index, str(path))
    manifest = tmp_path / "metadata.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["file_sha256"]["index.faiss"] = sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="FAISS incompatible"):
        DenseRetriever.load(tmp_path)


def test_real_encoder_factory_is_lazy_pinned_and_reused(built, tmp_path, monkeypatch):
    built[0].save(tmp_path)
    created = []
    thread_counts = []
    encoder = Encoder()

    def factory(name, **kwargs):
        created.append((name, kwargs))
        return encoder

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        SentenceTransformer=factory
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=thread_counts.append
    ))
    loaded = DenseRetriever.load(tmp_path)
    assert created == []
    loaded.search(" Query ORIGINAL! ")
    loaded.search(" Query ORIGINAL! ")
    assert created == [(MODEL_NAME, {
        "revision": MODEL_REVISION, "device": "cpu", "trust_remote_code": False
    })]
    assert thread_counts == [8]
    assert [call[0] for call in encoder.calls] == [[" Query ORIGINAL! "]] * 2
