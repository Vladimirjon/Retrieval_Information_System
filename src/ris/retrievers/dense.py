"""MiniLM sobre texto original y búsqueda exacta con producto interno FAISS."""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import faiss
import numpy as np

from ris.data import Doc
from ris.retrievers.base import Result

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIMENSION = 384
MAX_SEQ_LENGTH = 256
_CONFIG = {
    "format_version": 1,
    "model_name": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "dimension": DIMENSION,
    "dtype": "float32",
    "normalization": "l2",
    "index_type": "IndexFlatIP",
    "max_seq_length": MAX_SEQ_LENGTH,
    "document_text": "title + ' ' + text",
}


def _json_hash(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                  .encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} debe ser un entero positivo")


def _normalized(vectors: object, count: int) -> np.ndarray:
    values = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if values.shape != (count, DIMENSION) or not np.isfinite(values).all():
        raise ValueError("Embeddings con dimensiones incorrectas o valores no finitos")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Los embeddings deben tener norma positiva y finita")
    faiss.normalize_L2(values)
    return values


class DenseRetriever:
    """Artefacto cargable sin codificar de nuevo el corpus ni cargar el modelo.

    El encoder se carga al construir o en la primera consulta no vacía; se puede
    inyectar un doble con encode() para pruebas sin red. No modificar el índice
    ni los metadatos durante el uso del recuperador.
    """

    def __init__(self, index, metadata: dict, *, encoder=None) -> None:
        self._index = index
        self.metadata = metadata
        self.doc_ids = tuple(metadata["doc_ids"])
        self._encoder = encoder

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        if self._encoder is None:
            # Importación diferida: los modelos clásicos no cargan torch/MiniLM.
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.metadata["threads"])
            self._encoder = SentenceTransformer(
                MODEL_NAME, revision=MODEL_REVISION, device="cpu",
                trust_remote_code=False,
            )
        self._encoder.max_seq_length = MAX_SEQ_LENGTH
        vectors = self._encoder.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return _normalized(vectors, len(texts))

    @classmethod
    def build(
        cls, corpus: dict[str, Doc], *, batch_size: int = 32, threads: int = 8,
        encoder=None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> DenseRetriever:
        """Codifica una vez cada documento, incluidos los de texto vacío."""
        _positive_int(batch_size, "batch_size")
        _positive_int(threads, "threads")
        faiss.omp_set_num_threads(threads)
        if any(not isinstance(doc_id, str) for doc_id in corpus):
            raise ValueError("Todos los doc_ids deben ser strings")
        doc_ids = sorted(corpus)
        metadata = {
            **_CONFIG, "doc_ids": doc_ids, "count": len(doc_ids),
            "doc_ids_sha256": _json_hash(doc_ids), "batch_size": batch_size,
            "threads": threads,
            "corpus_sha256": _json_hash([
                [doc_id, corpus[doc_id].title, corpus[doc_id].text]
                for doc_id in doc_ids
            ]),
            "library_versions": {name: version(name) for name in (
                "sentence-transformers", "transformers", "torch", "faiss-cpu", "numpy"
            )},
        }
        retriever = cls(faiss.IndexFlatIP(DIMENSION), metadata, encoder=encoder)
        for start in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[start:start + batch_size]
            texts = [f"{corpus[id_].title} {corpus[id_].text}" for id_ in batch_ids]
            retriever._index.add(retriever._encode(texts, batch_size))
            if progress_callback is not None:
                progress_callback(start + len(batch_ids), len(doc_ids))
        return retriever

    def search(self, query: str, k: int = 10) -> list[Result]:
        """Devuelve cosenos, incluidos negativos, con desempate global por ID."""
        _positive_int(k, "k")
        if not query.strip() or not self.doc_ids:
            return []
        query_vector = self._encode([query], 1)
        # IndexFlatIP ya examina N vectores. Pedir los N scores evita perder
        # empates de frontera: FAISS no garantiza orden lexicográfico por ID.
        scores, rows = self._index.search(query_vector, len(self.doc_ids))
        results = [Result(self.doc_ids[int(row)], float(score))
                   for score, row in zip(scores[0], rows[0], strict=True)]
        results.sort(key=lambda result: (-result.score, result.doc_id))
        return results[:k]

    def save(self, path: str | Path) -> None:
        """Guarda vectores, índice y manifiesto; este último se escribe al final.

        Los hashes detectan corrupción accidental, no autentican al publicador.
        Usar secuencialmente: no escribir y leer el mismo directorio en paralelo.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        embeddings = self._index.reconstruct_n(0, len(self.doc_ids))
        np.save(path / "embeddings.npy", embeddings, allow_pickle=False)
        faiss.write_index(self._index, str(path / "index.faiss"))
        metadata = {**self.metadata, "file_sha256": {
            name: _file_hash(path / name) for name in ("embeddings.npy", "index.faiss")
        }}
        with (path / "metadata.json").open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
            file.write("\n")

    @classmethod
    def load(
        cls, path: str | Path, *, encoder=None,
        expected_corpus_sha256: str | None = None,
    ) -> DenseRetriever:
        """Comprueba integridad, configuración y correspondencia fila-doc_id."""
        path = Path(path)
        try:
            with (path / "metadata.json").open(encoding="utf-8") as file:
                metadata = json.load(file)
            cls._validate_metadata(metadata, expected_corpus_sha256)
            # Verificar antes de deserializar FAISS; distribuir desde un Hub de confianza.
            for name in ("embeddings.npy", "index.faiss"):
                if _file_hash(path / name) != metadata["file_sha256"][name]:
                    raise ValueError(f"SHA-256 incorrecto: {name}")
            vectors = np.load(path / "embeddings.npy", allow_pickle=False, mmap_mode="r")
            count = metadata["count"]
            if vectors.dtype != np.float32 or vectors.shape != (count, DIMENSION):
                raise ValueError("Formato o dimensiones de embeddings incompatibles")
            index = faiss.read_index(str(path / "index.faiss"))
            if (not isinstance(index, faiss.IndexFlatIP) or index.d != DIMENSION
                    or index.ntotal != count or index.metric_type != faiss.METRIC_INNER_PRODUCT):
                raise ValueError("Índice FAISS incompatible con los metadatos")
            for start in range(0, count, 1024):
                batch = vectors[start:start + 1024]
                if (not np.isfinite(batch).all()
                        or not np.allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-5)
                        or not np.array_equal(index.reconstruct_n(start, len(batch)), batch)):
                    raise ValueError("Vectores sin normalizar o correspondencia FAISS incorrecta")
        except (OSError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f"Artefacto denso inválido o incompleto: {error}") from error
        return cls(index, metadata, encoder=encoder)

    @staticmethod
    def _validate_metadata(metadata: dict, expected_corpus_sha256: str | None) -> None:
        if not isinstance(metadata, dict) or any(
            metadata.get(key) != value for key, value in _CONFIG.items()
        ):
            raise ValueError("Configuración o versión del artefacto denso incompatible")
        ids = metadata["doc_ids"]
        if (not isinstance(ids, list) or any(not isinstance(id_, str) for id_ in ids)
                or len(ids) != len(set(ids)) or ids != sorted(ids)
                or type(metadata["count"]) is not int or metadata["count"] != len(ids)
                or metadata["doc_ids_sha256"] != _json_hash(ids)):
            raise ValueError("Mapeo de doc_ids inválido o sin correspondencia")
        _positive_int(metadata["batch_size"], "batch_size")
        _positive_int(metadata["threads"], "threads")
        digest = metadata["corpus_sha256"]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("Hash del corpus inválido")
        if expected_corpus_sha256 is not None and digest != expected_corpus_sha256:
            raise ValueError("El artefacto pertenece a otro corpus")
