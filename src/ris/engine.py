"""Motor de búsqueda único: carga perezosa, validada y reutilizable de los cuatro modelos.

No depende de ninguna interfaz. La app Streamlit (#13) y las vistas de comparación
(#14) y métricas (#15) usan este mismo objeto; ningún otro código debe construir
recuperadores por su cuenta.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ris.data import Doc, load_corpus
from ris.index import InvertedIndex
from ris.preprocessing import preprocess
from ris.retrievers import (
    BM25Retriever,
    DenseRetriever,
    JaccardRetriever,
    Retriever,
    TfidfRetriever,
)
from ris.retrievers.dense import _json_hash

MODELS = ("jaccard", "tfidf", "bm25", "dense")
MODEL_LABELS = {
    "jaccard": "Jaccard (binario)",
    "tfidf": "TF-IDF (coseno)",
    "bm25": "BM25 (probabilístico)",
    "dense": "Embeddings (MiniLM + FAISS)",
}
CLASSICAL_MODELS = ("jaccard", "tfidf", "bm25")
_CLASSICAL_CLASSES = {
    "jaccard": JaccardRetriever,
    "tfidf": TfidfRetriever,
    "bm25": BM25Retriever,
}
_INDEX_FILES = {False: "index_no_stem.json", True: "index_stem.json"}
_DENSE_FILES = ("metadata.json", "embeddings.npy", "index.faiss")
_WARMUP_QUERY = "investment"


class EngineError(Exception):
    """Fallo esperable con un mensaje legible para la persona usuaria."""


class EmptyQueryError(EngineError):
    """La consulta no contiene texto."""


class ArtifactsMissingError(EngineError):
    """Faltan archivos de índices o embeddings."""


class ModelUnavailableError(EngineError):
    """Un modelo existe en disco pero no se pudo cargar o no es coherente."""


@dataclass(frozen=True)
class Hit:
    """Resultado listo para mostrar: une el ranking con el texto original."""

    rank: int
    doc_id: str
    title: str
    text: str
    score: float


@dataclass(frozen=True)
class SearchResponse:
    """Respuesta de una búsqueda. `elapsed_ms` mide solo la recuperación."""

    model: str
    stem: bool | None
    query: str
    k: int
    hits: tuple[Hit, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class LoadEvent:
    """Registro de un paso de carga, para medir arranque y memoria."""

    step: str
    seconds: float
    rss_mib: float | None


def current_rss_mib() -> float | None:
    """Memoria residente del proceso en MiB, o None si psutil no está disponible."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**20


def corpus_sha256(corpus: Mapping[str, Doc]) -> str:
    """Hash del corpus con la fórmula que DenseRetriever.build guarda en su manifiesto
    (la misma que usa scripts/evaluate.py)."""
    return _json_hash([
        [doc_id, corpus[doc_id].title, corpus[doc_id].text] for doc_id in sorted(corpus)
    ])


def _stem_key(model: str, stem: bool) -> tuple[str, bool | None]:
    return (model, None if model == "dense" else stem)


class SearchEngine:
    """Carga cada modelo una sola vez y atiende búsquedas de lectura concurrentes.

    Los recuperadores son de solo lectura tras cargarse, de modo que varias sesiones
    pueden buscar a la vez. La carga está protegida con un candado por modelo.
    """

    def __init__(
        self,
        artifacts_dir: str | Path = "artifacts",
        *,
        corpus_loader: Callable[[], Mapping[str, Doc]] = load_corpus,
        dense_loader: Callable[[Path], Retriever] | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._corpus_loader = corpus_loader
        self._dense_loader = dense_loader
        self._corpus: Mapping[str, Doc] | None = None
        self._empty_doc_ids: frozenset[str] | None = None
        self._indexes: dict[bool, InvertedIndex] = {}
        self._retrievers: dict[tuple[str, bool | None], Retriever] = {}
        self._failures: dict[tuple[str, bool | None], str] = {}
        self._locks: dict[object, threading.Lock] = {}
        self._registry_lock = threading.Lock()
        self.load_events: list[LoadEvent] = []

    # ------------------------------------------------------------------ estado

    def _lock(self, key: object) -> threading.Lock:
        with self._registry_lock:
            return self._locks.setdefault(key, threading.Lock())

    def _record(self, step: str, started: float) -> None:
        self.load_events.append(
            LoadEvent(step, round(perf_counter() - started, 3), current_rss_mib())
        )

    def missing_files(self, model: str, stem: bool = False) -> list[Path]:
        """Archivos de artefactos que faltan para un modelo (sin cargar nada)."""
        self._check_model(model)
        if model == "dense":
            base = self.artifacts_dir / "dense"
            return [base / name for name in _DENSE_FILES if not (base / name).is_file()]
        path = self.artifacts_dir / _INDEX_FILES[stem]
        return [] if path.is_file() else [path]

    def missing_message(self, model: str, stem: bool = False) -> str | None:
        """Mensaje legible si faltan archivos del modelo; None si están todos."""
        missing = self.missing_files(model, stem)
        return self._missing_message(model, stem, missing) if missing else None

    def availability(self, stem: bool = False) -> dict[str, str | None]:
        """Para cada modelo: None si está listo, o el motivo legible si no lo está."""
        return {
            model: self.missing_message(model, stem)
            or self._failures.get(_stem_key(model, stem))
            for model in MODELS
        }

    @staticmethod
    def _check_model(model: str) -> None:
        if model not in MODELS:
            raise ValueError(f"Modelo desconocido: {model!r}. Opciones: {MODELS}")

    @staticmethod
    def _missing_message(model: str, stem: bool, missing: list[Path]) -> str:
        names = ", ".join(path.name for path in missing)
        if model == "dense":
            command = "python scripts/build_artifacts.py --dense"
        else:
            command = "python scripts/build_artifacts.py" + (" --stem" if stem else "")
        return (
            f"Faltan los archivos de {MODEL_LABELS[model]}: {names}. "
            f"Se generan con {command} o se descargan del Hub del equipo."
        )

    # ------------------------------------------------------------------- carga

    @property
    def corpus(self) -> Mapping[str, Doc]:
        """Corpus completo (doc_id → Doc), cargado una sola vez."""
        if self._corpus is None:
            with self._lock("corpus"):
                if self._corpus is None:
                    started = perf_counter()
                    try:
                        corpus = self._corpus_loader()
                    except Exception as error:
                        raise ModelUnavailableError(
                            "No se pudo cargar el corpus FiQA. Comprueba la conexión "
                            f"o la caché de Hugging Face. Detalle: {error}"
                        ) from error
                    if not corpus:
                        raise ModelUnavailableError("El corpus FiQA está vacío.")
                    self._empty_doc_ids = frozenset(
                        doc_id for doc_id, doc in corpus.items() if not doc.text.strip()
                    )
                    self._corpus = corpus
                    self._record("corpus", started)
        return self._corpus

    @property
    def empty_doc_ids(self) -> frozenset[str]:
        """Ids sin texto visible, calculados una vez al cargar el corpus."""
        if self._empty_doc_ids is None:
            _ = self.corpus
        return self._empty_doc_ids

    def _index(self, stem: bool) -> InvertedIndex:
        if stem not in self._indexes:
            with self._lock(("index", stem)):
                if stem not in self._indexes:
                    started = perf_counter()
                    path = self.artifacts_dir / _INDEX_FILES[stem]
                    index = InvertedIndex.load(path, expected_stem=stem)
                    if index.N != len(self.corpus):
                        raise ValueError(
                            f"El índice tiene {index.N} documentos y el corpus "
                            f"{len(self.corpus)}"
                        )
                    if index.doc_lengths.keys() != self.corpus.keys():
                        raise ValueError(
                            "Los identificadores del índice no coinciden con los del corpus"
                        )
                    self._indexes[stem] = index
                    self._record(f"index_{'stem' if stem else 'no_stem'}", started)
        return self._indexes[stem]

    def _build(self, model: str, stem: bool) -> Retriever:
        if model == "dense":
            path = self.artifacts_dir / "dense"
            if self._dense_loader is None:
                # El manifiesto guarda el hash del corpus con el que se codificó.
                retriever = DenseRetriever.load(
                    path, expected_corpus_sha256=corpus_sha256(self.corpus)
                )
            else:
                retriever = self._dense_loader(path)
            unknown = set(retriever.doc_ids) - set(self.corpus)
            if unknown:
                raise ValueError(
                    f"{len(unknown)} documentos del índice denso no están en el corpus"
                )
            return retriever
        return _CLASSICAL_CLASSES[model](self._index(stem), stem=stem)

    def get_retriever(self, model: str, stem: bool = False) -> Retriever:
        """Devuelve el recuperador cargado, o lo carga la primera vez."""
        self._check_model(model)
        key = _stem_key(model, stem)
        if key in self._retrievers:
            return self._retrievers[key]
        missing = self.missing_files(model, stem)
        if missing:
            raise ArtifactsMissingError(self._missing_message(model, stem, missing))
        with self._lock(key):
            if key in self._retrievers:
                return self._retrievers[key]
            started = perf_counter()
            try:
                retriever = self._build(model, stem)
            except EngineError as error:
                self._failures[key] = str(error)
                raise
            except Exception as error:
                message = (
                    f"No se pudo cargar {MODEL_LABELS[model]}: {error}. "
                    "Reconstruye los artefactos o vuelve a descargarlos."
                )
                self._failures[key] = message
                raise ModelUnavailableError(message) from error
            self._failures.pop(key, None)
            self._retrievers[key] = retriever
            self._record(f"retriever_{model}" + ("_stem" if stem else ""), started)
            return retriever

    def warm_up(self, models: tuple[str, ...] = MODELS, stem: bool = False) -> None:
        """Carga los modelos pedidos y fuerza la carga perezosa de MiniLM.

        Sin esto, la primera consulta densa de cada proceso paga la carga del modelo.
        """
        self.corpus  # noqa: B018 - la propiedad carga el corpus
        for model in models:
            retriever = self.get_retriever(model, stem)
            if model == "dense":
                started = perf_counter()
                self._run(model, stem, retriever, _WARMUP_QUERY, 1)
                self._record("dense_encoder", started)

    # ------------------------------------------------------------- explicación

    @staticmethod
    def query_terms(query: str, *, stem: bool = False) -> tuple[str, ...]:
        """Términos con los que buscan los modelos clásicos, sin repetir y en orden."""
        return tuple(dict.fromkeys(preprocess(query, stem=stem)))

    def matched_terms(
        self, doc_id: str, terms: tuple[str, ...], *, stem: bool = False
    ) -> tuple[str, ...]:
        """Cuáles de `terms` aparecen en el documento según el índice clásico."""
        postings = self._index(stem).postings
        return tuple(term for term in terms if doc_id in postings.get(term, ()))

    # ---------------------------------------------------------------- búsqueda

    def _run(self, model: str, stem: bool, retriever: Retriever, query: str, k: int):
        """Busca y convierte fallos inesperados (p. ej. MiniLM no descargable) en
        ModelUnavailableError: un modelo roto no debe tumbar la app ni a los demás."""
        try:
            return retriever.search(query, k)
        except EngineError:
            raise
        except Exception as error:
            message = f"No se pudo usar {MODEL_LABELS[model]}: {error}"
            self._failures[_stem_key(model, stem)] = message
            raise ModelUnavailableError(message) from error

    def search(
        self, model: str, query: str, k: int = 10, *, stem: bool = False,
        skip_empty: bool = False,
    ) -> SearchResponse:
        """Busca en un modelo y une cada doc_id con su título y texto originales."""
        self._check_model(model)
        if not query or not query.strip():
            raise EmptyQueryError("Escribe una consulta para buscar.")
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k debe ser un entero positivo")
        retriever = self.get_retriever(model, stem)
        started = perf_counter()
        corpus = self.corpus
        if skip_empty and model == "dense":
            requested = min(len(corpus), k + len(self.empty_doc_ids))
        else:
            requested = k
        results = self._run(model, stem, retriever, query, requested)
        if skip_empty:
            results = [result for result in results
                       if result.doc_id not in self.empty_doc_ids][:k]
        elapsed_ms = (perf_counter() - started) * 1000
        hits = tuple(
            Hit(rank, result.doc_id, corpus[result.doc_id].title,
                corpus[result.doc_id].text, result.score)
            for rank, result in enumerate(results, start=1)
        )
        return SearchResponse(
            model=model, stem=_stem_key(model, stem)[1], query=query, k=k,
            hits=hits, elapsed_ms=elapsed_ms,
        )
