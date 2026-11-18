"""Motor de búsqueda único: carga perezosa, validada y reutilizable de los cuatro modelos.

No depende de ninguna interfaz. La app Streamlit (#13) y las vistas de comparación
(#14) y métricas (#15) usan este mismo objeto; ningún otro código debe construir
recuperadores por su cuenta.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ris.data import Doc, load_corpus
from ris.graph import QueryGraph, build_query_graph
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
_WORD = re.compile(r"[A-Za-z]+")
# Una palabra que sí está en FiQA pero en menos de _RARE_DF documentos se corrige si hay
# otra muy parecida al menos _MORE_COMMON veces más frecuente: el corpus tiene erratas
# ("morgage", "retirment") y sin esto nunca se sugeriría la forma correcta.
_RARE_DF = 10
_MORE_COMMON = 20
_CORRECTIONS_CACHE = 20_000


def _edit_distance(a: str, b: str, limit: int) -> int | None:
    """Distancia de Levenshtein si no pasa de `limit`; None si la supera."""
    if abs(len(a) - len(b)) > limit:
        return None
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (char_a != char_b)))
        if min(current) > limit:
            return None
        previous = current
    return previous[-1] if previous[-1] <= limit else None


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
        self._spelling: dict[str, tuple[tuple[str, int], ...]] | None = None
        self._corrections: dict[tuple[str, bool], str | None] = {}
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

    # ------------------------------------------------------------- ortografía

    def _spelling_terms(self) -> dict[str, tuple[tuple[str, int], ...]]:
        """Términos alfabéticos del índice sin Porter, agrupados por su primera letra."""
        if self._spelling is None:
            with self._lock("spelling"):
                if self._spelling is None:
                    groups: dict[str, list[tuple[str, int]]] = {}
                    for term, docs in self._index(False).postings.items():
                        if len(term) >= 3 and term.isascii() and term.isalpha():
                            groups.setdefault(term[0], []).append((term, len(docs)))
                    self._spelling = {key: tuple(value) for key, value in groups.items()}
        return self._spelling

    def top_terms(self, limit: int) -> tuple[str, ...]:
        """Los `limit` términos alfabéticos más frecuentes (más documentos primero)."""
        terms = [entry for group in self._spelling_terms().values() for entry in group]
        terms.sort(key=lambda entry: (-entry[1], entry[0]))
        return tuple(term for term, _ in terms[:limit])

    def _closest_term(self, word: str, *, known: bool = False) -> str | None:
        """Término del corpus más probable para `word`, o None si no hay uno convincente.

        Puntúa ln(df) − 2 × distancia de edición: hasta 1 en palabras de hasta 7 letras
        y 2 en las más largas. Completar una palabra empezada (desde 4 letras) cuenta
        como distancia 1: "diabet" sugiere "diabetes". Si `word` ya está en el corpus
        (`known`), solo vale añadir la letra que falta ("morgage" → "mortgage"): así no
        se cambian plurales, conjugaciones ni grafías británicas.
        """
        key = (word, known)
        if key not in self._corrections:
            if len(self._corrections) >= _CORRECTIONS_CACHE:  # lo escrito por usuarios no crece sin fin
                self._corrections.clear()
            limit = 1 if len(word) <= 7 else 2
            candidates = []
            for term, df in self._spelling_terms().get(word[0], ()):
                if known:
                    if len(term) != len(word) + 1 or term.startswith(word):
                        continue
                    distance = _edit_distance(word, term, 1)
                else:
                    distance = _edit_distance(word, term, limit)
                    if len(word) >= 4 and term.startswith(word) and term != word:
                        distance = 1 if distance is None else min(distance, 1)
                if distance:  # None: demasiado lejos; 0: es la misma palabra
                    candidates.append((math.log(df) - 2 * distance, term))
            # Mayor puntuación; a igualdad, el término alfabéticamente menor.
            best = min(candidates, key=lambda c: (-c[0], c[1]), default=None)
            self._corrections[key] = None if best is None else best[1]
        return self._corrections[key]

    def correct_query(self, query: str) -> str | None:
        """La consulta con sus palabras mal escritas corregidas, o None si no hace falta.

        Solo se revisan palabras en minúsculas de 3 letras o más que el preprocesamiento
        conserva: las que llevan mayúsculas suelen ser siglas o nombres propios (EWU,
        Dalbar). Se corrigen las que no están en el índice sin Porter y las que están en
        muy pocos documentos si hay una forma parecida mucho más frecuente.
        """
        try:
            postings = self._index(False).postings
        except Exception:  # noqa: BLE001 - sin índice no hay sugerencia, la búsqueda sigue
            return None
        changed = False

        def fix(match: re.Match[str]) -> str:
            nonlocal changed
            word = match.group()
            terms = preprocess(word) if len(word) >= 3 and word.islower() else []
            if not terms:
                return word
            df = min(len(postings.get(term, ())) for term in terms)
            if df >= _RARE_DF:
                return word
            correction = self._closest_term(word, known=df > 0)
            if correction is None or (df and len(postings[correction]) < _MORE_COMMON * df):
                return word
            changed = True
            return correction

        fixed = _WORD.sub(fix, query)
        return fixed if changed else None

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

    # ------------------------------------------------------------------- grafo

    def query_graph(
        self, query: str, doc_ids: Sequence[str], *, neighbors: int = 2,
        linked: int | None = None,
    ) -> QueryGraph:
        """Grafo 3D de la consulta y esos documentos, calculado en el momento.

        Usa siempre los embeddings MiniLM, sea cual sea el modelo que produjo el
        ranking: sitúa los resultados por significado, no cambia su orden. Codifica
        la consulta (una consulta corta) y lee los vectores ya guardados de esos
        documentos. Necesita el modelo denso; si no está, lanza EngineError.
        """
        if not query or not query.strip():
            raise EmptyQueryError("Escribe una consulta para buscar.")
        retriever = self.get_retriever("dense")
        try:
            return build_query_graph(
                retriever.encode_query(query), doc_ids, retriever.vectors(doc_ids),
                neighbors=neighbors, linked=linked,
            )
        except Exception as error:
            raise ModelUnavailableError(f"No se pudo calcular el grafo: {error}") from error
