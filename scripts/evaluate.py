"""Evaluación reproducible de los cuatro modelos sobre las consultas test de FiQA.

Escribe ``metrics_per_query.csv`` y ``metrics_summary.json`` (esquema en
``reports/README.md``). Las métricas salen de ``ris.evaluation``; aquí solo se
recuperan los rankings, se ordenan las filas y se documenta la procedencia.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import nltk

from ris import data
from ris.data import Doc, load_corpus, load_qrels, load_queries
from ris.evaluation import (
    average_precision,
    mean_average_precision,
    precision_at_k,
    recall_at_k,
)
from ris.index import InvertedIndex
from ris.preprocessing import preprocess, setup_nltk
from ris.retrievers import (
    BM25Retriever,
    DenseRetriever,
    JaccardRetriever,
    Result,
    Retriever,
    TfidfRetriever,
)
from ris.retrievers.dense import _json_hash

SCHEMA_VERSION = 1
SPLIT = "test"
# MAP@1000: el corte de AP del contrato (docs/ESPECIFICACION.md, AGENTS.md).
AP_CUTOFF = 1000
DEFAULT_K = 10
MODELS = ("jaccard", "tfidf", "bm25", "dense")
STEM_ORDER = ("false", "true", "na")
CSV_COLUMNS = ("model", "stem", "qid", "k", "precision", "recall", "ap_at_1000")
CSV_NAME = "metrics_per_query.csv"
SUMMARY_NAME = "metrics_summary.json"
# InvertedIndex.load rechaza cualquier otra versión de formato.
INDEX_FORMAT_VERSION = 1
INDEX_SAMPLE_SIZE = 500
_PUNKT_FILES = (
    "abbrev_types.txt", "collocations.tab", "ortho_context.tab", "sent_starters.txt",
)
BUILD_HINT = (
    "Genérelos con: python scripts/build_artifacts.py (sin Porter), "
    "python scripts/build_artifacts.py --stem (con Porter) y "
    "python scripts/build_artifacts.py --dense (MiniLM + FAISS)"
)
PACKAGES = (
    "faiss-cpu", "huggingface_hub", "nltk", "numpy", "pyarrow",
    "sentence-transformers", "torch", "retrieval-information-system",
)
_RETRIEVER_CLASSES = {
    "jaccard": JaccardRetriever, "tfidf": TfidfRetriever, "bm25": BM25Retriever,
}


def say(message: str) -> None:
    """Progreso por stdout, sin búfer para verlo en tiempo real."""
    print(message, flush=True)


def relevant_sets(qrels: Mapping[str, Mapping[str, int]]) -> dict[str, set[str]]:
    """Qrels a ``{qid: {doc_id con score > 0}}``, con qid en orden ascendente."""
    return {
        qid: {doc_id for doc_id, score in docs.items() if score > 0}
        for qid, docs in sorted(qrels.items())
    }


def select_queries(
    queries: Mapping[str, str], relevant: Mapping[str, set[str]]
) -> dict[str, str]:
    """Consultas de los qrels, en orden ascendente de qid (como cadena)."""
    missing = sorted(set(relevant) - set(queries))
    if missing:
        raise ValueError(f"Los qrels referencian consultas ausentes: {missing[:5]}")
    return {qid: queries[qid] for qid in sorted(relevant)}


def validate_cutoffs(k: int, depth: int) -> None:
    """k y depth enteros positivos; depth >= 1000 (corte de AP) y k <= depth."""
    for name, value in (("k", k), ("depth", depth)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} debe ser un entero positivo, se recibió {value!r}")
    if depth < AP_CUTOFF:
        raise ValueError(
            f"depth ({depth}) no puede ser menor que {AP_CUTOFF}: la columna "
            f"ap_at_1000 y MAP@1000 exigen rankings de al menos {AP_CUTOFF} posiciones"
        )
    if k > depth:
        raise ValueError(f"k ({k}) no puede superar depth ({depth})")


def corpus_sha256(corpus: Mapping[str, Doc]) -> str:
    """Hash del corpus con la misma fórmula con la que DenseRetriever.build lo guarda."""
    return _json_hash([
        [doc_id, corpus[doc_id].title, corpus[doc_id].text] for doc_id in sorted(corpus)
    ])


def sample_doc_ids(doc_ids: list[str], sample_size: int) -> list[str]:
    """Muestra determinista: ids ordenados, paso uniforme, con el primero y el último."""
    ordered = sorted(doc_ids)
    if len(ordered) <= sample_size:
        return ordered
    if sample_size == 1:
        return ordered[:1]
    last = len(ordered) - 1
    positions = sorted({round(i * last / (sample_size - 1)) for i in range(sample_size)})
    return [ordered[position] for position in positions]


def verify_index_sample(
    index: InvertedIndex, corpus: Mapping[str, Doc], name: str = "el índice",
    sample_size: int = INDEX_SAMPLE_SIZE,
) -> int:
    """Comprueba por muestra que el índice coincide con corpus y preprocesamiento actuales.

    Recalcula, con la misma entrada que ``InvertedIndex.build``, longitud y
    frecuencias de cada documento de la muestra. Es una comprobación por muestra,
    no total: no garantiza que todo el índice sea correcto. Devuelve el tamaño de
    la muestra comprobada.
    """
    sample = sample_doc_ids(list(corpus), sample_size)
    for doc_id in sample:
        doc = corpus[doc_id]
        tokens = preprocess(f"{doc.title} {doc.text}", stem=index.stem)
        frequencies = Counter(tokens)
        # Si la longitud coincide y cada tf recalculado coincide, no queda ningún
        # otro término con tf > 0 en ese documento.
        if index.doc_lengths.get(doc_id) != len(tokens) or any(
            index.postings.get(term, {}).get(doc_id) != tf
            for term, tf in frequencies.items()
        ):
            raise ValueError(
                f"{name} no corresponde al corpus o al preprocesamiento actuales "
                f"(comprobación por muestra de {len(sample)} documentos, no total; "
                f"primer desacuerdo en el documento {doc_id}). {BUILD_HINT}."
            )
    return len(sample)


def retrieve_rankings(
    retriever: Retriever,
    queries: Mapping[str, str],
    depth: int,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[Result]]:
    """Ranking de profundidad ``depth`` por consulta (puede tener menos de depth)."""
    rankings: dict[str, list[Result]] = {}
    for done, (qid, text) in enumerate(queries.items(), start=1):
        results = retriever.search(text, k=depth)
        if len(results) > depth:
            raise ValueError(f"El retriever devolvió {len(results)} > depth={depth}")
        rankings[qid] = results
        if progress is not None:
            progress(done, len(queries))
    return rankings


def rankings_sha256(rankings: Mapping[str, list[Result]], qids: list[str]) -> str:
    """SHA-256 de una serialización canónica de todos los rankings (con scores repr)."""
    digest = hashlib.sha256()
    for qid in qids:
        entry = [qid, [[r.doc_id, repr(r.score)] for r in rankings.get(qid, [])]]
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        digest.update((line + "\n").encode("utf-8"))
    return digest.hexdigest()


def score_rankings(
    rankings: Mapping[str, list[Result]], relevant: Mapping[str, set[str]], k: int
) -> list[dict]:
    """Una fila por consulta de los qrels; sin ranking equivale a ranking vacío."""
    rows = []
    for qid, relevant_docs in relevant.items():
        ranking = [result.doc_id for result in rankings.get(qid, [])]
        rows.append({
            "qid": qid,
            "precision": precision_at_k(ranking, relevant_docs, k),
            "recall": recall_at_k(ranking, relevant_docs, k),
            "ap_at_1000": average_precision(ranking, relevant_docs, AP_CUTOFF),
        })
    return rows


def evaluate_configuration(
    model: str,
    stem: str,
    retriever: Retriever,
    queries: Mapping[str, str],
    relevant: Mapping[str, set[str]],
    *,
    k: int,
    depth: int,
    rankings_path: Path | None = None,
    out: Callable[[str], None] = say,
) -> tuple[list[dict], dict]:
    """Evalúa un (modelo, stem): devuelve sus filas CSV y su entrada de resumen."""
    validate_cutoffs(k, depth)
    label = f"{model} | stem={stem}"
    started = perf_counter()

    def progress(done: int, total: int) -> None:
        if done == total or done % 100 == 0:
            out(f"  [{label}] consultas: {done}/{total} | {perf_counter() - started:.1f} s")

    rankings = retrieve_rankings(retriever, queries, depth, progress)
    rows = score_rankings(rankings, relevant, k)
    for row in rows:
        row.update(model=model, stem=stem, k=k)
    run = {qid: [result.doc_id for result in results] for qid, results in rankings.items()}
    qids = list(relevant)
    entry = {
        "model": model,
        "stem": stem,
        "n_queries": len(rows),
        "n_empty_rankings": sum(1 for qid in qids if not rankings.get(qid)),
        "mean_precision_at_k": math.fsum(r["precision"] for r in rows) / len(rows),
        "mean_recall_at_k": math.fsum(r["recall"] for r in rows) / len(rows),
        "map_at_1000": mean_average_precision(run, relevant),
        "rankings_sha256": rankings_sha256(rankings, qids),
    }
    if rankings_path is not None:
        dump_rankings(rankings_path, rankings, qids)
    out(
        f"  [{label}] listo en {perf_counter() - started:.1f} s | "
        f"P@{k}={entry['mean_precision_at_k']:.4f} R@{k}={entry['mean_recall_at_k']:.4f} "
        f"MAP@{AP_CUTOFF}={entry['map_at_1000']:.4f}"
    )
    return rows, entry


def sort_rows(rows: list[dict]) -> list[dict]:
    """Orden documentado: modelo (jaccard, tfidf, bm25, dense), stem false/true/na, qid."""
    return sorted(
        rows,
        key=lambda r: (MODELS.index(r["model"]), STEM_ORDER.index(r["stem"]), r["qid"]),
    )


def render_csv(rows: list[dict]) -> str:
    """CSV con floats en repr completo y saltos de línea ``\\n``."""
    lines = [",".join(CSV_COLUMNS)]
    for row in sort_rows(rows):
        lines.append(",".join((
            row["model"], row["stem"], row["qid"], str(row["k"]),
            repr(row["precision"]), repr(row["recall"]), repr(row["ap_at_1000"]),
        )))
    return "\n".join(lines) + "\n"


def render_summary(summary: dict) -> str:
    return json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"


def write_atomically(files: Mapping[Path, str]) -> None:
    """Prepara todos los temporales y solo entonces los mueve, en el orden dado.

    Los reemplazos no son atómicos en conjunto: si el segundo falla queda el primer
    archivo nuevo y el segundo viejo. ``verify_reports`` lo detecta.
    """
    temporaries: list[tuple[Path, Path]] = []
    try:
        for destination, content in files.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(
                dir=destination.parent, prefix=destination.name + ".", suffix=".tmp"
            )
            temporaries.append((Path(name), destination))
            with os.fdopen(handle, "wb") as file:
                file.write(content.encode("utf-8"))
        for temporary, destination in temporaries:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in temporaries:
            temporary.unlink(missing_ok=True)


def verify_reports(output_dir: Path) -> None:
    """Relee el CSV y el JSON y comprueba que el JSON describe exactamente ese CSV."""
    csv_path, summary_path = output_dir / CSV_NAME, output_dir / SUMMARY_NAME
    again = "Ejecute de nuevo `python scripts/evaluate.py`"
    for path in (csv_path, summary_path):
        if not path.is_file():
            raise ValueError(f"Falta {path}. {again}.")
    csv_bytes = csv_path.read_bytes()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary["provenance"]["metrics_per_query_sha256"]
    if hashlib.sha256(csv_bytes).hexdigest() != expected:
        raise ValueError(
            f"{CSV_NAME} no corresponde a {SUMMARY_NAME} (SHA-256 distinto): "
            f"una escritura anterior quedó a medias. {again}."
        )
    rows_in_csv = len(csv_bytes.splitlines()) - 1  # sin la cabecera
    if rows_in_csv != summary["counts"]["n_rows"]:
        raise ValueError(f"{CSV_NAME} no tiene las filas que declara {SUMMARY_NAME}. {again}.")


def dump_rankings(path: Path, rankings: Mapping[str, list[Result]], qids: list[str]) -> None:
    """JSONL comprimido (gzip con mtime=0, reproducible): un ranking por línea."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    temporary = Path(name)
    try:
        with (
            os.fdopen(handle, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        ):
            for qid in qids:
                line = json.dumps(
                    {"qid": qid,
                     "ranking": [[r.doc_id, r.score] for r in rankings.get(qid, [])]},
                    ensure_ascii=False, separators=(",", ":"),
                )
                compressed.write((line + "\n").encode("utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_rankings_dir(rankings_dir: Path, output_dir: Path) -> None:
    """Los rankings pesados nunca se versionan: prohíbe volcarlos dentro de reports/."""
    forbidden = (
        output_dir.resolve(),
        Path("reports").resolve(),
        (Path(__file__).resolve().parents[1] / "reports"),
    )
    target = rankings_dir.resolve()
    for directory in forbidden:
        if target == directory or target.is_relative_to(directory):
            raise ValueError(
                f"--rankings-dir ({rankings_dir}) no puede estar dentro de reports/: "
                "los rankings completos no se versionan; use artifacts/ o similar"
            )


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def nltk_provenance() -> dict:
    """Versión de NLTK y SHA-256 de los recursos de ``preprocess`` (rutas relativas)."""
    setup_nltk()
    pointers = {
        f"tokenizers/punkt_tab/english/{filename}": nltk.data.find(
            f"tokenizers/punkt_tab/english/{filename}"
        )
        for filename in _PUNKT_FILES
    }
    pointers["corpora/stopwords/english"] = nltk.data.find("corpora/stopwords/english")
    files = {}
    for relative_name, pointer in sorted(pointers.items()):
        with pointer.open() as stream:
            files[relative_name] = hashlib.sha256(stream.read()).hexdigest()
    return {"version": version("nltk"), "files": files}


def dataset_provenance() -> list[dict]:
    """Revisiones fijadas y SHA-256 de los archivos que consume ris.data."""
    # Constantes privadas de ris.data, usadas solo como metadatos (como en ris.eda).
    sources = []
    for repo, revision, files in (
        (data._FIQA_REPO, data._FIQA_REVISION, (
            "corpus/corpus-00000-of-00001.parquet",
            "queries/queries-00000-of-00001.parquet",
        )),
        (data._QRELS_REPO, data._QRELS_REVISION, (data._QRELS_FILES[SPLIT],)),
    ):
        for filename in files:
            path = data._download_file(repo, filename, revision)
            sources.append({
                "repo": repo, "revision": revision, "file": filename,
                "sha256": file_sha256(path),
            })
    return sources


def hyperparameters(bm25_k1: float, bm25_b: float, dense_metadata: Mapping) -> dict:
    """Configuración de cada modelo, leída de las instancias o metadatos reales."""
    return {
        "jaccard": {
            "representation": "conjunto binario de términos únicos",
            "score": "|Q ∩ D| / |Q ∪ D|",
            "candidates": "documentos con al menos un término de la consulta",
        },
        "tfidf": {
            "tf": "1 + ln(f) si f > 0",
            "idf": "ln(N / df), sin suavizado",
            "normalization": "L2 (coseno) en documentos y consulta",
            "candidates": "documentos con coseno positivo",
        },
        "bm25": {
            "k1": bm25_k1,
            "b": bm25_b,
            "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
            "term_score": "idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))",
            "candidates": "documentos con al menos un término de la consulta",
        },
        "dense": {
            **{key: dense_metadata[key] for key in (
                "model_name", "model_revision", "dimension", "normalization",
                "index_type", "max_seq_length", "document_text",
            )},
            "score": "producto interno de vectores normalizados (coseno; puede ser negativo)",
            "candidates": "todos los documentos del corpus",
            "input": "texto original, sin preprocesamiento léxico ni Porter",
        },
    }


def build_config(k: int, depth: int, hyper: dict) -> dict:
    return {
        "split": SPLIT,
        "k": k,
        "depth": depth,
        "ap_cutoff": AP_CUTOFF,
        "models": list(MODELS),
        "stem_values": {
            "jaccard": ["false", "true"], "tfidf": ["false", "true"],
            "bm25": ["false", "true"], "dense": ["na"],
        },
        "query_order": "qid ascendente como cadena",
        "row_order": "modelo (jaccard, tfidf, bm25, dense), stem (false, true, na), qid",
        "ranking_order": "score descendente; empate por doc_id ascendente",
        "relevance": "documentos con score > 0 en los qrels del split",
        "metrics": {
            "precision": "relevantes en el top-k / k",
            "recall": "relevantes en el top-k / todos los relevantes de la consulta",
            "ap_at_1000": "AP sobre las primeras 1000 posiciones; divide por todos los relevantes",
            "map_at_1000": "promedio de ap_at_1000 sobre todas las consultas de los qrels",
        },
        "preprocessing": {
            "classical": (
                "ris.preprocessing.preprocess: minúsculas, nltk.word_tokenize, "
                "stopwords inglesas de NLTK, solo tokens str.isalnum(); Porter opcional; "
                "misma configuración en documentos y consulta"
            ),
            "dense": "ninguno: título + ' ' + texto originales",
            "known_limitation": (
                "preprocess descarta tokens no estrictamente alfanuméricos (guiones, "
                "decimales, contracciones) en documentos y consultas; su efecto en "
                "las métricas no se ha medido (corresponde a la issue #3)"
            ),
        },
        "hyperparameters": hyper,
    }


def run(
    *,
    artifacts_dir: Path,
    output_dir: Path,
    k: int = DEFAULT_K,
    depth: int = AP_CUTOFF,
    rankings_dir: Path | None = None,
    dense_encoder=None,
    out: Callable[[str], None] = say,
) -> dict:
    """Evalúa las 7 configuraciones y escribe el CSV y el JSON de forma atómica."""
    validate_cutoffs(k, depth)
    if rankings_dir is not None:
        check_rankings_dir(rankings_dir, output_dir)
    index_paths = {
        "false": artifacts_dir / "index_no_stem.json",
        "true": artifacts_dir / "index_stem.json",
    }
    dense_dir = artifacts_dir / "dense"
    required = [*index_paths.values(), dense_dir / "metadata.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos: {', '.join(missing)}. {BUILD_HINT}.")

    started = perf_counter()
    out("Fase 1/4 - Cargando FiQA (corpus, consultas y qrels test)...")
    corpus = load_corpus()
    qrels = load_qrels(SPLIT)
    relevant = relevant_sets(qrels)
    queries = select_queries(load_queries(SPLIT), relevant)
    if not queries:
        raise ValueError("No hay consultas para evaluar")
    qids = list(queries)
    all_relevant = {doc_id for docs in relevant.values() for doc_id in docs}
    counts = {
        "n_queries": len(queries),
        "n_queries_without_relevant": sum(1 for docs in relevant.values() if not docs),
        "n_qrel_pairs": sum(len(docs) for docs in qrels.values()),
        "n_relevant_pairs": sum(len(docs) for docs in relevant.values()),
        "n_relevant_docs_distinct": len(all_relevant),
        "n_relevant_docs_not_in_corpus": len(all_relevant - corpus.keys()),
        "n_corpus_docs": len(corpus),
    }
    out(f"  Corpus: {len(corpus):,} documentos; consultas test: {len(queries):,}.")

    # Antes de gastar minutos de cómputo: el denso debe corresponder a este corpus.
    dense = DenseRetriever.load(
        dense_dir, encoder=dense_encoder, expected_corpus_sha256=corpus_sha256(corpus)
    )

    def rankings_file(model: str, stem: str) -> Path | None:
        if rankings_dir is None:
            return None
        suffix = "" if stem == "na" else f"_{'stem' if stem == 'true' else 'no_stem'}"
        return rankings_dir / f"rankings_{model}{suffix}.jsonl.gz"

    rows: list[dict] = []
    results: list[dict] = []
    index_provenance: dict[str, dict] = {}
    bm25_params: tuple[float, float] | None = None
    for position, stem in enumerate(("false", "true"), start=2):
        out(f"Fase {position}/4 - Modelos clásicos {'con' if stem == 'true' else 'sin'} Porter...")
        path = index_paths[stem]
        index = InvertedIndex.load(path, expected_stem=stem == "true")
        if index.N != len(corpus) or index.doc_lengths.keys() != corpus.keys():
            raise ValueError(
                f"{path.name} no corresponde al corpus cargado "
                f"(N={index.N}, corpus={len(corpus)}). {BUILD_HINT}."
            )
        sampled = verify_index_sample(index, corpus, path.name)
        out(f"  {path.name}: coincide con el corpus y el preprocesamiento en una "
            f"muestra de {sampled} documentos (comprobación por muestra, no total).")
        index_provenance[path.name] = {
            "sha256": file_sha256(path), "format_version": INDEX_FORMAT_VERSION,
            "stem": index.stem, "n_docs": index.N, "n_terms": len(index.postings),
        }
        for model in ("jaccard", "tfidf", "bm25"):
            retriever = _RETRIEVER_CLASSES[model](index, stem=index.stem)
            if model == "bm25":
                bm25_params = (retriever.k1, retriever.b)
            model_rows, entry = evaluate_configuration(
                model, stem, retriever, queries, relevant, k=k, depth=depth,
                rankings_path=rankings_file(model, stem), out=out,
            )
            rows += model_rows
            results.append(entry)
        del index, retriever
    out("Fase 4/4 - Modelo denso (MiniLM + FAISS; una sola vez, sin Porter)...")
    model_rows, entry = evaluate_configuration(
        "dense", "na", dense, queries, relevant, k=k, depth=depth,
        rankings_path=rankings_file("dense", "na"), out=out,
    )
    rows += model_rows
    results.append(entry)

    expected_rows = len(queries) * len(results)
    if len(rows) != expected_rows:
        raise ValueError(f"Se esperaban {expected_rows} filas y hay {len(rows)}")
    results.sort(key=lambda r: (MODELS.index(r["model"]), STEM_ORDER.index(r["stem"])))
    metadata = dense.metadata
    counts["n_rows"] = len(rows)
    csv_text = render_csv(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": build_config(k, depth, hyperparameters(*bm25_params, metadata)),
        "counts": counts,
        "provenance": {
            "dataset": dataset_provenance(),
            "artifacts_dir_files": {
                **index_provenance,
                "dense/metadata.json": {
                    "sha256": file_sha256(dense_dir / "metadata.json"),
                    "model_revision": metadata["model_revision"],
                    "corpus_sha256": metadata["corpus_sha256"],
                    "doc_ids_sha256": metadata["doc_ids_sha256"],
                    "format_version": metadata["format_version"],
                    "n_docs": metadata["count"],
                    "file_sha256": metadata["file_sha256"],
                    "build_batch_size": metadata["batch_size"],
                    "build_threads": metadata["threads"],
                    "build_library_versions": metadata["library_versions"],
                },
            },
            "metrics_per_query_sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
            "nltk_resources": nltk_provenance(),
            "query_ids_sha256": _json_hash(qids),
        },
        "versions": package_versions(),
        "results": results,
    }
    # CSV primero y JSON después: el JSON lleva el hash del CSV exacto.
    write_atomically({
        output_dir / CSV_NAME: csv_text,
        output_dir / SUMMARY_NAME: render_summary(summary),
    })
    verify_reports(output_dir)
    out(f"Archivos escritos en {output_dir} ({len(rows):,} filas) | "
        f"tiempo total: {perf_counter() - started:.1f} s")
    return summary


def _positive_argument(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser un entero positivo")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa Jaccard, TF-IDF y BM25 (con y sin Porter) y el denso sobre las "
            "consultas test de FiQA y escribe reports/metrics_*."
        )
    )
    parser.add_argument("--k", type=_positive_argument, default=DEFAULT_K,
                        help="Corte de P@k y R@k (por defecto 10; debe ser <= --depth).")
    parser.add_argument("--depth", type=_positive_argument, default=AP_CUTOFF,
                        help=f"Profundidad del ranking (por defecto {AP_CUTOFF}; el MAP del "
                             f"contrato es @{AP_CUTOFF}, por lo que debe ser >= {AP_CUTOFF}).")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"),
                        help="Directorio con los artefactos de build_artifacts.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"),
                        help="Destino del CSV y del JSON (por defecto reports/).")
    parser.add_argument("--rankings-dir", type=Path,
                        help="Opcional: vuelca los rankings completos (JSONL gzip) aquí. "
                             "Debe estar fuera de reports/ (use artifacts/); nunca se versionan.")
    args = parser.parse_args()
    try:
        validate_cutoffs(args.k, args.depth)
        if args.rankings_dir is not None:
            check_rankings_dir(args.rankings_dir, args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    try:
        run(
            artifacts_dir=args.artifacts_dir, output_dir=args.output_dir, k=args.k,
            depth=args.depth, rankings_dir=args.rankings_dir,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
