"""Lectura de los reportes de #11, sin corpus, recuperación ni reevaluación.

No hay caché: se validan los dos archivos en cada ejecución de la vista. El
validador de #12 exige k=10 y qids externos, por eso aquí solo se comparten sus
configuraciones y su tolerancia. Las medias se contrastan con fsum, rtol=0 y
atol=1e-12 para admitir únicamente diferencias de redondeo de coma flotante.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from ris.comparison import CONFIGURATIONS, METRICS, TOLERANCE
from scripts.evaluate import (
    AP_CUTOFF,
    CSV_COLUMNS,
    CSV_NAME,
    MODELS,
    SCHEMA_VERSION,
    SPLIT,
    SUMMARY_NAME,
    validate_cutoffs,
    verify_reports,
)


class ReportError(ValueError):
    """Reporte ausente, ilegible o incompatible; no se pueden mostrar resultados."""


@dataclass(frozen=True)
class MetricsReport:
    rows: tuple[dict, ...]
    summary: dict
    qids: tuple[str, ...]
    csv_bytes: bytes
    summary_bytes: bytes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def _fields(value: object, name: str, fields: dict[str, type | tuple]) -> None:
    _require(type(value) is dict, f"{name} debe ser un objeto.")
    for key, expected in fields.items():
        allowed = expected if isinstance(expected, tuple) else (expected,)
        _require(key in value and type(value[key]) in allowed,
                 f"Falta {name}.{key} o tiene un tipo incompatible.")


def _integer(value: int, name: str, minimum: int = 0) -> None:
    _require(type(value) is int and value >= minimum,
             f"{name} debe ser un entero mayor o igual a {minimum}.")


def _digest(value: object, name: str) -> None:
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name} debe contener un SHA-256 válido.")


def _metric(value: object, name: str) -> None:
    _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
             f"{name}: se esperaba un número finito entre 0 y 1.")


def _texts(value: object, name: str, keys: tuple[str, ...]) -> None:
    _fields(value, name, dict.fromkeys(keys, str))
    _require(all(value[key].strip() for key in keys), f"{name} contiene texto vacío.")


def _validate_provenance(summary: dict) -> None:
    provenance, counts = summary["provenance"], summary["counts"]
    _fields(provenance, "provenance", {
        "dataset": list, "artifacts_dir_files": dict, "nltk_resources": dict,
        "metrics_per_query_sha256": str, "query_ids_sha256": str,
    })
    for key in ("metrics_per_query_sha256", "query_ids_sha256"):
        _digest(provenance[key], f"provenance.{key}")
    _require(len(provenance["dataset"]) == 3, "Se esperaban corpus, consultas y qrels.")
    for item in provenance["dataset"]:
        _texts(item, "dataset", ("repo", "revision", "file", "sha256"))
        _digest(item["sha256"], "dataset.sha256")
    _require(len({(d["repo"], d["file"]) for d in provenance["dataset"]}) == 3,
             "Hay archivos de datos duplicados en la procedencia.")
    nltk = provenance["nltk_resources"]
    _fields(nltk, "nltk_resources", {"version": str, "files": dict})
    _require(bool(nltk["version"].strip()) and bool(nltk["files"]),
             "Falta la procedencia de los recursos NLTK.")
    for name, digest in nltk["files"].items():
        _require(bool(name.strip()), "Recurso NLTK sin nombre.")
        _digest(digest, "nltk_resources.files")
    artifacts = provenance["artifacts_dir_files"]
    for name, stem in (("index_no_stem.json", False), ("index_stem.json", True)):
        _fields(artifacts.get(name), name, {
            "sha256": str, "format_version": int, "stem": bool,
            "n_docs": int, "n_terms": int,
        })
        entry = artifacts[name]
        _digest(entry["sha256"], name)
        _integer(entry["format_version"], "format_version", 1)
        _integer(entry["n_terms"], "n_terms")
        _require(entry["stem"] is stem and entry["n_docs"] == counts["n_corpus_docs"],
                 "La procedencia de los índices no coincide con stem o el corpus.")
    dense = artifacts.get("dense/metadata.json")
    _fields(dense, "dense/metadata.json", {
        "sha256": str, "format_version": int, "model_revision": str,
        "corpus_sha256": str, "doc_ids_sha256": str, "n_docs": int,
        "file_sha256": dict, "build_batch_size": int, "build_threads": int,
        "build_library_versions": dict,
    })
    for name in ("sha256", "corpus_sha256", "doc_ids_sha256"):
        _digest(dense[name], f"dense.{name}")
    _texts(dense["file_sha256"], "dense.file_sha256", ("embeddings.npy", "index.faiss"))
    for name in ("embeddings.npy", "index.faiss"):
        _digest(dense["file_sha256"][name], name)
    for name in ("format_version", "build_batch_size", "build_threads"):
        _integer(dense[name], f"dense.{name}", 1)
    _require(dense["n_docs"] == counts["n_corpus_docs"],
             "El tamaño del corpus denso no coincide con counts.")
    for versions in (summary["versions"], dense["build_library_versions"]):
        _require(bool(versions) and all((type(v) is str and bool(v.strip())) or v is None
                                       for v in versions.values()),
                 "Versiones de bibliotecas incompatibles.")
    _texts(summary["versions"], "versions", ("python",))
    _require(dense["model_revision"] ==
             summary["config"]["hyperparameters"]["dense"]["model_revision"],
             "La revisión del modelo denso no coincide con su configuración.")


def _validate_summary(summary: object) -> None:
    _fields(summary, "resumen", {
        "schema_version": int, "config": dict, "counts": dict,
        "provenance": dict, "versions": dict, "results": list,
    })
    _require(summary["schema_version"] == SCHEMA_VERSION,
             f"Versión de esquema incompatible; se admite {SCHEMA_VERSION}.")
    config, counts = summary["config"], summary["counts"]
    _fields(config, "config", {
        "split": str, "k": int, "depth": int, "ap_cutoff": int,
        "models": list, "stem_values": dict, "metrics": dict,
        "preprocessing": dict, "hyperparameters": dict,
        "query_order": str, "row_order": str, "ranking_order": str, "relevance": str,
    })
    _require(config["split"] == SPLIT and config["ap_cutoff"] == AP_CUTOFF,
             f"Esquema incompatible: se requiere split {SPLIT} y AP@{AP_CUTOFF}.")
    try:
        validate_cutoffs(config["k"], config["depth"])
    except ValueError as error:
        raise ReportError(f"Cortes de evaluación incompatibles: {error}") from error
    _require(config["models"] == list(MODELS) and config["stem_values"] == {
        model: [stem for m, stem in CONFIGURATIONS if m == model] for model in MODELS
    }, "Los modelos o variantes Porter no corresponden a las siete configuraciones.")
    _texts(config, "config", ("query_order", "row_order", "ranking_order", "relevance"))
    _texts(config["metrics"], "config.metrics", (*METRICS, "map_at_1000"))
    _texts(config["preprocessing"], "config.preprocessing",
           ("classical", "dense", "known_limitation"))
    hyper = config["hyperparameters"]
    _fields(hyper, "hyperparameters", dict.fromkeys(MODELS, dict))
    _texts(hyper["jaccard"], "jaccard", ("representation", "score", "candidates"))
    _texts(hyper["tfidf"], "tfidf", ("tf", "idf", "normalization", "candidates"))
    _texts(hyper["bm25"], "bm25", ("idf", "term_score", "candidates"))
    _fields(hyper["bm25"], "bm25", {"k1": (int, float), "b": (int, float)})
    _require(math.isfinite(hyper["bm25"]["k1"]) and hyper["bm25"]["k1"] > 0,
             "k1 de BM25 debe ser positivo y finito.")
    _metric(hyper["bm25"]["b"], "b de BM25")
    _texts(hyper["dense"], "dense", (
        "model_name", "model_revision", "input", "normalization", "index_type",
        "document_text", "score", "candidates",
    ))
    for name in ("dimension", "max_seq_length"):
        _integer(hyper["dense"].get(name), f"dense.{name}", 1)
    count_names = (
        "n_queries", "n_queries_without_relevant", "n_qrel_pairs", "n_relevant_pairs",
        "n_relevant_docs_distinct", "n_relevant_docs_not_in_corpus", "n_corpus_docs", "n_rows",
    )
    _fields(counts, "counts", dict.fromkeys(count_names, int))
    for name in count_names:
        _integer(counts[name], f"counts.{name}", 1 if name in ("n_queries", "n_rows") else 0)
    _require(counts["n_queries_without_relevant"] <= counts["n_queries"] and
             counts["n_queries"] - counts["n_queries_without_relevant"] <=
             counts["n_relevant_pairs"] <= counts["n_qrel_pairs"] and
             counts["n_relevant_docs_not_in_corpus"] <= counts["n_relevant_docs_distinct"] <=
             counts["n_relevant_pairs"] and counts["n_relevant_docs_distinct"] -
             counts["n_relevant_docs_not_in_corpus"] <= counts["n_corpus_docs"],
             "Los conteos de consultas, corpus y relevancia son incompatibles entre sí.")
    _validate_provenance(summary)


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise ReportError(f"Falta {path.name} en la carpeta de reportes.") from error
    except OSError as error:
        raise ReportError(f"No se puede leer {path.name}; compruebe sus permisos.") from error


def _unique_object(pairs: list[tuple]) -> dict:
    value = {}
    for key, item in pairs:
        _require(key not in value, f"El JSON repite el campo {key}.")
        value[key] = item
    return value


def _invalid_number(value: str) -> None:
    raise ReportError(f"El JSON contiene un número no finito: {value}.")


def load_reports(directory: Path) -> MetricsReport:
    """Valida un par completo o lanza ReportError, sin devolver resultados parciales.

    verify_reports es obligatorio por el contrato de la app; además se comprueban
    estructura, cobertura, hash de qids y medias. Una segunda lectura detecta si
    los archivos cambiaron mientras se validaban (no son una transacción conjunta).
    """
    raw, metadata = _read(directory / CSV_NAME), _read(directory / SUMMARY_NAME)
    try:
        summary = json.loads(metadata.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=_invalid_number)
        _validate_summary(summary)
        verify_reports(directory)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        _require(reader.fieldnames == list(CSV_COLUMNS),
                 "Las columnas del CSV no corresponden al esquema de evaluación.")
        grouped = {key: {} for key in CONFIGURATIONS}
        rows = []
        for row in reader:
            _require(None not in row and all(value is not None for value in row.values()),
                     "Hay una fila CSV con campos faltantes o sobrantes.")
            key, qid = (row["model"], row["stem"]), row["qid"]
            _require(key in grouped, "Modelo o variante Porter inválidos en el CSV.")
            _require(bool(qid.strip()) and qid == qid.strip() and
                     not any(char in qid for char in "\r\n"),
                     "El qid debe ser texto no vacío, sin espacios exteriores ni saltos de línea.")
            _require(qid not in grouped[key], "Hay filas duplicadas por modelo, Porter y qid.")
            _require(row["k"] == str(summary["config"]["k"]),
                     "El corte k del CSV no coincide con la configuración del JSON.")
            row["k"] = summary["config"]["k"]
            for metric in METRICS:
                row[metric] = float(row[metric])
                _metric(row[metric], metric)
            grouped[key][qid] = row
            rows.append(row)
        qids = tuple(sorted(grouped[CONFIGURATIONS[0]]))
        _require(bool(qids) and all(set(group) == set(qids) for group in grouped.values()),
                 "Faltan consultas: todas las configuraciones deben cubrir los mismos qids.")
        counts = summary["counts"]
        _require(counts["n_rows"] == len(rows) == len(CONFIGURATIONS) * len(qids) and
                 counts["n_queries"] == len(qids),
                 "Los conteos declarados no coinciden con las filas y consultas del CSV.")
        qid_bytes = json.dumps(qids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        _require(hashlib.sha256(qid_bytes).hexdigest() == summary["provenance"]["query_ids_sha256"],
                 "Los qids no coinciden con la huella de consultas de la evaluación.")
        entries = {}
        always_zero = set(qids)
        for entry in summary["results"]:
            _fields(entry, "results", {
                "model": str, "stem": str, "n_queries": int, "n_empty_rankings": int,
                "rankings_sha256": str, "mean_precision_at_k": (int, float),
                "mean_recall_at_k": (int, float), "map_at_1000": (int, float),
            })
            key = entry["model"], entry["stem"]
            _require(key in grouped and key not in entries,
                     "Configuración inválida o repetida en los resultados del JSON.")
            _require(entry["n_queries"] == len(qids) and
                     0 <= entry["n_empty_rankings"] <= len(qids),
                     "Conteos incoherentes en los resultados de una configuración.")
            _digest(entry["rankings_sha256"], "results.rankings_sha256")
            for metric, target in zip(METRICS, (
                "mean_precision_at_k", "mean_recall_at_k", "map_at_1000",
            ), strict=True):
                _metric(entry[target], target)
                actual = math.fsum(row[metric] for row in grouped[key].values()) / len(qids)
                _require(math.isclose(actual, entry[target], rel_tol=0, abs_tol=TOLERANCE),
                         f"La media de {metric} del CSV no coincide con el JSON ({key[0]}/{key[1]}).")
            zero_qids = {qid for qid, row in grouped[key].items()
                         if all(row[m] == 0 for m in METRICS)}
            _require(entry["n_empty_rankings"] <= len(zero_qids),
                     "El conteo de rankings vacíos no concuerda con las métricas.")
            always_zero.intersection_update(zero_qids)
            entries[key] = entry
        _require(set(entries) == set(CONFIGURATIONS),
                 "Faltan resultados globales de alguna de las siete configuraciones.")
        _require(counts["n_queries_without_relevant"] <= len(always_zero),
                 "El conteo de consultas sin relevantes supera los qids con métricas cero "
                 "en todas las configuraciones.")
        _require(hashlib.sha256(raw).hexdigest() == summary["provenance"]["metrics_per_query_sha256"],
                 "El SHA-256 del CSV no coincide con el JSON.")
        _require(raw == _read(directory / CSV_NAME) and metadata == _read(directory / SUMMARY_NAME),
                 "Los reportes cambiaron durante la lectura. Vuelva a intentar.")
    except ReportError:
        raise
    except (ValueError, UnicodeError, csv.Error, OSError, OverflowError,
            KeyError, TypeError, RecursionError) as error:
        raise ReportError("Los reportes tienen un formato inválido o no forman un par coherente.") from error
    return MetricsReport(tuple(rows), summary, qids, raw, metadata)
