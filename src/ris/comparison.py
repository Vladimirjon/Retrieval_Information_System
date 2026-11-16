"""Análisis descriptivo de #12; consume la evaluación, sin ajustar modelos."""

import csv
import hashlib
import io
import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

from ris.preprocessing import preprocess

CONFIGURATIONS = tuple(
    (model, stem)
    for model in ("jaccard", "tfidf", "bm25", "dense")
    for stem in (("na",) if model == "dense" else ("false", "true"))
)
METRICS = ("precision", "recall", "ap_at_1000")
TOLERANCE = 1e-12
BASELINE = ("bm25", "true")
GROUP_RULES = {
    "length_0_3": "0–3 tokens de preprocess(query, stem=False)",
    "length_4_7": "4–7 tokens de preprocess(query, stem=False)",
    "length_8_plus": "8 o más tokens de preprocess(query, stem=False)",
    "digits_yes": r"al menos un dígito en el texto original: re.search(r'\d', query)",
    "digits_no": "ningún dígito en el texto original",
    "uppercase_yes": r"forma de 2+ letras ASCII mayúsculas: re.search(r'\b[A-Z]{2,}\b', query)",
    "uppercase_no": "sin forma de 2+ letras ASCII mayúsculas",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_reports(directory: Path, qids: set[str]) -> tuple[list[dict], dict]:
    """Rechaza entradas incoherentes antes de interpretar las siete corridas."""
    raw = (directory / "metrics_per_query.csv").read_bytes()
    summary = json.loads((directory / "metrics_summary.json").read_text("utf-8"))
    if hashlib.sha256(raw).hexdigest() != summary["provenance"]["metrics_per_query_sha256"]:
        raise ValueError("SHA-256 del CSV distinto del declarado en #11")
    config = summary["config"]
    if type(summary["schema_version"]) is not int or summary["schema_version"] != 1 or any(
        type(config[key]) is not type(value) or config[key] != value
        for key, value in {"split": "test", "k": 10, "ap_cutoff": 1000}.items()
    ) or type(config["depth"]) is not int or config["depth"] < 1000:
        raise ValueError("Se exige esquema 1, test, P/R@10 y AP@1000")
    if any(type(qid) is not str or not qid for qid in qids):
        raise ValueError("Los qids deben ser cadenas no vacías")
    if config["models"] != ["jaccard", "tfidf", "bm25", "dense"] or config[
        "stem_values"
    ] != {model: list(stems) for model, stems in (
        ("jaccard", ("false", "true")), ("tfidf", ("false", "true")),
        ("bm25", ("false", "true")), ("dense", ("na",)),
    )}:
        raise ValueError("Modelos o variantes stem inesperados")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames != ["model", "stem", "qid", "k", *METRICS]:
        raise ValueError("Esquema CSV inesperado")
    rows, seen = [], set()
    for row in reader:
        key = row["model"], row["stem"], row["qid"]
        if key[:2] not in CONFIGURATIONS or key[2] not in qids or key in seen:
            raise ValueError("Configuración/qid inválido o duplicado")
        if row["k"] != "10":
            raise ValueError("Corte CSV distinto de 10")
        row["k"] = 10
        for metric in METRICS:
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
                raise ValueError("Métrica fuera de [0,1] o no finita")
        seen.add(key)
        rows.append(row)
    expected = {(model, stem, qid) for model, stem in CONFIGURATIONS for qid in qids}
    counts = summary["counts"]
    if seen != expected or any(type(counts[key]) is not int for key in ("n_rows", "n_queries")) \
            or counts["n_rows"] != len(rows) or counts["n_queries"] != len(qids):
        raise ValueError("Consultas faltantes o conteos incoherentes")
    entries = {(e["model"], e["stem"]): e for e in summary["results"]}
    if len(summary["results"]) != 7 or set(entries) != set(CONFIGURATIONS):
        raise ValueError("Se esperaban siete resultados únicos en el JSON")
    for model, stem in CONFIGURATIONS:
        selected = [r for r in rows if (r["model"], r["stem"]) == (model, stem)]
        entry = entries[model, stem]
        if entry["n_queries"] != len(qids):
            raise ValueError("n_queries de una configuración incoherente")
        for metric, target in zip(METRICS, (
            "mean_precision_at_k", "mean_recall_at_k", "map_at_1000"
        ), strict=True):
            actual = math.fsum(r[metric] for r in selected) / len(qids)
            if not math.isclose(actual, entry[target], rel_tol=0, abs_tol=TOLERANCE):
                raise ValueError(f"Media CSV/JSON distinta: {model}/{stem}/{metric}")
    return rows, summary


def query_groups(queries: dict[str, str]) -> dict[str, list[str]]:
    """Tres particiones observables, independientes de los resultados de modelos."""
    groups = {name: [] for name in GROUP_RULES}
    for qid, text in sorted(queries.items()):
        length = len(preprocess(text, stem=False))
        groups["length_0_3" if length <= 3 else "length_4_7" if length <= 7
               else "length_8_plus"].append(qid)
        groups["digits_yes" if re.search(r"\d", text) else "digits_no"].append(qid)
        groups["uppercase_yes" if re.search(r"\b[A-Z]{2,}\b", text)
               else "uppercase_no"].append(qid)
    return groups


def aggregate(rows: list[dict], qids: list[str]) -> list[dict]:
    selected = set(qids)
    result = []
    for model, stem in CONFIGURATIONS:
        subset = [r for r in rows if (r["model"], r["stem"]) == (model, stem)
                  and r["qid"] in selected]
        if len(subset) != len(selected):
            raise ValueError("Agregación con consultas faltantes o duplicadas")
        result.append({"model": model, "stem": stem, "n": len(subset), **{
            metric: math.fsum(r[metric] for r in subset) / len(subset) if subset else None
            for metric in METRICS
        }})
    return result


def paired_differences(rows: list[dict], target: tuple, reference: tuple,
                       qids: list[str]) -> dict:
    """Deltas target-reference pareados; percentiles lineales, empates ±1e-12."""
    lookup = {(r["model"], r["stem"], r["qid"]): r for r in rows}
    result = {"target": list(target), "reference": list(reference), "n": len(qids)}
    for metric in METRICS:
        deltas = [lookup[*target, qid][metric] - lookup[*reference, qid][metric]
                  for qid in sorted(qids)]
        base_mean = math.fsum(lookup[*reference, qid][metric] for qid in qids) / len(qids)
        mean = math.fsum(deltas) / len(deltas)
        result[metric] = {
            "mean_delta": mean,
            "relative_mean_delta": mean / base_mean if base_mean else None,
            "improved": sum(d > TOLERANCE for d in deltas),
            "worsened": sum(d < -TOLERANCE for d in deltas),
            "tied": sum(abs(d) <= TOLERANCE for d in deltas),
            "percentiles": dict(zip(("min", "p25", "median", "p75", "max"),
                                    np.quantile(deltas, [0, .25, .5, .75, 1]).tolist(),
                                    strict=True)),
        }
    return result


def select_cases(rows: list[dict]) -> dict[str, dict | None]:
    """Extremos de AP denso−BM25/Porter; empate exacto por qid lexicográfico."""
    lookup = {(r["model"], r["stem"], r["qid"]): r for r in rows}
    candidates = [{"qid": qid, "delta_ap_at_1000": row["ap_at_1000"]
                   - lookup[*BASELINE, qid]["ap_at_1000"]}
                  for (model, stem, qid), row in lookup.items() if model == "dense"]
    output = {}
    for name, sign in (("dense_improves", 1), ("dense_worsens", -1)):
        eligible = [r for r in candidates if sign * r["delta_ap_at_1000"] > TOLERANCE]
        output[name] = min(eligible, key=lambda r: (
            -sign * r["delta_ap_at_1000"], r["qid"]
        )) if eligible else None
    return output


def metrics_from_positions(positions: list[int | None], n_relevant: int) -> dict:
    """Referencia independiente exacta desde posiciones, sin llamar ris.evaluation."""
    if n_relevant < len(positions) or any(p is not None and p <= 0 for p in positions):
        raise ValueError("Posiciones o denominador inválidos")
    hits = sorted(p for p in positions if p is not None and p <= 1000)
    if len(hits) != len(set(hits)):
        raise ValueError("Posiciones duplicadas")
    top10 = sum(p <= 10 for p in hits)
    return {
        "precision": top10 / 10,
        "recall": top10 / n_relevant if n_relevant else 0.0,
        "ap_at_1000": float(sum((Fraction(i, p) for i, p in enumerate(hits, 1)),
                               Fraction()) / n_relevant) if n_relevant else 0.0,
    }


def case_evidence(qid, query, qrels, corpus, rankings, expected):
    """Solo top-10 y relevantes: posiciones, scores y fragmentos fieles con offsets."""
    relevant = sorted(doc_id for doc_id, score in qrels.items() if score > 0)
    evidence = {"qid": qid, "query": query, "n_relevant": len(relevant),
                "relevant_doc_ids": relevant, "rankings": {}}
    for label, ranking in rankings.items():
        ids = [r.doc_id for r in ranking]
        if len(ids) != len(set(ids)) or len(ids) > 1000 or ranking != sorted(
            ranking, key=lambda r: (-r.score, r.doc_id)
        ):
            raise ValueError("Ranking duplicado, desordenado o mayor de 1000")
        by_id = {r.doc_id: (position, r.score) for position, r in enumerate(ranking, 1)}
        positions = [by_id[doc_id][0] if doc_id in by_id else None for doc_id in relevant]
        metrics = metrics_from_positions(positions, len(relevant))
        for metric in METRICS:
            if not math.isclose(metrics[metric], expected[label][metric],
                                rel_tol=0, abs_tol=TOLERANCE):
                raise ValueError(f"Caso {qid}/{label}: {metric} difiere del CSV de #11")
        docs = []
        for doc_id in dict.fromkeys([*ids[:10], *relevant]):
            doc = corpus[doc_id]
            position, score = by_id.get(doc_id, (None, None))
            docs.append({
                "doc_id": doc_id, "rank": position, "score": score,
                "rank_status": "recuperado" if position else "no recuperado en top-1000",
                "judgment": "no juzgado" if doc_id not in qrels else (
                    "relevante" if qrels[doc_id] > 0 else "juzgado no relevante"),
                "qrel_score": qrels.get(doc_id), "title": doc.title,
                "snippet": doc.text[:600], "snippet_start": 0,
                "snippet_end": min(600, len(doc.text)), "text_length": len(doc.text),
            })
        serialized = [[r.doc_id, repr(r.score)] for r in ranking]
        evidence["rankings"][label] = {
            "n_retrieved": len(ranking), "metrics": metrics,
            "ranking_sha256": hashlib.sha256(json.dumps(
                serialized, separators=(",", ":")
            ).encode()).hexdigest(), "documents": docs,
        }
    return evidence
