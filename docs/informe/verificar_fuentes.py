"""Audita reportes y cifras del informe sin descargar datos ni ejecutar búsquedas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "66aea9d2849cf4d4e0ed58c3bfff1ce5b23bfaa0"
METRICS = ("precision", "recall", "ap_at_1000")
CONFIGS = [(m, s) for m in ("jaccard", "tfidf", "bm25", "dense")
           for s in (("na",) if m == "dense" else ("false", "true"))]
NAMES = {"jaccard": "Jaccard", "tfidf": "TF-IDF", "bm25": "BM25", "dense": "Denso"}
TOL = 1e-12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equal(actual: float, expected: float) -> None:
    require(math.isclose(actual, expected, abs_tol=TOL, rel_tol=0),
            f"Valores distintos: {actual} / {expected}")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)


def number(value: float, places: int = 6) -> str:
    return f"{value:.{places}f}".replace(".", ",")


def tables(source: str) -> list[list[list[str]]]:
    result = []
    for block in re.findall(r"(?:^\|.*\n?)+", source, flags=re.MULTILINE):
        rows = [[v.strip().strip("`") for v in line.strip().strip("|").split("|")]
                for line in block.splitlines()]
        result.append([rows[0], *rows[2:]])
    return result


def plain_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(https?://[^\s)]+\)", r"\1", text)
    lines = []
    for line in text.splitlines():
        if re.match(r"^\|[-: |]+\|$", line):
            continue
        if line.startswith("|"):
            line = " ".join(cell.strip() for cell in line.strip("|").split("|"))
        lines.append(re.sub(r"^#{1,3} ", "", line).replace("`", "").replace("*", ""))
    return re.sub(r"\s+", "", " ".join(lines))


def verify_pdf() -> dict:
    from pypdf import PdfReader

    path = HERE / "informe_tecnico.pdf"
    reader = PdfReader(path)
    require(len(reader.pages) == 5, "El PDF no tiene cinco páginas")
    require(reader.metadata.author == "Ismael Freire, Johann Passquel y Jorge Torres",
            "Los integrantes del PDF no coinciden")
    markdown = (HERE / "informe_tecnico.md").read_text(encoding="utf-8")
    blocks = markdown.split("<!-- pagebreak -->")
    expected_links = set(re.findall(r"\[[^\]]+\]\((https?://[^\s)]+)\)", markdown))
    links = set()
    for i, (page, source) in enumerate(zip(reader.pages, blocks, strict=True), 1):
        folio, _, body = page.extract_text().partition("\n")
        require(folio.strip() == str(i), f"Número de página distinto: página {i}")
        text = re.sub(r"\s+", "", body)
        require(text == plain_markdown(source), f"Texto fuente/PDF distinto en página {i}")
        require(abs(float(page.mediabox.width) - 595.276) < 0.01
                and abs(float(page.mediabox.height) - 841.890) < 0.01,
                "Tamaño de página distinto de A4")
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            if action.get("/URI"):
                links.add(str(action["/URI"]))
    require(links == expected_links and all(url.startswith("https://") for url in links),
            "Los enlaces HTTPS del PDF no coinciden con el Markdown")
    return {"pages": 5, "format": "A4", "sha256": digest(path.read_bytes()),
            "complete_text_matches_markdown_per_page": True, "links": sorted(links),
            "visual_review": "Debe revisarse también el renderizado; la extracción no valida el diseño."}


def verify() -> dict:
    paths = ["reports/metrics_per_query.csv", "reports/metrics_summary.json",
             "reports/comparison_summary.json", "reports/eda_summary.json",
             "deploy/artifacts.json"]
    files, hashes = {}, {}
    for path in paths:
        local = (ROOT / path).read_bytes()
        original = git_bytes(path)
        if path.endswith(".json"):
            require(json.loads(local) == json.loads(original), f"Contenido modificado: {path}")
            files[path] = json.loads(local)
        else:
            require(local == original, f"CSV distinto al snapshot: {path}")
        hashes[path] = {
            "sha256_git": digest(original), "sha256_worktree": digest(local),
            "only_line_endings_differ": local != original
            and local.replace(b"\r\n", b"\n") == original,
        }
    metrics = files[paths[1]]
    comparison = files[paths[2]]
    eda = files[paths[3]]
    manifest = files[paths[4]]
    raw_csv = (ROOT / paths[0]).read_bytes()
    require(digest(raw_csv) == metrics["provenance"]["metrics_per_query_sha256"],
            "Falla el hash CSV/JSON del proyecto")
    require(comparison["provenance"]["metrics_per_query_sha256"] == digest(raw_csv),
            "El análisis no procede del CSV actual")
    require(comparison["provenance"]["metrics_summary_sha256"] == hashes[paths[1]]["sha256_git"],
            "El análisis no procede del JSON versionado actual")
    require(metrics["schema_version"] == comparison["schema_version"] == 1,
            "Esquema inesperado")
    require(metrics["config"]["split"] == "test" and metrics["config"]["k"] == 10
            and metrics["config"]["depth"] == metrics["config"]["ap_cutoff"] == 1000,
            "Cortes inesperados")
    rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8"))))
    lookup = {}
    for row in rows:
        key = row["model"], row["stem"], row["qid"]
        require(key not in lookup and key[:2] in CONFIGS and row["k"] == "10",
                "Duplicado, modelo o corte inválido")
        for metric in METRICS:
            row[metric] = float(row[metric])
            require(math.isfinite(row[metric]) and 0 <= row[metric] <= 1,
                    "Métrica fuera de rango")
        lookup[key] = row
    qids = sorted({r["qid"] for r in rows})
    require(set(lookup) == {(*c, q) for c in CONFIGS for q in qids}, "Cobertura desigual")
    require(len(rows) == metrics["counts"]["n_rows"] == 4536 and len(qids) == 648,
            "Conteos inesperados")
    require(comparison["counts"] == metrics["counts"], "Conteos de #11 y #12 distintos")
    means = []
    for model, stem in CONFIGS:
        mean = {"model": model, "stem": stem, "n": len(qids), **{
            m: math.fsum(lookup[model, stem, q][m] for q in qids) / len(qids)
            for m in METRICS}}
        saved = next(r for r in metrics["results"]
                     if (r["model"], r["stem"]) == (model, stem))
        derived = next(r for r in comparison["overall"]
                       if (r["model"], r["stem"]) == (model, stem))
        for m, name in zip(METRICS, ("mean_precision_at_k", "mean_recall_at_k", "map_at_1000")):
            equal(mean[m], saved[name])
            equal(mean[m], derived[m])
        means.append(mean)
    for family in (("length_0_3", "length_4_7", "length_8_plus"),
                   ("digits_yes", "digits_no"), ("uppercase_yes", "uppercase_no")):
        members = [q for group in family for q in comparison["groups"][group]["qids"]]
        require(len(members) == len(set(members)) == len(qids) and set(members) == set(qids),
                "La familia de grupos no particiona test")
    for group in comparison["groups"].values():
        require(group["n"] == len(group["qids"]) and len(group["means"]) == 7,
                "Grupo incompleto")
        for entry in group["means"]:
            for m in METRICS:
                equal(math.fsum(lookup[entry["model"], entry["stem"], q][m]
                                for q in group["qids"]) / group["n"], entry[m])
    for pair in [*comparison["porter"], *comparison["versus_bm25_porter"]]:
        for m in METRICS:
            delta = [lookup[*pair["target"], q][m] - lookup[*pair["reference"], q][m]
                     for q in qids]
            mean_delta = math.fsum(delta) / len(qids)
            equal(mean_delta, pair[m]["mean_delta"])
            reference = math.fsum(lookup[*pair["reference"], q][m] for q in qids) / len(qids)
            equal(mean_delta / reference, pair[m]["relative_mean_delta"])
            require((sum(v > TOL for v in delta), sum(v < -TOL for v in delta),
                     sum(abs(v) <= TOL for v in delta))
                    == tuple(pair[m][key] for key in ("improved", "worsened", "tied")),
                    "Conteos pareados distintos")
    case_checks = {}
    deltas = {q: lookup["dense", "na", q]["ap_at_1000"]
              - lookup["bm25", "true", q]["ap_at_1000"] for q in qids}
    for label, sign in (("dense_improves", 1), ("dense_worsens", -1)):
        case = comparison["cases"][label]
        selected = min((q for q in qids if sign * deltas[q] > TOL),
                       key=lambda q: (-sign * deltas[q], q))
        require(case["qid"] == selected, "Caso distinto del extremo declarado")
        equal(case["delta_ap_at_1000"], deltas[selected])
        relevant = case["relevant_doc_ids"]
        require(case["n_relevant"] == len(set(relevant)), "Denominador de caso inválido")
        case_checks[selected] = {}
        for name, ranking in case["rankings"].items():
            by_id = {d["doc_id"]: d for d in ranking["documents"]}
            require(len(by_id) == len(ranking["documents"]), "Documento de caso duplicado")
            positions = sorted(by_id[d]["rank"] for d in relevant if by_id[d]["rank"] is not None)
            require(all(1 <= p <= 1000 for p in positions), "Posición fuera del corte")
            for doc_id in relevant:
                require(by_id[doc_id]["qrel_score"] == 1, "Juicio relevante inválido")
            hits = sum(p <= 10 for p in positions)
            actual = {"precision": hits / 10, "recall": hits / len(relevant),
                      "ap_at_1000": float(sum((Fraction(i, p) for i, p in enumerate(positions, 1)),
                                              Fraction()) / len(relevant))}
            model, stem = name.split("_")
            for m in METRICS:
                equal(actual[m], ranking["metrics"][m])
                equal(actual[m], lookup[model, stem, selected][m])
            case_checks[selected][name] = actual
    for key in ("lexical", "porter"):
        length = eda["documents"][key]["length"]
        distribution = length["distribution"]
        require(sum(distribution.values()) == eda["counts"]["documents"], "EDA: documentos")
        total = sum(int(n) * count for n, count in distribution.items())
        require(total == length["total"], "EDA: apariciones")
        equal(total / length["count"], length["mean"])
    reduction = 100 * (1 - eda["documents"]["porter"]["vocabulary_size"]
                       / eda["documents"]["lexical"]["vocabulary_size"])
    equal(reduction, eda["reductions_documents"]["vocabulary_porter_pct"])
    require(sum(v["size"] for v in manifest["files"].values()) == 261655898,
            "Tamaño de artefactos distinto")
    for path in ("index_no_stem.json", "index_stem.json", "dense/metadata.json"):
        expected = metrics["provenance"]["artifacts_dir_files"][path]["sha256"]
        require(manifest["files"][path]["sha256"] == expected, "Manifiesto no coincide con #11")
        require(comparison["provenance"]["local_artifacts_sha256"][path] == expected,
                "Artefacto de #12 distinto de #11")
    for filename in ("embeddings.npy", "index.faiss"):
        expected = metrics["provenance"]["artifacts_dir_files"]["dense/metadata.json"]["file_sha256"][filename]
        require(comparison["provenance"]["local_dense_build"]["file_sha256"][filename] == expected
                and manifest["files"][f"dense/{filename}"]["sha256"] == expected,
                "Hashes densos declarados incompatibles")
    source = (HERE / "informe_tecnico.md").read_text(encoding="utf-8")
    report_tables = tables(source)
    corpus_table = report_tables[0]
    expected_counts = [eda["counts"]["documents"], eda["counts"]["global_queries"],
                       eda["counts"]["test_queries"], eda["relevance_test"]["judgment_pairs"],
                       eda["relevance_test"]["unique_relevant_documents"]]
    require([row[1] for row in corpus_table[1:]] == [f"{v:,}".replace(",", " ") for v in expected_counts],
            "Conteos de la tabla del corpus distintos")
    expected_global = [[NAMES[r["model"]], {"false": "No", "true": "Sí", "na": "No aplica"}[r["stem"]],
                        *[number(r[m]) for m in METRICS]] for r in means]
    require(next(t for t in report_tables if t[0][0] == "Modelo")[1:] == expected_global,
            "La tabla global del informe no coincide con las fuentes")
    group_rows = next(t for t in report_tables if t[0][0] == "Longitud de consulta")[1:]
    for row, name in zip(group_rows, ("length_0_3", "length_4_7", "length_8_plus"), strict=True):
        group = comparison["groups"][name]
        expected = [number(next(m for m in group["means"] if (m["model"], m["stem"]) == c)
                           ["ap_at_1000"]) for c in (("bm25", "true"), ("dense", "na"))]
        require(row[1:] == [str(group["n"]), *expected], "Tabla de grupos distinta")
    metric_tables = [t for t in report_tables if t[0][0] == "Configuración"]
    rank_tables = [t for t in report_tables if t[0][0].startswith("Documento relevante")]
    for i, name in enumerate(("dense_improves", "dense_worsens")):
        case = comparison["cases"][name]
        require(case["query"] in source, "Consulta original no conservada")
        for row, config in zip(metric_tables[i][1:], ("dense_na", "bm25_true"), strict=True):
            require(row[1:] == [number(case["rankings"][config]["metrics"][m]) for m in METRICS],
                    "Tabla de métricas del caso distinta")
        for row in rank_tables[i][1:]:
            for index, config in enumerate(("dense_na", "bm25_true"), 1):
                doc = next(d for d in case["rankings"][config]["documents"] if d["doc_id"] == row[0])
                require(row[index] == (str(doc["rank"]) if doc["rank"] else "No recuperado en top-1000"),
                        "Tabla de posiciones distinta")
    require(source.count("<!-- pagebreak -->") == 4, "Paginación editorial distinta")
    selected_eda = {
        "lexical_total": eda["documents"]["lexical"]["length"]["total"],
        "lexical_mean": eda["documents"]["lexical"]["length"]["mean"],
        "lexical_median": eda["documents"]["lexical"]["length"]["median"],
        "lexical_p99": eda["documents"]["lexical"]["length"]["percentiles"]["99"],
        "query_mean": eda["queries_test"]["lexical"]["length"]["mean"],
        "vocabulary_lexical": eda["documents"]["lexical"]["vocabulary_size"],
        "vocabulary_porter": eda["documents"]["porter"]["vocabulary_size"],
        "vocabulary_reduction_pct": reduction,
        "empty_texts": eda["empty_fields"]["text"]["empty_exact"],
        "relevance_mean": eda["relevance_test"]["mean"],
        "queries_le2_relevant": eda["relevance_test"]["queries_le2"],
    }
    for key in ("lexical_total", "vocabulary_lexical", "vocabulary_porter"):
        require(f"{selected_eda[key]:,}".replace(",", " ") in source, f"Falta cifra EDA: {key}")
    for key in ("lexical_mean", "query_mean", "vocabulary_reduction_pct"):
        require(number(selected_eda[key], 2) in source, f"Cifra EDA distinta: {key}")
    require(number(selected_eda["relevance_mean"], 3) in source, "Media de relevantes distinta")
    return {
        "base_commit": BASE, "date": "2026-11-19", "tolerance_absolute": TOL,
        "inputs": hashes, "counts": metrics["counts"], "overall": means,
        "verified_group_mean_rows": 49, "verified_paired_metric_sets": 27,
        "case_metrics_recomputed_from_positions": case_checks,
        "eda_selected_values": selected_eda,
        "artifact_hashes_compared_to_evaluation": True,
        "dataset_revisions": metrics["provenance"]["dataset"],
        "deployment_revision": manifest["revision"],
        "source_sha256": digest((HERE / "informe_tecnico.md").read_bytes()),
        "scope": "Auditoría de reportes y texto; no reevalúa test ni verifica los artefactos completos.",
        "discrepancies": [
            ("El aporte 12_comparacion_modelos.md conserva la advertencia antigua de hashes densos "
            "distintos. Los campos actuales artifact_bytes_equal_to_original son true y los hashes "
            "locales de #12 coinciden con #11 y deploy/artifacts.json. Se sigue la evidencia estructurada."),
            ("La copia Windows de los JSON usa CRLF. Los hashes de procedencia se contrastan con los "
            "blobs Git fijados y con igualdad semántica de los JSON; no se modifican los originales. "
            "El CSV conserva LF y satisface su hash exacto."),
        ],
        "pending": ["Revisión de esta versión por el equipo",
                    "Comparar (#14)", "Arranque tras reposo", "Pico de memoria del servidor bajo uso"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "verificacion_fuentes.json")
    parser.add_argument("--pdf", action="store_true", help="Contrasta las cinco páginas con el Markdown")
    args = parser.parse_args()
    result = verify()
    if args.pdf:
        result["pdf"] = verify_pdf()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Fuentes verificadas: 4536 filas, 7 configuraciones, 49 medias de grupo y 4 rankings de casos")


if __name__ == "__main__":
    main()
