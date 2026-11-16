"""Reproduce #12 sin sobrescribir las métricas originales de #11."""

import argparse
import json
import math
import subprocess
from importlib.metadata import version
from pathlib import Path

import faiss

from ris import comparison
from ris.data import load_corpus, load_qrels, load_queries
from ris.evaluation import average_precision, precision_at_k, recall_at_k
from ris.index import InvertedIndex
from ris.retrievers import BM25Retriever, DenseRetriever
from scripts.evaluate import (
    corpus_sha256,
    dataset_provenance,
    dump_rankings,
    nltk_provenance,
    package_versions,
    verify_index_sample,
)


def run(root: Path, artifacts_dir: Path) -> dict:
    """Carga fuentes fijadas, compara agregados y recupera solo los dos casos."""
    queries, qrels = load_queries("test"), load_qrels("test")
    if len(queries) != 648 or set(queries) != set(qrels):
        raise ValueError("Se esperaban exactamente las 648 consultas test de FiQA")
    rows, original = comparison.validate_reports(root / "reports", set(qrels))
    sources = dataset_provenance()
    nltk = nltk_provenance()
    if sources != original["provenance"]["dataset"]:
        raise ValueError("Revisiones/hashes de FiQA distintos de #11")
    if nltk != original["provenance"]["nltk_resources"]:
        raise ValueError("Recursos o versión NLTK distintos de #11")
    corpus = load_corpus()
    relevant = {qid: {doc for doc, score in docs.items() if score > 0}
                for qid, docs in qrels.items()}
    observed = {
        "n_queries": len(queries), "n_corpus_docs": len(corpus),
        "n_qrel_pairs": sum(map(len, qrels.values())),
        "n_relevant_pairs": sum(map(len, relevant.values())),
        "n_relevant_docs_distinct": len(set.union(*relevant.values())),
        "n_queries_without_relevant": sum(not docs for docs in relevant.values()),
        "n_relevant_docs_not_in_corpus": len(set.union(*relevant.values()) - corpus.keys()),
        "n_rows": len(rows),
    }
    if observed != original["counts"]:
        raise ValueError("Conteos de fuentes reales distintos de #11")
    from ris.retrievers.dense import _json_hash

    if _json_hash(sorted(queries)) != original["provenance"]["query_ids_sha256"]:
        raise ValueError("Hash de qids distinto de #11")
    print("Entradas verificadas: 4 536 filas, 648 qids × 7, hashes y medias (atol=1e-12).",
          flush=True)
    groups = comparison.query_groups(queries)
    qids = sorted(queries)
    selected = comparison.select_cases(rows)
    # BM25/Porter se fija por convención, sin elegir el comparador por su métrica test.
    index_path = artifacts_dir / "index_stem.json"
    index = InvertedIndex.load(index_path, expected_stem=True)
    if index.N != len(corpus) or index.doc_lengths.keys() != corpus.keys():
        raise ValueError("Índice BM25 de otro corpus")
    verify_index_sample(index, corpus, "BM25/Porter")
    bm25 = BM25Retriever(index, stem=True)
    dense = DenseRetriever.load(artifacts_dir / "dense",
                                expected_corpus_sha256=corpus_sha256(corpus))
    if dense.metadata["model_revision"] != original["config"]["hyperparameters"][
        "dense"
    ]["model_revision"]:
        raise ValueError("Revisión del modelo distinta de #11")
    faiss.omp_set_num_threads(dense.metadata["threads"])
    cases = {}
    lookup = {(r["model"], r["stem"], r["qid"]): r for r in rows}
    for kind, selection in selected.items():
        if selection is None:
            cases[kind] = None
            continue
        qid = selection["qid"]
        rankings = {"bm25_true": bm25.search(queries[qid], k=1000),
                    "dense_na": dense.search(queries[qid], k=1000)}
        expected = {"bm25_true": lookup["bm25", "true", qid],
                    "dense_na": lookup["dense", "na", qid]}
        case = comparison.case_evidence(qid, queries[qid], qrels[qid], corpus,
                                        rankings, expected)
        for label, ranking in rankings.items():
            ids = [r.doc_id for r in ranking]
            calculated = (precision_at_k(ids, relevant[qid], 10),
                          recall_at_k(ids, relevant[qid], 10),
                          average_precision(ids, relevant[qid], 1000))
            for metric, value in zip(comparison.METRICS, calculated, strict=True):
                if not math.isclose(value, case["rankings"][label]["metrics"][metric],
                                    rel_tol=0, abs_tol=comparison.TOLERANCE):
                    raise ValueError("Referencia por posiciones difiere de ris.evaluation")
            # Evidencia completa local, excluida de Git. No modifica reports/ de #11.
            dump_rankings(artifacts_dir / "analysis" / f"case_{qid}_{label}.jsonl.gz",
                          {qid: ranking}, [qid])
        cases[kind] = {**selection, **case}
        print(f"Caso {kind}: qid={qid}, ΔAP={selection['delta_ap_at_1000']:.9f}", flush=True)
    files = ("index_no_stem.json", "index_stem.json", "dense/metadata.json")
    artifact_hashes = {name: comparison.sha256(artifacts_dir / name) for name in files}
    source_files = ["src/ris/comparison.py", "scripts/analyze_models.py",
                    "src/ris/data.py", "src/ris/preprocessing.py", "src/ris/index.py",
                    "scripts/evaluate.py", *[
                        p.relative_to(root).as_posix()
                        for p in sorted((root / "src/ris/retrievers").glob("*.py"))
                    ]]
    core_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", "src/ris/retrievers",
         "src/ris/preprocessing.py", "src/ris/data.py", "scripts/evaluate.py"],
        cwd=root, text=True,
    ).strip()
    return {
        "schema_version": 1, "counts": observed,
        "methodology": {
            "baseline": list(comparison.BASELINE), "paired_delta": "target - reference",
            "tie_absolute_tolerance": comparison.TOLERANCE,
            "percentiles": "lineales (numpy.quantile)",
            "case_selection": "extremos de AP@1000 denso - BM25/Porter; empate exacto por qid ascendente como cadena",
            "groups": comparison.GROUP_RULES, "randomness": "ninguna; sin semillas",
            "relevance": "score > 0; no juzgado se distingue de juzgado no relevante",
            "inference": "comparación descriptiva test; sin ajuste ni pruebas de significancia",
        },
        "overall": comparison.aggregate(rows, qids),
        "porter": [comparison.paired_differences(rows, (model, "true"),
                    (model, "false"), qids) for model in ("jaccard", "tfidf", "bm25")],
        "versus_bm25_porter": [comparison.paired_differences(rows, config,
                               comparison.BASELINE, qids)
                               for config in comparison.CONFIGURATIONS
                               if config != comparison.BASELINE],
        "groups": {name: {"n": len(ids), "qids": ids,
                           "means": comparison.aggregate(rows, ids)}
                   for name, ids in groups.items()},
        "cases": cases,
        "provenance": {
            "base_commit": "51222038f2f6ff764e0715b7194799fe2df54913",
            "retrieval_evaluation_code_commit": core_commit,
            "source_sha256": {name: comparison.sha256(root / name) for name in source_files},
            "metrics_per_query_sha256": comparison.sha256(root / "reports/metrics_per_query.csv"),
            "metrics_summary_sha256": comparison.sha256(root / "reports/metrics_summary.json"),
            "dataset": sources, "nltk_resources": nltk,
            "original_artifacts": original["provenance"]["artifacts_dir_files"],
            "local_artifacts_sha256": artifact_hashes,
            "artifact_bytes_equal_to_original": {name: artifact_hashes[name]
                == original["provenance"]["artifacts_dir_files"][name]["sha256"] for name in files},
            "local_dense_build": {key: dense.metadata[key] for key in (
                "model_revision", "corpus_sha256", "file_sha256", "batch_size",
                "threads", "library_versions", "max_seq_length")},
            "evaluation_config": original["config"],
            "evaluation_versions": original["versions"],
            "analysis_versions": {**package_versions(), **{
                name: version(name)
                for name in ("pandas", "matplotlib", "transformers")}},
            "index_check": "500 documentos de BM25/Porter: muestra, no garantía completa",
            "case_metrics_validation": "4 rankings: posiciones/Fraction, ris.evaluation y CSV; atol=1e-12",
        },
    }


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2,
                                 sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("reports/comparison_summary.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    write_summary(run(root, args.artifacts_dir.resolve()), args.output)
