"""Estadística descriptiva para #4; no construye índices ni evalúa modelos."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

from ris.data import Doc
from ris.preprocessing import preprocess

STAGES = ("original", "normalized", "lexical", "porter")
LABELS = {
    "original": "Original (NLTK, con puntuación y mayúsculas)",
    "normalized": "Minúsculas + alfanuméricos (diagnóstico)",
    "lexical": "Sin stopwords, sin Porter",
    "porter": "Sin stopwords, con Porter",
}


def percentage(part: int, total: int) -> float:
    """Porcentaje con denominador explícito; universo vacío aporta cero."""
    return 100.0 * part / total if total else 0.0


def describe(values: Iterable[int]) -> dict:
    """Población completa: desviación ddof=0 y percentiles lineales de NumPy."""
    array = np.asarray(list(values), dtype=np.int64)
    if np.any(array < 0):
        raise ValueError("Los conteos no pueden ser negativos")
    counts = Counter(int(value) for value in array)
    quantiles = (1, 5, 25, 50, 75, 90, 95, 99)
    return {
        "count": int(array.size),
        "total": int(array.sum()),
        "min": int(array.min()) if array.size else None,
        "max": int(array.max()) if array.size else None,
        "mean": float(array.mean()) if array.size else None,
        "median": float(np.median(array)) if array.size else None,
        "std_population": float(array.std(ddof=0)) if array.size else None,
        "percentiles": {
            str(q): float(np.percentile(array, q, method="linear"))
            if array.size
            else None
            for q in quantiles
        },
        "distribution": {str(value): counts[value] for value in sorted(counts)},
        "zeros": counts[0],
    }


@dataclass
class Profile:
    """Solo agregados y longitudes; nunca conserva los tokens de la colección."""

    cf: Counter = field(default_factory=Counter)
    df: Counter = field(default_factory=Counter)
    lengths: list[int] = field(default_factory=list)

    def add(self, tokens: list[str]) -> None:
        local_tf = Counter(tokens)  # Bag of Words de un documento o consulta.
        self.cf.update(local_tf)
        self.df.update(local_tf.keys())  # Una unidad como máximo por documento.
        self.lengths.append(len(tokens))

    def summarize(self, ids: list[str]) -> dict:
        lengths = describe(self.lengths)
        vocabulary = len(self.cf)
        ordered = sorted(zip(ids, self.lengths), key=lambda pair: (pair[1], pair[0]))
        top = sorted(self.cf, key=lambda term: (-self.cf[term], term))[:20]
        most_widespread = sorted(self.df, key=lambda term: (-self.df[term], term))[:20]
        result = {
            "length": lengths,
            "vocabulary_size": vocabulary,
            "collection_frequency_total": self.cf.total(),
            "hapax_cf1": sum(cf == 1 for cf in self.cf.values()),
            "rare_cf_le5": sum(cf <= 5 for cf in self.cf.values()),
            "single_document_df1": sum(df == 1 for df in self.df.values()),
            "top_by_cf": [{"term": t, "cf": self.cf[t], "df": self.df[t]} for t in top],
            "top_by_df": [
                {"term": t, "cf": self.cf[t], "df": self.df[t]} for t in most_widespread
            ],
            "cf_distribution": {
                str(cf): count
                for cf, count in sorted(Counter(self.cf.values()).items())
            },
            "shortest": [{"id": i, "tokens": n} for i, n in ordered[:5]],
            "longest": [
                {"id": i, "tokens": n}
                for i, n in sorted(ordered, key=lambda pair: (-pair[1], pair[0]))[:5]
            ],
            "length_le5": sum(n <= 5 for n in self.lengths),
        }
        for name in ("hapax_cf1", "rare_cf_le5", "single_document_df1"):
            result[f"{name}_pct_types"] = percentage(result[name], vocabulary)
        if lengths["total"] != self.cf.total():
            raise ValueError("La suma de TF no coincide con los tokens")
        if any(not 1 <= df <= len(ids) or df > self.cf[t] for t, df in self.df.items()):
            raise ValueError("DF debe estar entre 1 y N, y no superar CF")
        return result


def profile_texts(
    texts: Iterable[tuple[str, str]],
) -> tuple[list[str], dict[str, Profile]]:
    """Compara bases descriptivas con las dos salidas del preprocesador aprobado.

    `normalized` solo aísla el efecto de las stopwords; no es una nueva política
    de recuperación. Las representaciones clásicas llaman siempre a preprocess.
    """
    profiles = {stage: Profile() for stage in STAGES}
    ids = []
    for identifier, text in texts:
        ids.append(identifier)
        profiles["original"].add(word_tokenize(text))
        profiles["normalized"].add(
            [token for token in word_tokenize(text.lower()) if token.isalnum()]
        )
        plain = preprocess(text, stem=False)
        profiles["lexical"].add(plain)
        stemmed = preprocess(text, stem=True)
        profiles["porter"].add(stemmed)
        if len(plain) != len(stemmed):
            raise ValueError("Porter debe conservar el número de tokens")
    return ids, profiles


def compare_stages(stages: dict) -> dict:
    def reduction(before: str, after: str, key: str) -> float:
        a = stages[before][key]
        b = stages[after][key]
        return percentage(a - b, a)

    return {
        "tokens_original_to_lexical_pct": reduction(
            "original", "lexical", "collection_frequency_total"
        ),
        "tokens_stopwords_only_pct": reduction(
            "normalized", "lexical", "collection_frequency_total"
        ),
        "vocabulary_original_to_lexical_pct": reduction(
            "original", "lexical", "vocabulary_size"
        ),
        "vocabulary_stopwords_only_pct": reduction(
            "normalized", "lexical", "vocabulary_size"
        ),
        "vocabulary_porter_pct": reduction("lexical", "porter", "vocabulary_size"),
    }


def analyze(corpus: dict[str, Doc], queries: dict[str, str], qrels: dict) -> dict:
    """Analiza el corpus completo y las consultas test, validando las referencias."""
    if any(not isinstance(i, str) for i in [*corpus, *queries, *qrels]):
        raise ValueError("Todos los identificadores deben ser str")
    if set(queries) != set(qrels):
        raise ValueError("Las consultas deben coincidir con los IDs de qrels")
    for judgments in qrels.values():
        if any(not isinstance(i, str) or i not in corpus for i in judgments):
            raise ValueError(
                "Los qrels referencian documentos ausentes o IDs inválidos"
            )
        if any(not isinstance(score, int) for score in judgments.values()):
            raise ValueError("Los scores deben ser enteros")

    doc_ids, documents = profile_texts(
        (i, f"{doc.title} {doc.text}".strip()) for i, doc in sorted(corpus.items())
    )
    query_ids, query_profiles = profile_texts(sorted(queries.items()))
    doc_summary = {s: documents[s].summarize(doc_ids) for s in STAGES}
    query_summary = {s: query_profiles[s].summarize(query_ids) for s in STAGES}
    for stage in STAGES:
        unknown = query_profiles[stage].cf.keys() - documents[stage].cf.keys()
        query_summary[stage]["oov_types"] = len(unknown)
        query_summary[stage]["oov_tokens"] = sum(
            query_profiles[stage].cf[t] for t in unknown
        )
        query_summary[stage]["oov_tokens_pct"] = percentage(
            query_summary[stage]["oov_tokens"], query_profiles[stage].cf.total()
        )

    relevant_counts = {
        q: sum(score > 0 for score in qrels[q].values()) for q in query_ids
    }
    relevant_ids = {
        i for judgments in qrels.values() for i, score in judgments.items() if score > 0
    }
    relevance = describe(relevant_counts.values())
    r = list(relevant_counts.values())
    relevance.update(
        {
            "judgment_pairs": sum(len(judgments) for judgments in qrels.values()),
            "score_distribution": {
                str(score): n
                for score, n in sorted(
                    Counter(
                        score
                        for judgments in qrels.values()
                        for score in judgments.values()
                    ).items()
                )
            },
            "unique_relevant_documents": len(relevant_ids),
            "queries_le2": sum(n <= 2 for n in r),
            "queries_gt10": sum(n > 10 for n in r),
            "fewest": sorted(relevant_counts, key=lambda q: (relevant_counts[q], q))[
                :5
            ],
            "most": sorted(relevant_counts, key=lambda q: (-relevant_counts[q], q))[:5],
            "example_counts": {
                q: relevant_counts[q]
                for q in sorted(
                    set(
                        sorted(relevant_counts, key=lambda q: (relevant_counts[q], q))[
                            :5
                        ]
                        + sorted(
                            relevant_counts, key=lambda q: (-relevant_counts[q], q)
                        )[:5]
                    )
                )
            },
            "macro_oracle_ceiling_p10": float(np.mean([min(n, 10) / 10 for n in r]))
            if r
            else 0.0,
            "macro_oracle_ceiling_r10": float(
                np.mean([min(n, 10) / n if n else 0 for n in r])
            )
            if r
            else 0.0,
            "macro_oracle_ceiling_ap1000": float(
                np.mean([min(n, 1000) / n if n else 0 for n in r])
            )
            if r
            else 0.0,
            "relevant_documents_empty_original": sum(
                not f"{corpus[i].title} {corpus[i].text}".strip() for i in relevant_ids
            ),
            "empty_relevant_examples": [
                {
                    "doc_id": i,
                    "qids": sorted(
                        q for q, judgments in qrels.items() if judgments.get(i, 0) > 0
                    ),
                }
                for i in sorted(relevant_ids)
                if not f"{corpus[i].title} {corpus[i].text}".strip()
            ][:5],
            "relevant_documents_empty_lexical": len(
                relevant_ids
                & {
                    i
                    for i, length in zip(doc_ids, documents["lexical"].lengths)
                    if length == 0
                }
            ),
        }
    )
    empty_fields = {}
    for name in ("title", "text"):
        exact = sum(getattr(doc, name) == "" for doc in corpus.values())
        blank = sum(not getattr(doc, name).strip() for doc in corpus.values())
        empty_fields[name] = {
            "empty_exact": exact,
            "blank": blank,
            "empty_exact_pct_documents": percentage(exact, len(corpus)),
            "blank_pct_documents": percentage(blank, len(corpus)),
        }
    return {
        "schema_version": 1,
        "counts": {"documents": len(corpus), "test_queries": len(queries)},
        "integrity": {
            "all_ids_str": True,
            "query_qrel_ids_equal": True,
            "missing_doc_ids": 0,
        },
        "empty_fields": empty_fields,
        "documents": doc_summary,
        "queries_test": query_summary,
        "reductions_documents": compare_stages(doc_summary),
        "reductions_queries": compare_stages(query_summary),
        "relevance_test": relevance,
    }


def provenance(root: Path) -> dict:
    """Origen real del cargador; hashes de archivos fijados y de los módulos usados."""
    from ris import data, preprocessing

    punkt = nltk.data.find("tokenizers/punkt_tab/english/")
    punkt_hashes = {}
    for name in (
        "abbrev_types.txt",
        "collocations.tab",
        "ortho_context.tab",
        "sent_starters.txt",
    ):
        with punkt.join(name).open() as stream:
            punkt_hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    sources = []
    # Constantes privadas solo como metadatos de #2; no se sustituye su cargador.
    for repo, revision, files in (
        (
            data._FIQA_REPO,
            data._FIQA_REVISION,
            (
                "corpus/corpus-00000-of-00001.parquet",
                "queries/queries-00000-of-00001.parquet",
            ),
        ),
        (data._QRELS_REPO, data._QRELS_REVISION, ("train.tsv", "dev.tsv", "test.tsv")),
    ):
        for filename in files:
            path = data._download_file(repo, filename, revision)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            sources.append(
                {
                    "repo": repo,
                    "revision": revision,
                    "file": filename,
                    "sha256": digest,
                }
            )
    return {
        "dataset": "FiQA (BEIR)",
        "split_analyzed": "test",
        "sources": sources,
        "python": platform.python_version(),
        "versions": {
            p: version(p)
            for p in (
                "numpy",
                "nltk",
                "pandas",
                "matplotlib",
                "huggingface_hub",
                "pyarrow",
            )
        },
        "module_sha256": {
            Path(module.__file__).relative_to(root).as_posix(): hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest()
            for module in (data, preprocessing)
        },
        "stopwords_sha256": hashlib.sha256(
            "\n".join(sorted(stopwords.words("english"))).encode("utf-8")
        ).hexdigest(),
        "stopwords_count": len(stopwords.words("english")),
        "punkt_english_sha256": punkt_hashes,
        "policy": {
            "document": "(title + ' ' + text).strip(); se conservan vacíos",
            "original": "nltk.word_tokenize sobre texto original; incluye puntuación y caso",
            "normalized": "word_tokenize(text.lower()); isalnum; sin eliminar stopwords",
            "lexical": "ris.preprocessing.preprocess(text, stem=False)",
            "porter": "ris.preprocessing.preprocess(text, stem=True)",
            "statistics": "población completa, ddof=0, percentiles lineales; sin muestreo",
            "ordering": "frecuencia descendente, desempate término o ID ascendente",
        },
    }


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def report_excerpt(summary: dict) -> str:
    """Texto breve para #17, generado con los mismos agregados del notebook."""

    def count(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    d = summary["documents"]["lexical"]
    q = summary["queries_test"]["lexical"]
    r = summary["relevance_test"]
    reductions = summary["reductions_documents"]
    return (
        "# Descripción del corpus utilizado\n\n"
        f"FiQA, en su versión BEIR en inglés y de dominio financiero, contiene "
        f"{count(summary['counts']['documents'])} documentos y "
        f"{count(summary['counts']['global_queries'])} consultas globales. "
        f"El split test comprende {count(summary['counts']['test_queries'])} consultas "
        f"distintas y {count(r['judgment_pairs'])} pares juzgados, todos con score 1. "
        f"Los {count(summary['empty_fields']['title']['empty_exact'])} títulos están vacíos; "
        f"se conservan los {summary['empty_fields']['text']['empty_exact']} textos vacíos.\n\n"
        f"Tras el pipeline inglés acordado (NLTK, minúsculas, tokens alfanuméricos "
        f"y stopwords), los documentos tienen media {d['length']['mean']:.2f}, "
        f"mediana {d['length']['median']:.1f} y percentil 99 "
        f"{d['length']['percentiles']['99']:.2f} tokens. Las consultas test tienen "
        f"media {q['length']['mean']:.2f} tokens. El vocabulario documental contiene "
        f"{count(d['vocabulary_size'])} términos; Porter lo reduce a "
        f"{count(summary['documents']['porter']['vocabulary_size'])} "
        f"({reductions['vocabulary_porter_pct']:.2f}% respecto de la versión sin Porter), "
        "sin alterar la cantidad de apariciones. Esta reducción no demuestra una "
        "mejora de recuperación.\n\n"
        f"Hay {count(r['unique_relevant_documents'])} documentos relevantes test distintos. "
        f"Por consulta se observan entre {r['min']} y {r['max']} relevantes "
        f"(media {r['mean']:.3f}, mediana {r['median']:.1f}); "
        f"{r['queries_le2']} consultas tienen como máximo dos. "
        f"El techo macro teórico de P@10 según estos qrels es "
        f"{r['macro_oracle_ceiling_p10']:.4f}; no es el resultado de un recuperador. "
        "Los juicios disponibles no establecen la irrelevancia de todos los "
        f"documentos sin juzgar. Hay {r['relevant_documents_empty_original']} "
        "documento relevante sin texto, un caso que debe inspeccionarse al evaluar "
        "la evidencia textual disponible. La efectividad se medirá posteriormente mediante "
        "P@10, R@10 y MAP@1000, sin ajustar hiperparámetros con test.\n\n"
        "Fuentes: [corpus y consultas](https://huggingface.co/datasets/BeIR/fiqa), "
        "[qrels](https://huggingface.co/datasets/BeIR/fiqa-qrels), "
        "[revisiones y hashes](../DATASET.md), "
        "[notebook ejecutado](../../notebooks/04_eda_fiqa.ipynb) y "
        "[agregados reproducibles](../../reports/eda_summary.json). "
        "Fundamento conceptual: Iván Carrera, *Fundamentos de Recuperación de "
        "Información* (pp. 20–33), *Modelo vectorial* (pp. 5–15, 24) y "
        "*Modelo BM25* (pp. 4–8, 14), material docente EPN.\n"
    )
