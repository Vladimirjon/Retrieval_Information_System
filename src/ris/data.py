"""Carga reproducible del corpus FiQA, sus consultas y sus qrels."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from pyarrow import parquet

_FIQA_REPO = "BeIR/fiqa"
_FIQA_REVISION = "979c07a7cb5ccc6ca009792241fa1250b98055dd"
_QRELS_REPO = "BeIR/fiqa-qrels"
_QRELS_REVISION = "252958f2d646e22cab6d0c72dd3f0d5de6d0655a"

_QRELS_FILES = {
    "train": "train.tsv",
    "validation": "dev.tsv",
    "dev": "dev.tsv",
    "test": "test.tsv",
}


@dataclass(frozen=True, slots=True)
class Doc:
    """Documento FiQA con el título y el texto originales."""

    title: str
    text: str


def _download_file(repo_id: str, filename: str, revision: str) -> Path:
    """Descarga un archivo de una revisión fijada y usa la caché de HF Hub."""

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
        )
    )


def _read_parquet_rows(
    repo_id: str,
    filename: str,
    revision: str,
    columns: list[str],
) -> list[dict[str, object]]:
    path = _download_file(repo_id, filename, revision)
    schema_columns = set(parquet.read_schema(path).names)
    missing_columns = set(columns) - schema_columns
    if missing_columns:
        raise ValueError(
            f"El Parquet {filename} no contiene las columnas requeridas: "
            f"{sorted(missing_columns)}"
        )
    table = parquet.read_table(path, columns=columns)
    return table.to_pylist()


def _string_id(value: object, field: str, row_number: int) -> str:
    if value is None:
        raise ValueError(f"ID nulo en {field}, fila {row_number}")
    return str(value)


def _text(value: object, field: str, row_number: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} no es texto en la fila {row_number}")
    return value


def load_corpus() -> dict[str, Doc]:
    """Carga todos los documentos FiQA, incluidos los textos vacíos."""

    rows = _read_parquet_rows(
        _FIQA_REPO,
        "corpus/corpus-00000-of-00001.parquet",
        _FIQA_REVISION,
        ["_id", "title", "text"],
    )
    corpus: dict[str, Doc] = {}
    for row_number, row in enumerate(rows, start=1):
        doc_id = _string_id(row["_id"], "_id", row_number)
        if doc_id in corpus:
            raise ValueError(f"ID de documento duplicado: {doc_id}")
        corpus[doc_id] = Doc(
            title=_text(row["title"], "title", row_number),
            text=_text(row["text"], "text", row_number),
        )
    return corpus


def _split_filename(split: str) -> str:
    try:
        return _QRELS_FILES[split]
    except KeyError as error:
        supported = "train, validation (dev), test"
        raise ValueError(f"split debe ser uno de: {supported}") from error


def load_qrels(split: str = "test") -> dict[str, dict[str, int]]:
    """Carga los qrels de un split y normaliza los dos tipos de ID a str."""

    filename = _split_filename(split)
    path = _download_file(_QRELS_REPO, filename, _QRELS_REVISION)
    qrels: dict[str, dict[str, int]] = {}
    required_columns = {"query-id", "corpus-id", "score"}

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"El archivo {filename} no tiene el esquema qrels esperado")

        for row_number, row in enumerate(reader, start=2):
            try:
                query_id = str(int(row["query-id"]))
                doc_id = str(int(row["corpus-id"]))
                score = int(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Qrel inválido en {filename}, fila {row_number}"
                ) from error

            if split == "test" and score != 1:
                raise ValueError(
                    f"Score distinto de 1 en {filename}, fila {row_number}: {score}"
                )

            relevant_docs = qrels.setdefault(query_id, {})
            if doc_id in relevant_docs:
                raise ValueError(
                    f"Par query/documento duplicado en {filename}, "
                    f"fila {row_number}: ({query_id}, {doc_id})"
                )
            relevant_docs[doc_id] = score

    return qrels


def load_queries(split: str = "test") -> dict[str, str]:
    """Carga solo las consultas mencionadas por los qrels del split."""

    query_ids = set(load_qrels(split))
    rows = _read_parquet_rows(
        _FIQA_REPO,
        "queries/queries-00000-of-00001.parquet",
        _FIQA_REVISION,
        ["_id", "text"],
    )

    queries: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=1):
        query_id = _string_id(row["_id"], "_id", row_number)
        if query_id in queries:
            raise ValueError(f"ID de consulta duplicado: {query_id}")
        if query_id in query_ids:
            queries[query_id] = _text(row["text"], "text", row_number)

    missing_ids = query_ids - queries.keys()
    if missing_ids:
        sample = sorted(missing_ids)[:5]
        raise ValueError(f"Los qrels referencian consultas ausentes: {sample}")
    return queries
