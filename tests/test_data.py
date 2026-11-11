import pyarrow as arrow
import pytest
from pyarrow import parquet

from ris import data


@pytest.fixture
def fiqa_files(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.parquet"
    parquet.write_table(
        arrow.table(
            {
                "_id": ["10", "20", "30"],
                "title": ["", "", "tercer título"],
                "text": ["texto diez", "", "texto treinta"],
            }
        ),
        corpus_path,
    )
    queries_path = tmp_path / "queries.parquet"
    parquet.write_table(
        arrow.table(
            {
                "_id": ["1", "2", "3"],
                "text": ["consulta uno", "consulta dos", "consulta tres"],
            }
        ),
        queries_path,
    )

    qrels_paths = {}
    for split, rows in {
        "train.tsv": "query-id\tcorpus-id\tscore\n1\t10\t1\n",
        "dev.tsv": "query-id\tcorpus-id\tscore\n3\t30\t1\n",
        "test.tsv": "query-id\tcorpus-id\tscore\n2\t20\t1\n2\t30\t1\n",
    }.items():
        qrels_path = tmp_path / split
        qrels_path.write_text(rows, encoding="utf-8", newline="\n")
        qrels_paths[split] = qrels_path

    files = {
        ("BeIR/fiqa", "corpus/corpus-00000-of-00001.parquet"): corpus_path,
        ("BeIR/fiqa", "queries/queries-00000-of-00001.parquet"): queries_path,
        **{
            ("BeIR/fiqa-qrels", filename): path
            for filename, path in qrels_paths.items()
        },
    }

    calls = []

    def download_file(*, repo_id, filename, repo_type, revision):
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "repo_type": repo_type,
                "revision": revision,
            }
        )
        return files[(repo_id, filename)]

    monkeypatch.setattr(data, "hf_hub_download", download_file)
    return {"files": files, "calls": calls}


def test_load_corpus_preserves_empty_fields_and_string_ids(fiqa_files):
    del fiqa_files

    corpus = data.load_corpus()

    assert len(corpus) == 3
    assert corpus["10"] == data.Doc(title="", text="texto diez")
    assert corpus["20"] == data.Doc(title="", text="")


def test_load_qrels_normalizes_ids_and_maps_validation_to_dev(fiqa_files):
    del fiqa_files

    assert data.load_qrels() == {"2": {"20": 1, "30": 1}}
    assert data.load_qrels("validation") == {"3": {"30": 1}}
    assert data.load_qrels("dev") == {"3": {"30": 1}}


def test_load_queries_filters_by_qrels_and_ids_match_corpus(fiqa_files):
    del fiqa_files

    corpus = data.load_corpus()
    qrels = data.load_qrels()
    queries = data.load_queries()

    assert queries == {"2": "consulta dos"}
    assert set(queries) == set(qrels)
    assert all(doc_id in corpus for relevant in qrels.values() for doc_id in relevant)
    assert data.load_queries("validation") == {"3": "consulta tres"}


def test_load_qrels_test_requires_binary_score_one(fiqa_files):
    test_path = fiqa_files["files"][("BeIR/fiqa-qrels", "test.tsv")]
    test_path.write_text(
        "query-id\tcorpus-id\tscore\n2\t20\t0\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(ValueError, match="Score distinto de 1"):
        data.load_qrels("test")


def test_load_qrels_rejects_unknown_split(fiqa_files):
    del fiqa_files

    with pytest.raises(ValueError, match=r"train, validation \(dev\), test"):
        data.load_qrels("unknown")


def test_loaders_use_frozen_hub_revisions(fiqa_files):
    data.load_corpus()
    data.load_queries()
    data.load_qrels("train")
    data.load_qrels("validation")

    calls = fiqa_files["calls"]
    source_files = {
        (call["repo_id"], call["filename"], call["revision"], call["repo_type"])
        for call in calls
    }
    assert source_files == {
        (
            "BeIR/fiqa",
            "corpus/corpus-00000-of-00001.parquet",
            "979c07a7cb5ccc6ca009792241fa1250b98055dd",
            "dataset",
        ),
        (
            "BeIR/fiqa",
            "queries/queries-00000-of-00001.parquet",
            "979c07a7cb5ccc6ca009792241fa1250b98055dd",
            "dataset",
        ),
        (
            "BeIR/fiqa-qrels",
            "test.tsv",
            "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
            "dataset",
        ),
        (
            "BeIR/fiqa-qrels",
            "train.tsv",
            "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
            "dataset",
        ),
        (
            "BeIR/fiqa-qrels",
            "dev.tsv",
            "252958f2d646e22cab6d0c72dd3f0d5de6d0655a",
            "dataset",
        ),
    }
