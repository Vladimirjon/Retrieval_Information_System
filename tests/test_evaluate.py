import csv
import getpass
import gzip
import json
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from ris.data import Doc
from ris.index import InvertedIndex
from ris.retrievers import DenseRetriever, Result
from ris.retrievers.dense import DIMENSION
from scripts import evaluate

REPO_ROOT = Path(__file__).resolve().parents[1]


class Fixed:
    """Retriever de doble: devuelve rankings fijos por texto de consulta."""

    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, query, k=10):
        return self.rankings.get(query, [])[:k]


def results(*doc_ids):
    # Scores descendentes e irrelevantes para las métricas (solo cuenta el orden).
    return [Result(doc_id, 1 / position) for position, doc_id in enumerate(doc_ids, 1)]


# ---------------------------------------------------------------- lógica pura


def test_relevant_sets_keep_only_positive_scores_with_str_ids():
    qrels = {"10": {"d2": 1, "d1": 0}, "2": {"d3": 2}}
    relevant = evaluate.relevant_sets(qrels)
    assert relevant == {"10": {"d2"}, "2": {"d3"}}
    assert list(relevant) == ["10", "2"]  # orden ascendente como cadena
    assert all(isinstance(qid, str) for qid in relevant)


def test_select_queries_orders_by_qid_and_ignores_queries_without_qrels():
    selected = evaluate.select_queries(
        {"2": "b", "10": "a", "99": "sin qrels"}, {"2": {"x"}, "10": {"y"}}
    )
    assert list(selected) == ["10", "2"]
    with pytest.raises(ValueError, match="ausentes"):
        evaluate.select_queries({"1": "a"}, {"1": {"x"}, "7": {"y"}})


@pytest.mark.parametrize(("k", "depth"), [
    (0, 1000), (-1, 1000), (10, 0), (1001, 1000), (10, 999), (True, 1000), (2.5, 1000),
])
def test_invalid_cutoffs_are_rejected(k, depth):
    with pytest.raises(ValueError):
        evaluate.validate_cutoffs(k, depth)


def test_valid_cutoffs_are_accepted():
    evaluate.validate_cutoffs(10, 1000)
    evaluate.validate_cutoffs(1500, 2000)


def test_hand_calculated_metrics_with_cutoffs_and_missing_rankings():
    k, depth = 2, 1001
    relevant = {
        "q1": {"a", "c"},   # relevantes en las posiciones 1 y 3
        "q2": {"r1"},       # relevante en la posición 500: más allá de k, dentro de 1000
        "q3": {"r2"},       # relevante en la posición 1001: fuera del corte de AP
        "q4": {"z"},        # el retriever no devuelve nada
    }
    filler = [f"f{i}" for i in range(1000)]
    queries = {"q1": "t1", "q2": "t2", "q3": "t3", "q4": "t4"}
    retriever = Fixed({
        "t1": results("a", "b", "c", "d"),
        "t2": results(*filler[:499], "r1", *filler[499:600]),
        "t3": results(*filler, "r2"),
        "t4": [],
    })
    rows, entry = evaluate.evaluate_configuration(
        "bm25", "true", retriever, queries, relevant, k=k, depth=depth, out=lambda _: None
    )
    by_qid = {row["qid"]: row for row in rows}
    # q1: P@2 = 1/2 (solo "a"); R@2 = 1/2 de 2 relevantes;
    #     AP = (1/1 + 2/3) / 2 = 5/6 (c está en la posición 3).
    assert by_qid["q1"]["precision"] == pytest.approx(0.5)
    assert by_qid["q1"]["recall"] == pytest.approx(0.5)
    assert by_qid["q1"]["ap_at_1000"] == pytest.approx(5 / 6)
    # q2: nada en el top-2; AP = (1/500) / 1 porque r1 sigue dentro de las 1000 primeras.
    assert by_qid["q2"]["precision"] == 0.0
    assert by_qid["q2"]["recall"] == 0.0
    assert by_qid["q2"]["ap_at_1000"] == pytest.approx(1 / 500)
    # q3: r2 en la posición 1001 queda fuera de AP@1000 (aunque depth=1001 lo recuperó).
    assert by_qid["q3"]["ap_at_1000"] == 0.0
    # q4: sin ranking aporta 0 y cuenta en el denominador.
    assert by_qid["q4"] == {
        "qid": "q4", "precision": 0.0, "recall": 0.0, "ap_at_1000": 0.0,
        "model": "bm25", "stem": "true", "k": 2,
    }
    assert entry["n_queries"] == 4
    assert entry["n_empty_rankings"] == 1
    assert entry["mean_precision_at_k"] == pytest.approx(0.5 / 4)
    assert entry["mean_recall_at_k"] == pytest.approx(0.5 / 4)
    # MAP@1000 = (5/6 + 1/500 + 0 + 0) / 4
    assert entry["map_at_1000"] == pytest.approx((5 / 6 + 1 / 500) / 4)


def test_dense_style_configuration_uses_stem_na():
    rows, entry = evaluate.evaluate_configuration(
        "dense", "na", Fixed({"t": results("x")}), {"q": "t"}, {"q": {"x"}},
        k=10, depth=1000, out=lambda _: None,
    )
    # P@10 divide por k aunque el ranking tenga 1 documento: 1/10.
    assert rows[0]["stem"] == "na"
    assert rows[0]["precision"] == pytest.approx(0.1)
    assert rows[0]["recall"] == 1.0
    assert entry["stem"] == "na"


def test_retriever_exceeding_depth_is_rejected():
    class Greedy:
        def search(self, query, k=10):
            return results("a", "b", "c")

    with pytest.raises(ValueError, match="depth"):
        evaluate.retrieve_rankings(Greedy(), {"q": "t"}, 2)


def test_rankings_sha256_depends_on_order_scores_and_queries():
    base = {"1": results("a", "b"), "2": results("c")}
    digest = evaluate.rankings_sha256(base, ["1", "2"])
    assert digest == evaluate.rankings_sha256(dict(base), ["1", "2"])
    swapped = {"1": results("b", "a"), "2": results("c")}
    assert digest != evaluate.rankings_sha256(swapped, ["1", "2"])
    rescored = {"1": [Result("a", 1.0), Result("b", 0.5000000001)], "2": results("c")}
    assert digest != evaluate.rankings_sha256(rescored, ["1", "2"])
    assert digest != evaluate.rankings_sha256(base, ["1"])


def test_csv_schema_row_order_and_float_repr():
    rows = [
        {"model": "dense", "stem": "na", "qid": "1", "k": 10,
         "precision": 0.1, "recall": 1 / 3, "ap_at_1000": 0.0},
        {"model": "bm25", "stem": "true", "qid": "2", "k": 10,
         "precision": 0.0, "recall": 0.0, "ap_at_1000": 0.0},
        {"model": "bm25", "stem": "false", "qid": "10", "k": 10,
         "precision": 0.2, "recall": 0.5, "ap_at_1000": 0.25},
        {"model": "bm25", "stem": "false", "qid": "2", "k": 10,
         "precision": 0.0, "recall": 0.0, "ap_at_1000": 0.0},
        {"model": "jaccard", "stem": "true", "qid": "1", "k": 10,
         "precision": 0.0, "recall": 0.0, "ap_at_1000": 0.0},
    ]
    text = evaluate.render_csv(rows)
    assert "\r" not in text
    lines = text.splitlines()
    assert lines[0] == "model,stem,qid,k,precision,recall,ap_at_1000"
    assert [line.split(",")[:3] for line in lines[1:]] == [
        ["jaccard", "true", "1"], ["bm25", "false", "10"], ["bm25", "false", "2"],
        ["bm25", "true", "2"], ["dense", "na", "1"],
    ]
    assert lines[2] == "bm25,false,10,10,0.2,0.5,0.25"
    assert lines[5] == f"dense,na,1,10,0.1,{1 / 3!r},0.0"


def test_corpus_sha256_matches_the_dense_artifact_formula():
    corpus = {"b": Doc("t", "x"), "a": Doc("", "y")}
    retriever = DenseRetriever.build(corpus, encoder=HashEncoder(), threads=1)
    assert evaluate.corpus_sha256(corpus) == retriever.metadata["corpus_sha256"]


def test_write_atomically_leaves_old_files_if_a_write_fails(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.json"
    first.write_text("viejo", encoding="utf-8")
    with pytest.raises(UnicodeEncodeError):
        evaluate.write_atomically({first: "nuevo", second: "\ud800"})
    assert first.read_text(encoding="utf-8") == "viejo"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["a.csv"]


# ------------------------------------------------------------- integración


class HashEncoder:
    """Encoder denso falso y determinista: el vector depende solo del texto."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            seed = int(sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vectors.append(np.random.default_rng(seed).normal(size=DIMENSION))
        return np.array(vectors, dtype=np.float32)


CORPUS = {
    "d1": Doc("", "gato perro"),
    "d2": Doc("gato", "gato raton"),
    "d3": Doc("", "perro pez loro casa"),
    "d4": Doc("", ""),
    "d5": Doc("", "banco prestamo"),
}
QUERIES = {"1": "gato perro", "10": "banco", "2": "casa", "3": "zzz", "99": "sin qrels"}
QRELS = {
    "1": {"d1": 1, "d3": 1},
    "10": {"d5": 1},
    "2": {"d1": 1, "d4": 0},
    "3": {"d2": 1},
}
QIDS = ["1", "10", "2", "3"]
MODEL_STEMS = [
    ("jaccard", "false"), ("jaccard", "true"), ("tfidf", "false"), ("tfidf", "true"),
    ("bm25", "false"), ("bm25", "true"), ("dense", "na"),
]


@pytest.fixture
def environment(monkeypatch, tmp_path):
    """Artefactos mínimos reales (índices y denso con encoder falso) y datos sintéticos."""

    def preprocess(text, *, stem=False):
        return text.lower().split()

    for module in ("ris.index", "ris.retrievers.jaccard", "ris.retrievers.tfidf",
                   "ris.retrievers.bm25"):
        monkeypatch.setattr(f"{module}.preprocess", preprocess)
    monkeypatch.setattr(evaluate, "load_corpus", lambda: CORPUS)
    monkeypatch.setattr(evaluate, "load_queries", lambda split: QUERIES)
    monkeypatch.setattr(evaluate, "load_qrels", lambda split: QRELS)
    monkeypatch.setattr(evaluate, "dataset_provenance", lambda: [{"repo": "sintético"}])
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    InvertedIndex.build(CORPUS, stem=False).save(artifacts / "index_no_stem.json")
    InvertedIndex.build(CORPUS, stem=True).save(artifacts / "index_stem.json")
    DenseRetriever.build(CORPUS, encoder=HashEncoder(), batch_size=2, threads=1).save(
        artifacts / "dense"
    )
    return artifacts


def execute(artifacts, output, **kwargs):
    return evaluate.run(
        artifacts_dir=artifacts, output_dir=output, k=2, dense_encoder=HashEncoder(),
        out=lambda _: None, **kwargs,
    )


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_run_writes_documented_schema_and_hand_calculated_jaccard(environment, tmp_path):
    output = tmp_path / "out"
    summary = execute(environment, output)
    text = (output / "metrics_per_query.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "model,stem,qid,k,precision,recall,ap_at_1000"
    rows = read_rows(output / "metrics_per_query.csv")
    # Una fila por (modelo, stem, qid), con el orden documentado (qid como cadena).
    assert [(r["model"], r["stem"], r["qid"]) for r in rows] == [
        (model, stem, qid) for (model, stem), qid in product(MODEL_STEMS, QIDS)
    ]
    assert {r["k"] for r in rows} == {"2"}
    assert {r["stem"] for r in rows if r["model"] == "dense"} == {"na"}

    jaccard = {r["qid"]: r for r in rows if (r["model"], r["stem"]) == ("jaccard", "false")}
    # "gato perro": d1=2/2, d2=1/3, d3=1/5 -> [d1, d2, d3]; relevantes {d1, d3}:
    #   P@2=1/2, R@2=1/2, AP=(1/1 + 2/3)/2 = 5/6.
    assert float(jaccard["1"]["precision"]) == pytest.approx(0.5)
    assert float(jaccard["1"]["recall"]) == pytest.approx(0.5)
    assert float(jaccard["1"]["ap_at_1000"]) == pytest.approx(5 / 6)
    # "banco": solo d5 (relevante): P@2=1/2, R@2=1, AP=1.
    assert float(jaccard["10"]["precision"]) == pytest.approx(0.5)
    assert float(jaccard["10"]["recall"]) == 1.0
    assert float(jaccard["10"]["ap_at_1000"]) == 1.0
    # "casa": solo d3, que no es relevante para esta consulta (relevante: d1; d4 vale 0).
    assert float(jaccard["2"]["ap_at_1000"]) == 0.0
    # "zzz": ningún término conocido, ranking vacío: aporta 0.
    assert (jaccard["3"]["precision"], jaccard["3"]["recall"]) == ("0.0", "0.0")

    entry = next(r for r in summary["results"]
                 if (r["model"], r["stem"]) == ("jaccard", "false"))
    assert entry["n_queries"] == 4
    assert entry["n_empty_rankings"] == 1
    assert entry["mean_precision_at_k"] == pytest.approx((0.5 + 0.5) / 4)
    assert entry["map_at_1000"] == pytest.approx((5 / 6 + 1) / 4)
    assert [(r["model"], r["stem"]) for r in summary["results"]] == MODEL_STEMS
    assert summary["counts"] == {
        "n_queries": 4, "n_queries_without_relevant": 0, "n_qrel_pairs": 6,
        "n_relevant_pairs": 5, "n_relevant_docs_distinct": 4,
        "n_relevant_docs_not_in_corpus": 0, "n_corpus_docs": 5,
    }
    assert summary["schema_version"] == 1
    assert summary["config"]["k"] == 2
    assert summary["config"]["hyperparameters"]["bm25"]["k1"] == 1.5
    assert summary["config"]["hyperparameters"]["bm25"]["b"] == 0.75
    assert json.loads((output / "metrics_summary.json").read_text(encoding="utf-8")) == summary


def test_two_runs_are_byte_identical_with_equal_ranking_hashes(environment, tmp_path):
    first = execute(environment, tmp_path / "run1")
    second = execute(environment, tmp_path / "run2")
    for name in ("metrics_per_query.csv", "metrics_summary.json"):
        assert (tmp_path / "run1" / name).read_bytes() == (tmp_path / "run2" / name).read_bytes()
    assert [r["rankings_sha256"] for r in first["results"]] == [
        r["rankings_sha256"] for r in second["results"]
    ]
    assert len({r["rankings_sha256"] for r in first["results"]}) > 1


def test_rerun_overwrites_outputs_in_place(environment, tmp_path):
    output = tmp_path / "out"
    execute(environment, output)
    before = (output / "metrics_summary.json").read_bytes()
    execute(environment, output)
    assert (output / "metrics_summary.json").read_bytes() == before
    assert sorted(path.name for path in output.iterdir()) == [
        "metrics_per_query.csv", "metrics_summary.json",
    ]


def test_summary_has_no_absolute_paths_or_username(environment, tmp_path):
    execute(environment, tmp_path / "out")
    text = (tmp_path / "out" / "metrics_summary.json").read_text(encoding="utf-8")
    forbidden = [str(tmp_path), tmp_path.as_posix(), str(REPO_ROOT), REPO_ROOT.as_posix(),
                 "C:\\", "C:/", "/Users/", "\\Users", "\\\\"]
    user = getpass.getuser()
    if len(user) >= 4:
        forbidden.append(user)
    for fragment in forbidden:
        assert fragment not in text, fragment
    provenance = json.loads(text)["provenance"]
    assert set(provenance["artifacts_dir_files"]) == {
        "index_no_stem.json", "index_stem.json", "dense/metadata.json",
    }
    assert provenance["artifacts_dir_files"]["dense/metadata.json"]["corpus_sha256"] == (
        evaluate.corpus_sha256(CORPUS)
    )


def test_rankings_dir_inside_reports_is_rejected_before_any_work(
    environment, tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluate, "load_corpus", lambda: pytest.fail("No cargar"))
    output = tmp_path / "reports"
    for inside in (output, output / "rankings", Path("reports") / "rankings"):
        with pytest.raises(ValueError, match="reports"):
            execute(environment, output, rankings_dir=inside)
    assert not output.exists()


def test_rankings_dir_outside_reports_writes_reproducible_gzip(environment, tmp_path):
    execute(environment, tmp_path / "reports", rankings_dir=tmp_path / "rk1")
    execute(environment, tmp_path / "reports", rankings_dir=tmp_path / "rk2")
    names = sorted(path.name for path in (tmp_path / "rk1").iterdir())
    assert names == [
        "rankings_bm25_no_stem.jsonl.gz", "rankings_bm25_stem.jsonl.gz",
        "rankings_dense.jsonl.gz", "rankings_jaccard_no_stem.jsonl.gz",
        "rankings_jaccard_stem.jsonl.gz", "rankings_tfidf_no_stem.jsonl.gz",
        "rankings_tfidf_stem.jsonl.gz",
    ]
    for name in names:
        assert (tmp_path / "rk1" / name).read_bytes() == (tmp_path / "rk2" / name).read_bytes()
    with gzip.open(tmp_path / "rk1" / "rankings_jaccard_no_stem.jsonl.gz", "rt",
                   encoding="utf-8") as file:
        lines = [json.loads(line) for line in file]
    assert [line["qid"] for line in lines] == QIDS
    assert [doc for doc, _ in lines[0]["ranking"]] == ["d1", "d2", "d3"]
    assert sorted(path.name for path in (tmp_path / "reports").iterdir()) == [
        "metrics_per_query.csv", "metrics_summary.json",
    ]


def test_missing_artifacts_explain_how_to_build_them(environment, tmp_path):
    (environment / "index_stem.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"index_stem\.json.*build_artifacts\.py"):
        execute(environment, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_dense_artifact_from_another_corpus_is_rejected(environment, tmp_path, monkeypatch):
    other = {**CORPUS, "d6": Doc("", "otro documento")}
    monkeypatch.setattr(evaluate, "load_corpus", lambda: other)
    with pytest.raises(ValueError, match="otro corpus"):
        execute(environment, tmp_path / "out")


def test_index_with_wrong_stem_configuration_is_rejected(environment, tmp_path):
    (environment / "index_stem.json").write_bytes(
        (environment / "index_no_stem.json").read_bytes()
    )
    with pytest.raises(ValueError, match="stemming"):
        execute(environment, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_stale_index_with_different_documents_is_rejected(environment, tmp_path):
    smaller = {key: CORPUS[key] for key in ("d1", "d2", "d3", "d4")}
    InvertedIndex.build(smaller, stem=False).save(environment / "index_no_stem.json")
    with pytest.raises(ValueError, match="corpus cargado"):
        execute(environment, tmp_path / "out")


@pytest.mark.parametrize("arguments", [
    ["--k", "0"], ["--k", "-3"], ["--depth", "0"], ["--k", "1001"], ["--depth", "999"],
    ["--rankings-dir", "reports/rankings"],
])
def test_cli_rejects_invalid_options_before_loading_data(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["evaluate.py", *arguments])
    monkeypatch.setattr(evaluate, "load_corpus", lambda: pytest.fail("No cargar"))
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
