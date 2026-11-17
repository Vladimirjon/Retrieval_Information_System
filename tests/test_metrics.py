"""Reportes sintéticos completos y pruebas de la vista sin motor ni red (#15)."""

import csv
import hashlib
import io
import json
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from streamlit.dataframe_util import convert_arrow_bytes_to_pandas_df
from streamlit.testing.v1 import AppTest

from app import metrics_data as data
from app.views import metrics
from scripts import evaluate


@pytest.fixture
def reports(tmp_path):
    """Dos qids de texto, uno con cero en todos los modelos; medias a mano."""
    summary = json.loads(Path("reports/metrics_summary.json").read_text("utf-8"))
    summary["counts"].update(n_queries=2, n_rows=14, n_queries_without_relevant=1)
    summary["provenance"]["query_ids_sha256"] = hashlib.sha256(b'["02","10"]').hexdigest()
    rows = []
    for index, (model, stem) in enumerate(data.CONFIGURATIONS, start=1):
        value = index / 10
        rows.extend([
            {"model": model, "stem": stem, "qid": "02", "k": 10,
             "precision": 0., "recall": 0., "ap_at_1000": 0.},
            {"model": model, "stem": stem, "qid": "10", "k": 10,
             "precision": value / 2, "recall": value, "ap_at_1000": value},
        ])
        summary["results"][index - 1].update(
            n_queries=2, n_empty_rankings=0,
            mean_precision_at_k=value / 4, mean_recall_at_k=value / 2, map_at_1000=value / 2,
        )
    write_reports(tmp_path, rows, summary)
    return tmp_path, rows, summary


def write_reports(path, rows, summary):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=evaluate.CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode("utf-8")
    summary["provenance"]["metrics_per_query_sha256"] = hashlib.sha256(raw).hexdigest()
    (path / evaluate.CSV_NAME).write_bytes(raw)
    write_summary(path, summary)


def write_summary(path, summary):
    (path / evaluate.SUMMARY_NAME).write_text(json.dumps(summary), encoding="utf-8")


def view_app(directory, *, query="", search_k=50):
    """Un motor que falla si se usa permite verificar el límite de la vista."""
    return AppTest.from_string(f'''
from pathlib import Path
from app.context import AppContext
from app.views.metrics import render
class ForbiddenEngine:
    def __getattribute__(self, name):
        raise AssertionError("La vista de métricas no debe usar el motor")
render(AppContext(engine=ForbiddenEngine(), query={query!r}, model="dense", k={search_k},
                  stem=True, reports_dir=Path({str(directory)!r})))
''').run(timeout=30)


def test_complete_report_uses_contract_verifier_and_preserves_string_ids(reports, monkeypatch):
    directory, _, _ = reports
    calls = []

    def verify(path):
        calls.append(path)
        evaluate.verify_reports(path)

    monkeypatch.setattr(data, "verify_reports", verify)
    report = data.load_reports(directory)
    assert calls == [directory]
    assert report.qids == ("02", "10")
    assert len(report.rows) == 14
    global_table = metrics.metric_table(report)
    assert global_table["ap_at_1000"].tolist() == pytest.approx([.05, .1, .15, .2, .25, .3, .35])
    assert global_table["precision"].tolist() == pytest.approx([.025, .05, .075, .1, .125, .15, .175])
    dense = global_table[global_table.Modelo == "Embeddings"]
    assert len(dense) == 1 and dense.iloc[0]["Porter"] == "No aplica"
    spec = metrics.chart_spec(global_table, "ap_at_1000", "MAP@1000")
    assert spec["encoding"]["x"]["scale"]["domain"] == [0, 1]
    assert [row["Valor"] for row in spec["data"]["values"]] == global_table.ap_at_1000.tolist()
    assert len({row["Configuración"] for row in spec["data"]["values"]}) == 7


@pytest.mark.parametrize("filename", [evaluate.CSV_NAME, evaluate.SUMMARY_NAME])
def test_missing_files_and_unreadable_files_are_report_errors(reports, monkeypatch, filename):
    directory, _, _ = reports
    original = Path.read_bytes

    def unavailable(path):
        if path.name == filename:
            raise PermissionError("private technical path")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", unavailable)
        with pytest.raises(data.ReportError, match="permisos"):
            data.load_reports(directory)
    (directory / filename).unlink()
    with pytest.raises(data.ReportError, match=filename):
        data.load_reports(directory)


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b'{}', b'\xff',
                                    b'{"schema_version":1,"schema_version":1}',
                                    b'[' * 2000 + b']' * 2000])
def test_malformed_or_incomplete_json(reports, payload):
    directory, _, _ = reports
    (directory / evaluate.SUMMARY_NAME).write_bytes(payload)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)


@pytest.mark.parametrize("payload", [
    b'wrong,header\n', b'model,stem,qid,k,precision,recall,ap_at_1000\n"unclosed',
    b'model,stem,qid,k,precision,recall,ap_at_1000\njaccard,false,02,10,0,0\n',
    b'model,stem,qid,k,precision,recall,ap_at_1000\njaccard,false,02,10,0,0,0,extra\n',
    b'\xff',
])
def test_malformed_csv_even_with_matching_hash(reports, payload):
    directory, _, summary = reports
    summary["counts"]["n_rows"] = len(payload.splitlines()) - 1
    summary["provenance"]["metrics_per_query_sha256"] = hashlib.sha256(payload).hexdigest()
    (directory / evaluate.CSV_NAME).write_bytes(payload)
    write_summary(directory, summary)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), 2), (("schema_version",), True),
    (("config", "k"), "10"), (("config", "k"), True), (("config", "k"), 0),
    (("config", "k"), 1001), (("config", "depth"), 999),
    (("config", "ap_cutoff"), 10), (("config", "split"), "train"),
    (("config", "models"), ["dense"]), (("config", "stem_values", "dense"), ["false", "true"]),
    (("config", "metrics"), {}), (("config", "preprocessing"), []),
    (("config", "hyperparameters", "dense", "dimension"), True),
    (("counts", "n_rows"), 13), (("counts", "n_queries"), 3),
    (("counts", "n_queries"), True), (("counts", "n_relevant_pairs"), -1),
    (("counts", "n_queries_without_relevant"), 3),
    (("provenance", "metrics_per_query_sha256"), "0" * 64),
    (("provenance", "query_ids_sha256"), "0" * 64),
    (("provenance", "dataset"), []), (("provenance", "artifacts_dir_files"), {}),
    (("versions", "python"), None), (("results",), []),
])
def test_incompatible_summary_fields(reports, path, value):
    directory, _, summary = reports
    target = summary
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    write_summary(directory, summary)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "qid", "blank_qid", "stem", "model", "k"])
def test_duplicate_missing_or_invalid_rows_with_updated_hash(reports, mutation):
    directory, rows, summary = reports
    if mutation == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif mutation == "missing":
        rows.pop()
    elif mutation == "qid":
        rows[-1]["qid"] = "11"
    else:
        field, value = {"blank_qid": ("qid", ""), "stem": ("stem", "true"),
                        "model": ("model", "unknown"), "k": ("k", 5)}[mutation]
        rows[-1][field] = value
    summary["counts"]["n_rows"] = len(rows)
    write_reports(directory, rows, summary)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -.01, 1.01, "oops"])
@pytest.mark.parametrize("location", ["csv", "json"])
def test_metrics_must_be_finite_numbers_in_unit_interval(reports, value, location):
    directory, rows, summary = reports
    if location == "csv":
        rows[0]["precision"] = value
    else:
        summary["results"][0]["map_at_1000"] = value
    write_reports(directory, rows, summary)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)


@pytest.mark.parametrize("mutation", ["mean", "count", "count_bool", "empty", "duplicate", "missing"])
def test_global_results_cannot_hide_inconsistent_means_or_counts(reports, mutation):
    directory, _, summary = reports
    if mutation == "mean":
        summary["results"][0]["map_at_1000"] += 1e-9
    elif mutation == "count":
        summary["results"][0]["n_queries"] = 1
    elif mutation == "count_bool":
        summary["results"][0]["n_queries"] = True
    elif mutation == "empty":
        summary["results"][0]["n_empty_rankings"] = 2
    elif mutation == "duplicate":
        summary["results"][-1] = deepcopy(summary["results"][0])
    else:
        summary["results"].pop()
    write_summary(directory, summary)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)


def test_queries_without_relevant_must_be_zero_in_every_configuration(reports):
    directory, rows, summary = reports
    # Un cero en cada modelo no implica que sea el mismo qid sin relevantes.
    # Intercambiar las métricas conserva medias, filas y cantidad de ceros.
    for metric in data.METRICS:
        rows[0][metric], rows[1][metric] = rows[1][metric], rows[0][metric]
    write_reports(directory, rows, summary)
    with pytest.raises(data.ReportError, match="sin relevantes"):
        data.load_reports(directory)
    # Cero no significa ausencia de relevantes: sin ese conteo falso es válido.
    summary["counts"]["n_queries_without_relevant"] = 0
    write_summary(directory, summary)
    assert len(data.load_reports(directory).rows) == 14


def test_mean_tolerance_is_absolute_and_keeps_all_zero_queries(reports):
    directory, _, summary = reports
    summary["results"][0]["map_at_1000"] += 5e-13
    write_summary(directory, summary)
    assert len(data.load_reports(directory).rows) == 14
    summary["results"][0]["map_at_1000"] += 2e-12
    write_summary(directory, summary)
    with pytest.raises(data.ReportError, match="media"):
        data.load_reports(directory)


def test_changed_reports_are_revalidated_and_recover_without_cache(reports):
    directory, rows, summary = reports
    original_csv = (directory / evaluate.CSV_NAME).read_bytes()
    original_json = (directory / evaluate.SUMMARY_NAME).read_bytes()
    assert data.load_reports(directory).qids == ("02", "10")
    # Misma longitud y mtime: también debe detectarse sin basarse en stat().
    stat = (directory / evaluate.CSV_NAME).stat()
    (directory / evaluate.CSV_NAME).write_bytes(original_csv.replace(b"02", b"03"))
    os.utime(directory / evaluate.CSV_NAME, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(data.ReportError):
        data.load_reports(directory)
    (directory / evaluate.CSV_NAME).write_bytes(original_csv)
    summary["schema_version"] = 99
    write_summary(directory, summary)
    with pytest.raises(data.ReportError):
        data.load_reports(directory)
    (directory / evaluate.SUMMARY_NAME).write_bytes(original_json)
    assert len(data.load_reports(directory).rows) == len(rows)


def test_change_during_read_is_rejected(reports, monkeypatch):
    directory, _, _ = reports

    def change_after_verification(path):
        evaluate.verify_reports(path)
        (path / evaluate.SUMMARY_NAME).write_bytes(b"{}")

    monkeypatch.setattr(data, "verify_reports", change_after_verification)
    with pytest.raises(data.ReportError, match="cambiaron"):
        data.load_reports(directory)


def test_streamlit_shows_global_map_and_query_ap_without_query_or_engine(reports, caplog):
    directory, _, _ = reports
    app = view_app(directory)
    assert not app.exception and not app.error
    assert [m.value for m in app.metric] == ["2", "57 638", "7"]
    assert app.selectbox[0].options == ["02", "10"]
    assert app.selectbox[0].value == "02"
    before = app.dataframe[0].value.copy()
    assert app.dataframe[1].value.ap_at_1000.tolist() == [0.] * 7
    global_columns = json.loads(app.dataframe[0].proto.columns)
    query_columns = json.loads(app.dataframe[1].proto.columns)
    assert global_columns["ap_at_1000"]["label"] == "MAP@1000"
    assert query_columns["ap_at_1000"]["label"] == "AP@1000"
    app.selectbox[0].select("10").run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.dataframe[0].value, before)
    assert app.dataframe[1].value.ap_at_1000.tolist() == pytest.approx([.1, .2, .3, .4, .5, .6, .7])
    app.radio[0].set_value("recall").run()
    assert not app.exception and app.radio[0].value == "recall"
    charts = app.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert spec["encoding"]["x"]["title"] == "R@10"
    chart_data = convert_arrow_bytes_to_pandas_df(charts[0].proto.data.data)
    assert chart_data.Valor.tolist() == before.recall.tolist()
    assert "top-n" in app.caption[0].value
    assert not [record for record in caplog.records
                if record.name == "streamlit.elements.vega_charts" and record.levelno >= 30]


def test_other_positive_k_is_labeled_from_reports_not_search_controls(reports):
    directory, rows, summary = reports
    summary["config"]["k"] = 5
    for row in rows:
        row["k"] = 5
    write_reports(directory, rows, summary)
    report = data.load_reports(directory)
    assert metrics.metric_labels(report) == {
        "precision": "P@5", "recall": "R@5", "ap_at_1000": "MAP@1000",
    }
    app = view_app(directory, query="free search is not a test qid", search_k=20)
    assert not app.exception and not app.error
    columns = json.loads(app.dataframe[0].proto.columns)
    assert columns["precision"]["label"] == "P@5"
    assert columns["recall"]["label"] == "R@5"
    assert app.selectbox[0].value == "02"
    assert app.dataframe[0].value.iloc[-1].ap_at_1000 == .35


def test_streamlit_removes_stale_results_and_restores_them_after_error(reports):
    directory, _, summary = reports
    app = view_app(directory)
    assert len(app.dataframe) == 2
    csv_path = directory / evaluate.CSV_NAME
    csv_bytes = csv_path.read_bytes()
    csv_path.unlink()
    app.button[0].click().run()
    assert not app.exception and len(app.error) == 1
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert "Restaure juntos" in app.info[0].value
    csv_path.write_bytes(csv_bytes)
    summary["schema_version"] = 2
    write_summary(directory, summary)
    app.button[0].click().run()
    assert not app.exception and app.error and not app.dataframe
    summary["schema_version"] = 1
    write_summary(directory, summary)
    app.button[0].click().run()
    assert not app.exception and not app.error and len(app.dataframe) == 2


def test_versioned_reports_are_displayed_exactly():
    report = data.load_reports(Path("reports"))
    table = metrics.metric_table(report)
    assert len(report.qids) == report.summary["counts"]["n_queries"]
    for row, entry in zip(table.to_dict("records"), report.summary["results"], strict=True):
        assert row["precision"] == entry["mean_precision_at_k"]
        assert row["recall"] == entry["mean_recall_at_k"]
        assert row["ap_at_1000"] == entry["map_at_1000"]
    app = view_app(Path("reports").resolve())
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.dataframe[0].value, table)
    for metric, label in metrics.metric_labels(report).items():
        app.radio[0].set_value(metric).run()
        chart = app.get("vega_lite_chart")[0].proto
        assert json.loads(chart.spec)["encoding"]["x"]["title"] == label
        plotted = convert_arrow_bytes_to_pandas_df(chart.data.data)
        assert plotted.Valor.tolist() == table[metric].tolist()
    qid = report.qids[-1]
    app.selectbox[0].select(qid).run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.dataframe[1].value, metrics.metric_table(report, qid))
    pd.testing.assert_frame_equal(app.dataframe[0].value, table)
