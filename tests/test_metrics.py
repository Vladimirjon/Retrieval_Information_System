"""Reportes sintéticos completos y pruebas de la vista sin motor ni red (#15)."""

import csv
import hashlib
import importlib.util
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


def view_app(directory, *, query="", search_k=50, with_navigation=False):
    """Un motor que falla si se usa permite verificar el límite de la vista."""
    return AppTest.from_string(f'''
from pathlib import Path
from app.context import AppContext
from app.views.metrics import render
import streamlit as st
class ForbiddenEngine:
    def __getattribute__(self, name):
        raise AssertionError("La vista de métricas no debe usar el motor")
show = not st.toggle("Otra vista", key="test_other_view") if {with_navigation!r} else True
if show:
    render(AppContext(engine=ForbiddenEngine(), query={query!r}, model="dense", k={search_k},
                      stem=True, reports_dir=Path({str(directory)!r})))
''').run(timeout=30)


def data_line(app):
    """Línea única de contexto: cifras de reports/."""
    lines = [text.value for text in app.caption if text.value.startswith("FiQA test:")]
    assert len(lines) == 1
    return lines[0]


def visible_configurations(app):
    return len(app.dataframe[0].value) if app.dataframe else 0


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
    assert len(dense) == 1 and dense.iloc[0]["Porter"] == "No usa"
    spec = metrics.chart_spec(global_table, "ap_at_1000", "MAP@1000")
    assert spec["encoding"]["x"]["scale"]["domain"] == [0, 1]
    assert [row["Valor"] for row in spec["data"]["values"]] == global_table.ap_at_1000.tolist()
    assert len({(row["Método"], row["Variante"]) for row in spec["data"]["values"]}) == 7


def test_theme_defaults_are_used_when_config_is_unavailable(monkeypatch):
    theme_path = metrics.ROOT / ".streamlit/config.toml"
    original_open = Path.open

    def open_without_theme(path, *args, **kwargs):
        if path == theme_path:
            raise FileNotFoundError("missing config")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_without_theme)
    spec = importlib.util.spec_from_file_location("metrics_without_config", metrics.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.THEME == metrics._DEFAULT_THEME


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


def test_verifier_reason_is_preserved(reports, monkeypatch):
    directory, _, _ = reports

    def reject(_path):
        raise ValueError("hash del CSV incorrecto")

    monkeypatch.setattr(data, "verify_reports", reject)
    with pytest.raises(data.ReportError, match="hash del CSV incorrecto"):
        data.load_reports(directory)


def test_rerun_resets_qid_when_new_report_pair_has_different_qids(reports, tmp_path):
    directory, rows, summary = reports
    other = tmp_path / "other"
    other.mkdir()
    other_rows = deepcopy(rows)
    for row in other_rows:
        row["qid"] = "20" if row["qid"] == "02" else "30"
    other_summary = deepcopy(summary)
    other_summary["provenance"]["query_ids_sha256"] = hashlib.sha256(b'["20","30"]').hexdigest()
    write_reports(other, other_rows, other_summary)

    app = view_app(directory)
    assert not app.exception and app.selectbox[0].value == "02"
    for name in (evaluate.CSV_NAME, evaluate.SUMMARY_NAME):
        (directory / name).write_bytes((other / name).read_bytes())
    app.run()
    assert not app.exception
    assert app.selectbox[0].options == ["20", "30"]
    assert app.selectbox[0].value == "20"


def test_streamlit_shows_global_map_and_query_ap_without_query_or_engine(reports, caplog):
    directory, _, _ = reports
    app = view_app(directory)
    assert not app.exception and not app.error
    assert not app.metric and not app.header
    line = data_line(app)
    assert line == "FiQA test: 2 consultas, 57 638 documentos"
    assert app.selectbox[0].options == ["02", "10"]
    assert app.selectbox[0].value == "02"
    before = app.dataframe[0].value.copy()
    assert app.dataframe[1].value.ap_at_1000.tolist() == [0.] * 7
    assert "font-weight" not in app.dataframe[1].proto.arrow_data.styler.styles
    global_columns = json.loads(app.dataframe[0].proto.columns)
    query_columns = json.loads(app.dataframe[1].proto.columns)
    assert global_columns["ap_at_1000"]["label"] == "MAP@1000"
    assert query_columns["ap_at_1000"]["label"] == "AP@1000"
    assert global_columns["ap_at_1000"]["type_config"]["format"] == "%.4f"
    app.selectbox[0].select("10").run()
    assert not app.exception
    assert any(text.value == "**Consulta 10**" for text in app.markdown)
    assert not any("Consulta 02" in text.value for text in app.markdown)
    pd.testing.assert_frame_equal(app.dataframe[0].value, before)
    assert app.dataframe[1].value.ap_at_1000.tolist() == pytest.approx([.1, .2, .3, .4, .5, .6, .7])
    assert "font-weight" in app.dataframe[1].proto.arrow_data.styler.styles
    app.segmented_control(key="metrics_metric").set_value("recall").run()
    assert not app.exception and app.segmented_control(key="metrics_metric").value == "recall"
    charts = app.get("vega_lite_chart")
    assert len(charts) == 1
    spec = json.loads(charts[0].proto.spec)
    assert spec["encoding"]["x"]["title"] == "R@10"
    chart_data = convert_arrow_bytes_to_pandas_df(charts[0].proto.data.data)
    assert chart_data.Valor.tolist() == before.recall.tolist()
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
    assert not any(button.label in {"Volver a leer reportes", "Reintentar carga"} for button in app.button)
    csv_path = directory / evaluate.CSV_NAME
    csv_bytes = csv_path.read_bytes()
    csv_path.unlink()
    app.run()
    assert not app.exception and len(app.error) == 1
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert "Restaure juntos" in app.info[0].value
    csv_path.write_bytes(csv_bytes)
    summary["schema_version"] = 2
    write_summary(directory, summary)
    app.button(key="metrics_retry").click().run()
    assert not app.exception and app.error and not app.dataframe
    summary["schema_version"] = 1
    write_summary(directory, summary)
    app.button(key="metrics_retry").click().run()
    assert not app.exception and not app.error and len(app.dataframe) == 2
    assert not any(button.label in {"Volver a leer reportes", "Reintentar carga"} for button in app.button)


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
        app.segmented_control(key="metrics_metric").set_value(metric).run()
        chart = app.get("vega_lite_chart")[0].proto
        assert json.loads(chart.spec)["encoding"]["x"]["title"] == label
        plotted = convert_arrow_bytes_to_pandas_df(chart.data.data)
        assert plotted.Valor.tolist() == table[metric].tolist()
    qid = report.qids[-1]
    app.selectbox[0].select(qid).run()
    assert not app.exception and not app.error
    pd.testing.assert_frame_equal(app.dataframe[1].value, metrics.metric_table(report, qid))
    pd.testing.assert_frame_equal(app.dataframe[0].value, table)


def test_fragment_reuses_only_the_unchanged_validated_pair(reports, monkeypatch):
    directory, _, _ = reports
    report = data.load_reports(directory)

    def unexpected_rerun():
        raise AssertionError("Los reportes idénticos no requieren un rerun completo")

    monkeypatch.setattr(metrics.st, "rerun", unexpected_rerun)
    metrics._check_report_unchanged(directory, report)


@pytest.mark.parametrize("name", [evaluate.CSV_NAME, evaluate.SUMMARY_NAME])
@pytest.mark.parametrize("change", ["same_size", "missing"])
def test_fragment_requests_full_validation_when_either_file_changes(
    reports, monkeypatch, name, change,
):
    directory, _, _ = reports
    report = data.load_reports(directory)
    path = directory / name
    if change == "same_size":
        raw = path.read_bytes()
        # Un cambio del mismo tamaño también debe invalidar el par.
        path.write_bytes(b"!" + raw[1:])
    else:
        path.unlink()

    class FullRerunRequested(Exception):
        pass

    def request_rerun():
        raise FullRerunRequested

    monkeypatch.setattr(metrics.st, "rerun", request_rerun)
    with pytest.raises(FullRerunRequested):
        metrics._check_report_unchanged(directory, report)


@pytest.mark.parametrize("porter,n_rows", [("Sin Porter", 4), ("Con Porter", 4), ("Comparar", 7)])
def test_porter_control_filters_summary_chart_and_both_tables_without_changing_values(reports, porter, n_rows):
    directory, _, _ = reports
    report = data.load_reports(directory)
    app = view_app(directory)
    app.segmented_control(key="metrics_porter").set_value(porter).run()
    assert not app.exception and not app.error
    assert visible_configurations(app) == n_rows
    for index, qid in enumerate((None, "02")):
        expected = metrics.filter_table(metrics.metric_table(report, qid), porter,
                                        list(metrics.MODEL_NAMES.values()))
        pd.testing.assert_frame_equal(app.dataframe[index].value, expected)
        dense = app.dataframe[index].value.query("Modelo == 'Embeddings'")
        assert len(dense) == 1 and dense.iloc[0].Porter == "No usa"
    plotted = convert_arrow_bytes_to_pandas_df(app.get("vega_lite_chart")[0].proto.data.data)
    assert plotted.Valor.tolist() == app.dataframe[0].value.ap_at_1000.tolist()
    assert len(plotted) == n_rows


def test_method_filters_empty_state_and_reset(reports, monkeypatch, caplog):
    from streamlit.elements.lib import policies

    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    app = view_app(reports[0])
    app.segmented_control(key="metrics_porter").set_value("Con Porter").run()
    app.pills(key="metrics_methods").set_value(["BM25"]).run()
    assert visible_configurations(app) == 1
    assert app.dataframe[0].value.Modelo.tolist() == ["BM25"]
    assert app.dataframe[0].value.Porter.tolist() == ["Con Porter"]
    app.pills(key="metrics_methods").set_value([]).run()
    assert not app.exception and visible_configurations(app) == 0
    assert not app.dataframe and not app.get("vega_lite_chart")
    assert any("No hay métodos" in message.value for message in app.info)
    app.button(key="metrics_reset").click().run()
    assert app.segmented_control(key="metrics_porter").value == "Comparar"
    assert len(app.dataframe[0].value) == 7
    assert not any("Session State API" in record.message for record in caplog.records)


def test_absolute_differences_match_real_values_and_exclude_dense():
    report = data.load_reports(Path("reports"))
    table = metrics.metric_table(report)
    differences = metrics.porter_differences(table, "ap_at_1000")
    assert list(differences.columns) == ["Modelo", "Sin Porter", "Con Porter", "Cambio absoluto"]
    differences = differences.set_index("Modelo")
    assert set(differences.index) == {"Jaccard", "TF-IDF", "BM25"}
    assert differences.loc["Jaccard", "Cambio absoluto"] == pytest.approx(
        0.038461083621982944 - 0.03851954210374813)
    assert differences.loc["Jaccard", "Cambio absoluto"] < 0
    assert differences.loc["BM25", "Cambio absoluto"] == pytest.approx(
        0.20621917443991963 - 0.18099260544155976)
    assert differences.loc["BM25", "Cambio absoluto"] > 0
    query = metrics.metric_table(report, "10034")
    deltas = metrics.porter_differences(query, "recall").set_index("Modelo")
    assert deltas.loc["TF-IDF", "Cambio absoluto"] == .5


def test_missing_variant_has_no_fabricated_zero_or_difference():
    table = pd.DataFrame([{"Modelo": "BM25", "Porter": "Sin Porter", "precision": .1}])
    comparison = metrics.porter_differences(table, "precision")
    assert pd.isna(comparison.iloc[0]["Con Porter"])
    assert pd.isna(comparison.iloc[0]["Cambio absoluto"])


def test_query_explorer_uses_shared_filters_and_literal_search(reports):
    app = view_app(reports[0])
    app.segmented_control(key="metrics_porter").set_value("Sin Porter").run()
    before = app.dataframe[0].value.copy()
    app.checkbox(key="metrics_explore").check().run()
    app.text_input[0].set_value("02").run()
    assert not app.exception and len(app.dataframe[2].value) == 4
    assert any("1 consulta, 4 filas" in text.value for text in app.caption)
    assert set(app.dataframe[2].value.qid) == {"02"}
    assert set(app.dataframe[2].value.Porter) == {"Sin Porter", "No usa"}
    app.text_input[0].set_value(".*").run()
    assert not app.exception and len(app.dataframe) == 2
    assert any("No hay consultas" in message.value for message in app.info)
    pd.testing.assert_frame_equal(app.dataframe[0].value, before)


def test_filters_survive_navigation_and_reset_restores_defaults(reports):
    app = view_app(reports[0], with_navigation=True)
    app.segmented_control(key="metrics_porter").set_value("Con Porter").run()
    app.pills(key="metrics_methods").set_value(["BM25"]).run()
    app.segmented_control(key="metrics_metric").set_value("recall").run()
    app.selectbox[0].select("10").run()
    app.toggle[0].set_value(True).run()
    assert not app.dataframe
    app.toggle[0].set_value(False).run()
    assert not app.exception
    assert app.segmented_control(key="metrics_porter").value == "Con Porter"
    assert app.pills(key="metrics_methods").value == ["BM25"]
    assert app.segmented_control(key="metrics_metric").value == "recall"
    assert app.selectbox[0].value == "10"
    app.button(key="metrics_reset").click().run()
    assert app.segmented_control(key="metrics_porter").value == "Comparar"
    assert app.pills(key="metrics_methods").value == list(metrics.MODEL_NAMES.values())
    assert app.segmented_control(key="metrics_metric").value == "ap_at_1000"
    assert app.selectbox[0].value == "02"


def test_chart_tooltips_name_the_metric_method_and_variant_and_keep_common_scale(reports):
    report = data.load_reports(reports[0])
    colors = dict(zip(metrics.PORTER_NAMES.values(), metrics.VARIANT_COLORS, strict=True))
    for porter in metrics.PORTER_MODES:
        table = metrics.filter_table(metrics.metric_table(report), porter, list(metrics.MODEL_NAMES.values()))
        for metric, label in metrics.metric_labels(report).items():
            spec = metrics.chart_spec(table, metric, label)
            assert spec["encoding"]["x"]["scale"]["domain"] == [0, 1]
            assert spec["encoding"]["x"]["title"] == label
            assert spec["encoding"]["tooltip"] == [
                {"field": "Método", "type": "nominal"}, {"field": "Variante", "type": "nominal"},
                {"field": "Métrica", "type": "nominal"},
                {"field": "Valor", "type": "quantitative", "format": ".6f"},
            ]
            assert all(row["Métrica"] == label for row in spec["data"]["values"])
            scale = spec["encoding"]["color"]["scale"]
            assert scale["domain"] == sorted(set(table.Porter), key=list(colors).index)
            assert scale["range"] == [colors[name] for name in scale["domain"]]


def test_chart_groups_bars_by_method_with_the_three_palette_colors(reports):
    report = data.load_reports(reports[0])
    table = metrics.metric_table(report)
    spec = metrics.chart_spec(table, "ap_at_1000", "MAP@1000")
    assert metrics.VARIANT_COLORS == ("#7A2E5E", "#246477", "#A86A12")
    assert spec["encoding"]["y"]["field"] == "Método" and spec["encoding"]["y"]["sort"] is None
    assert spec["encoding"]["color"]["scale"]["domain"] == ["Sin Porter", "Con Porter", "No usa"]
    assert spec["encoding"]["color"]["scale"]["range"] == list(metrics.VARIANT_COLORS)
    legend = spec["encoding"]["color"]["legend"]
    assert legend["orient"] == "top" and "'Embeddings'" in legend["labelExpr"]
    classical, dense = spec["layer"][:3], spec["layer"][3:]
    assert all(layer["encoding"]["yOffset"]["field"] == "Variante" for layer in classical)
    assert not any("yOffset" in layer["encoding"] for layer in dense)
    assert dense[0]["mark"]["height"] == {"band": .45}
    assert [row["Método"] for row in spec["data"]["values"]].count("Embeddings") == 1
    assert [row["Etiqueta"] for row in spec["data"]["values"]][:2] == ["Sin 0.0500", "Con 0.1000"]
    assert spec["height"] == 4 * 52 + 110


@pytest.mark.parametrize("porter,variants", [("Sin Porter", ["Sin Porter", "No usa"]),
                                             ("Con Porter", ["Con Porter", "No usa"])])
def test_chart_stays_valid_with_a_single_porter_mode(reports, porter, variants):
    report = data.load_reports(reports[0])
    table = metrics.filter_table(metrics.metric_table(report), porter, list(metrics.MODEL_NAMES.values()))
    spec = metrics.chart_spec(table, "recall", "R@10")
    assert spec["encoding"]["color"]["scale"]["domain"] == variants
    assert {row["Variante"] for row in spec["data"]["values"]} == set(variants)
    assert spec["height"] == 4 * 44 + 110
    # Sin pares que agrupar, las barras no reservan una franja vacía.
    assert not any("yOffset" in layer["encoding"] for layer in spec["layer"])
    assert all(layer["mark"].get("height") == {"band": .45} for layer in spec["layer"][::3])
    only_dense = metrics.chart_spec(table[table.Modelo == "Embeddings"], "recall", "R@10")
    assert only_dense["encoding"]["color"]["scale"]["domain"] == ["No usa"]
    assert only_dense["height"] == 44 + 110
    empty = metrics.chart_spec(table.iloc[0:0], "recall", "R@10")
    assert empty["data"]["values"] == [] and empty["encoding"]["color"]["scale"]["domain"] == []


@pytest.mark.parametrize("case", ["empty", "zero", "tie", "close", "single"])
def test_metric_summary_handles_empty_zero_ties_and_full_precision(case):
    values = {"empty": [], "zero": [0., 0.], "tie": [.3, .3], "single": [.1, .3],
              "close": [.3000003, .3000004]}[case]
    table = pd.DataFrame([
        {"Modelo": method, "Porter": porter, "precision": value}
        for method, porter, value in zip(("Jaccard", "BM25"), ("Sin Porter", "Con Porter"), values,
                                         strict=False)
    ], columns=["Modelo", "Porter", "precision"])
    text = metrics.metric_summary(table, "precision")
    if case == "empty":
        assert text == "Sin datos."
    elif case == "zero":
        assert text == "Todos los valores son 0."
    elif case == "tie":
        assert text == "Mejor: Jaccard Sin Porter y BM25 Con Porter 0.3000"
    elif case == "single":
        assert text == "Mejor: BM25 Con Porter 0.3000"
    else:
        assert "BM25" in text and "Jaccard" not in text and " y " not in text


def test_metric_summary_names_embeddings_without_a_variant():
    table = pd.DataFrame([
        {"Modelo": "Embeddings", "Porter": "No usa", "ap_at_1000": 0.311586},
        {"Modelo": "BM25", "Porter": "Con Porter", "ap_at_1000": 0.2},
    ])
    assert metrics.metric_summary(table, "ap_at_1000") == "Mejor: Embeddings 0.3116"


def test_empty_methods_recover_without_resetting_other_filters(reports):
    app = view_app(reports[0])
    app.segmented_control(key="metrics_porter").set_value("Con Porter").run()
    app.segmented_control(key="metrics_metric").set_value("recall").run()
    app.selectbox[0].select("10").run()
    app.pills(key="metrics_methods").set_value([]).run()
    assert not app.dataframe
    app.button(key="metrics_restore_methods").click().run()
    assert not app.exception
    assert app.segmented_control(key="metrics_porter").value == "Con Porter"
    assert app.segmented_control(key="metrics_metric").value == "recall" and app.selectbox[0].value == "10"
    assert app.pills(key="metrics_methods").value == list(metrics.MODEL_NAMES.values())
    assert len(app.dataframe[0].value) == 4


def test_empty_query_filter_clears_without_resetting_selection(reports):
    app = view_app(reports[0])
    app.segmented_control(key="metrics_porter").set_value("Con Porter").run()
    app.pills(key="metrics_methods").set_value(["BM25"]).run()
    app.segmented_control(key="metrics_metric").set_value("recall").run()
    app.selectbox[0].select("10").run()
    before = app.dataframe[0].value.copy()
    app.checkbox(key="metrics_explore").check().run()
    app.text_input[0].set_value("does not exist").run()
    assert len(app.dataframe) == 2
    app.button(key="metrics_clear_query_filter").click().run()
    assert not app.exception and app.text_input[0].value == ""
    assert app.selectbox[0].value == "10" and app.segmented_control(key="metrics_metric").value == "recall"
    assert app.segmented_control(key="metrics_porter").value == "Con Porter"
    assert app.pills(key="metrics_methods").value == ["BM25"]
    assert len(app.dataframe[2].value) == 2
    pd.testing.assert_frame_equal(app.dataframe[0].value, before)


def test_tables_keep_one_context_column_fixed_for_horizontal_scroll(reports):
    app = view_app(reports[0])
    app.checkbox(key="metrics_explore").check().run()
    for element in app.dataframe:
        columns = json.loads(element.proto.columns)
        active_pinned = [name for name in element.value.columns if columns.get(name, {}).get("pinned")]
        assert active_pinned == (["qid"] if "qid" in element.value.columns else ["Modelo"])


def test_global_summary_follows_visible_methods_and_metric_without_changing_data(reports):
    app = view_app(reports[0])
    app.selectbox[0].select("10").run()
    app.segmented_control(key="metrics_porter").set_value("Con Porter").run()
    app.pills(key="metrics_methods").set_value(["BM25"]).run()
    before = app.dataframe[0].value.copy()
    app.segmented_control(key="metrics_metric").set_value("precision").run()
    summaries = [text.value for text in app.caption if text.value.startswith("Mejor:")]
    assert summaries == ["Mejor: BM25 Con Porter 0.1500"]
    pd.testing.assert_frame_equal(app.dataframe[0].value, before)


def test_chart_marks_the_best_value_with_text_and_outline_not_only_color():
    table = pd.DataFrame([
        {"Modelo": "Jaccard", "Porter": "Sin Porter", "precision": .1},
        {"Modelo": "BM25", "Porter": "Con Porter", "precision": .3},
        {"Modelo": "TF-IDF", "Porter": "Sin Porter", "precision": .3},
    ])
    spec = metrics.chart_spec(table, "precision", "P@10")
    values = spec["data"]["values"]
    assert [row["Mejor"] for row in values] == [False, True, True]
    assert [row["Etiqueta"] for row in values] == ["Sin 0.1000", "Con 0.3000", "Sin 0.3000"]
    assert [row["Valor"] for row in values] == table.precision.tolist()
    bar, plain, bold = spec["layer"][:3]
    assert bar["encoding"]["stroke"]["condition"]["test"] == "datum.Mejor"
    assert (plain["mark"]["fontWeight"], bold["mark"]["fontWeight"]) == (400, 700)
    assert "datum.Mejor" in bold["transform"][0]["filter"]
    zero = metrics.chart_spec(table.assign(precision=0.), "precision", "P@10")
    assert not any(row["Mejor"] for row in zero["data"]["values"])
    assert spec["padding"]["right"] == 8
    assert metrics.chart_spec(table.assign(precision=.95), "precision", "P@10")["padding"]["right"] == 56


def visible_texts(app):
    """Todo el texto que el usuario lee en la vista renderizada."""
    texts = [element.value for kind in (app.markdown, app.caption, app.subheader, app.info, app.error)
             for element in kind]
    texts += [element.label for kind in (app.expander, app.button, app.checkbox, app.selectbox,
                                         app.text_input) for element in kind]
    texts += [element.proto.label for element in (*app.get("segmented_control"), *app.get("button_group"))]
    texts += [element.proto.label for element in app.get("download_button")]
    texts += [element.proto.placeholder for element in app.text_input]
    return texts


def strings_in(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in (*value.keys(), *value.values()):
            yield from strings_in(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings_in(item)


def test_visible_copy_has_no_long_dashes_or_dot_separators(reports):
    directory, _, _ = reports
    report = data.load_reports(directory)
    table = metrics.metric_table(report)
    copy = [metrics.data_line(report), *metrics.help_lines(report),
            *metrics.PORTER_NAMES.values(), *metrics.MODEL_NAMES.values(),
            *metrics.metric_labels(report).values(), *metrics.metric_labels(report, per_query=True).values()]
    for metric in metrics.metric_labels(report):
        copy.append(metrics.metric_summary(table, metric))
        copy.extend(strings_in(metrics.chart_spec(table, metric, metrics.metric_labels(report)[metric])))
        copy.extend(strings_in(metrics.porter_differences(table, metric).columns.tolist()))
    copy.extend(strings_in(metrics.chart_spec(table.iloc[:0], "recall", "R@10")))
    app = view_app(directory)
    app.checkbox(key="metrics_explore").check().run()
    copy += visible_texts(app)
    app.text_input[0].set_value("zzz").run()
    copy += visible_texts(app)
    app.pills(key="metrics_methods").set_value([]).run()
    copy += visible_texts(app)
    (directory / evaluate.CSV_NAME).unlink()
    app.run()
    copy += visible_texts(app)
    assert len(copy) > 40
    for text in copy:
        assert "\u2014" not in text and " \u00b7 " not in text and "\u2191" not in text, text
    source = Path(metrics.__file__).read_text("utf-8")
    assert "\u2014" not in source and "\u00b7" not in source and "\u2191" not in source


def test_help_is_short_and_follows_report_config(reports):
    directory, rows, summary = reports
    summary["config"]["k"] = 5
    for row in rows:
        row["k"] = 5
    write_reports(directory, rows, summary)
    lines = metrics.help_lines(data.load_reports(directory))
    assert len(lines) <= 5 and all(len(line) < 110 for line in lines)
    assert any(line.startswith("**P@5**") for line in lines)
    assert any(line.startswith("**MAP@1000**") for line in lines)


def test_large_metric_table_skips_pandas_styler(monkeypatch):
    table = pd.DataFrame({
        "qid": [str(index) for index in range(metrics.LARGE_TABLE_ROW_THRESHOLD)],
        "Modelo": ["BM25"] * metrics.LARGE_TABLE_ROW_THRESHOLD,
        "Porter": ["Sin Porter"] * metrics.LARGE_TABLE_ROW_THRESHOLD,
        "precision": [0.5] * metrics.LARGE_TABLE_ROW_THRESHOLD,
        "recall": [0.5] * metrics.LARGE_TABLE_ROW_THRESHOLD,
        "ap_at_1000": [0.5] * metrics.LARGE_TABLE_ROW_THRESHOLD,
    })
    received = []

    def fail_on_style(_):
        raise AssertionError("Las tablas grandes no deben construir un Styler")

    class FakeColumnConfig:
        TextColumn = staticmethod(lambda *args, **kwargs: None)
        NumberColumn = staticmethod(lambda *args, **kwargs: None)

    monkeypatch.setattr(pd.DataFrame, "style", property(fail_on_style))
    monkeypatch.setattr(metrics.st, "dataframe", lambda data, **kwargs: received.append(data))
    monkeypatch.setattr(metrics.st, "column_config", FakeColumnConfig)
    metrics._show_table(table, {"precision": "P@10", "recall": "R@10",
                               "ap_at_1000": "AP@1000"},
                        description="Detalle", key="large_table")
    assert len(received) == 1 and received[0] is table


def test_view_has_no_page_heading_or_scattered_help_icons(reports):
    app = view_app(reports[0])
    assert not app.header and not app.subheader[0].proto.help
    for element in (*app.get("segmented_control"), *app.get("button_group"), *app.selectbox):
        assert not element.proto.help
    assert all(not json.loads(table.proto.columns)[column].get("help")
               for table in app.dataframe for column in table.value.columns
               if column in json.loads(table.proto.columns))
    labels = [expander.label for expander in app.expander]
    assert labels[-2:] == ["Definiciones", "Procedencia"]
    assert labels.count("Efecto de Porter") == 2
    assert [element.proto.label for element in app.get("button_group")] == ["Porter", "Métrica", "Métodos"]
    assert app.selectbox[0].label == "Consulta (qid)"


def test_provenance_offers_csv_and_json_downloads_side_by_side(reports):
    app = view_app(reports[0])
    downloads = app.get("download_button")
    assert [element.proto.label for element in downloads] == ["CSV", "JSON"]
