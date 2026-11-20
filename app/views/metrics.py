"""Métricas guardadas de FiQA: medias globales y detalle por consulta (#15)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import streamlit as st

from app.context import AppContext
from app.metrics_data import MetricsReport, ReportError, load_reports
from ris.comparison import CONFIGURATIONS

MODEL_NAMES = {"jaccard": "Jaccard", "tfidf": "TF-IDF", "bm25": "BM25", "dense": "Embeddings"}
PORTER_NAMES = {"false": "Sin Porter", "true": "Con Porter", "na": "No usa"}
PORTER_MODES = ("Sin Porter", "Con Porter", "Comparar")
NO_PORTER = PORTER_NAMES["na"]
TABLE_ROW_HEIGHT = 36
LARGE_TABLE_ROW_THRESHOLD = 500
PORTER_PREFIX = {"Sin Porter": "Sin ", "Con Porter": "Con "}  # la variante también se lee, no solo se ve
DECIMALS = 4  # solo presentación: los valores del reporte y sus descargas conservan todo
# Color de datos por variante: ciruela, teal y ocre (Embeddings no usa Porter).
VARIANT_COLORS = ("#7A2E5E", "#246477", "#A86A12")
ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THEME = {"backgroundColor": "#FBFAF9", "textColor": "#201A1F"}
try:
    with (ROOT / ".streamlit/config.toml").open("rb") as _theme_file:
        THEME = tomllib.load(_theme_file)["theme"]
except (OSError, KeyError, tomllib.TOMLDecodeError):
    THEME = _DEFAULT_THEME


def metric_labels(report: MetricsReport, *, per_query: bool = False) -> dict[str, str]:
    config = report.summary["config"]
    return {
        "precision": f"P@{config['k']}", "recall": f"R@{config['k']}",
        "ap_at_1000": f"{'AP' if per_query else 'MAP'}@{config['ap_cutoff']}",
    }


def metric_table(report: MetricsReport, qid: str | None = None) -> pd.DataFrame:
    """Las medias salen del JSON completo; seleccionar un qid nunca las altera."""
    if qid is None:
        entries = {(r["model"], r["stem"]): {
            "precision": r["mean_precision_at_k"], "recall": r["mean_recall_at_k"],
            "ap_at_1000": r["map_at_1000"],
        } for r in report.summary["results"]}
    else:
        entries = {(r["model"], r["stem"]): r for r in report.rows if r["qid"] == qid}
    return pd.DataFrame([
        {"Modelo": MODEL_NAMES[model], "Porter": PORTER_NAMES[stem], **entries[model, stem]}
        for model, stem in CONFIGURATIONS
    ])[["Modelo", "Porter", "precision", "recall", "ap_at_1000"]]


def filter_table(table: pd.DataFrame, porter: str, methods: list[str]) -> pd.DataFrame:
    """Filtra configuraciones sin alterar valores ni duplicar el denso."""
    variants = list(PORTER_NAMES.values()) if porter == "Comparar" else [porter, NO_PORTER]
    return table[table.Modelo.isin(methods) & table.Porter.isin(variants)].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _all_query_table(
    _rows: tuple[dict, ...], report_csv: bytes, porter: str, methods: tuple[str, ...], query: str,
) -> pd.DataFrame:
    """Construye y filtra el detalle; el CSV validado identifica esta versión del reporte."""
    table = pd.DataFrame(_rows)
    table["Modelo"] = table.model.map(MODEL_NAMES)
    table["Porter"] = table.stem.map(PORTER_NAMES)
    table = filter_table(table, porter, list(methods))
    return table[table.qid.str.contains(query, regex=False)]


def _leader_name(row) -> str:
    return row.Modelo if row.Porter == NO_PORTER else f"{row.Modelo} {row.Porter}"


def metric_summary(table: pd.DataFrame, metric: str) -> str:
    """Mejor valor visible, calculado con los valores sin redondear."""
    if table.empty:
        return "Sin datos."
    maximum = table[metric].max()
    if maximum == 0:
        return "Todos los valores son 0."
    names = [_leader_name(row) for row in
             table.loc[table[metric] == maximum, ["Modelo", "Porter"]].itertuples(index=False)]
    joined = names[0] if len(names) == 1 else f"{', '.join(names[:-1])} y {names[-1]}"
    return f"Mejor: {joined} {maximum:.{DECIMALS}f}"


def porter_differences(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Diferencias absolutas Con menos Sin dentro del mismo método y corte."""
    rows = []
    for method in table.Modelo.unique():
        if method == "Embeddings":
            continue
        values = table[table.Modelo == method].set_index("Porter")[metric]
        without, with_porter = values.get("Sin Porter"), values.get("Con Porter")
        delta = None if without is None or with_porter is None else with_porter - without
        rows.append({"Modelo": method, "Sin Porter": without, "Con Porter": with_porter,
                     "Cambio absoluto": delta})
    return pd.DataFrame(rows, columns=["Modelo", "Sin Porter", "Con Porter", "Cambio absoluto"])


def help_lines(report: MetricsReport) -> list[str]:
    config = report.summary["config"]
    k, cutoff = config["k"], config["ap_cutoff"]
    return [
        "Todas las métricas van de 0 a 1; más es mejor.",
        f"**P@{k}**: relevantes entre los {k} primeros, dividido por {k}.",
        f"**R@{k}**: relevantes entre los {k} primeros, dividido por todos los relevantes.",
        f"**MAP@{cutoff}**: media de la precisión media (AP) de cada consulta, hasta el puesto {cutoff}.",
        "**Porter**: reduce palabras a su raíz; Embeddings no lo usa. Cambio: Con menos Sin.",
    ]


def _chart_layers(test: str, offset: dict) -> list[dict]:
    """Barra y etiquetas de un subconjunto; el mejor valor lleva contorno y negrita."""
    bar_mark = {"type": "bar", "cornerRadiusEnd": 3, "strokeWidth": 2}
    if not offset:  # Embeddings: una sola barra centrada, del grosor de las otras
        bar_mark["height"] = {"band": .45}
    layers = [{
        "transform": [{"filter": test}], "mark": bar_mark,
        "encoding": {**offset, "stroke": {"condition": {"test": "datum.Mejor", "value": THEME["textColor"]},
                                          "value": "transparent"}},
    }]
    for flag, weight in (("!datum.Mejor", 400), ("datum.Mejor", 700)):
        layers.append({
            "transform": [{"filter": f"({test}) && {flag}"}],
            "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 6,
                     "fontSize": 13, "fontWeight": weight},
            "encoding": {**offset, "text": {"field": "Etiqueta", "type": "nominal"},
                         "color": {"value": THEME["textColor"]}},
        })
    return layers


def chart_spec(table: pd.DataFrame, metric: str, label: str) -> dict:
    """Barras agrupadas por método: Sin y Con Porter lado a lado; Embeddings, una sola barra."""
    best = table[metric].max() if len(table) else 0
    values = [
        {"Método": row["Modelo"], "Variante": row["Porter"], "Métrica": label, "Valor": row[metric],
         "Mejor": bool(best > 0 and row[metric] == best), "Etiqueta": f"{PORTER_PREFIX.get(row['Porter'], '')}{row[metric]:.{DECIMALS}f}"}
        for row in table.to_dict("records")
    ]
    colors = dict(zip(PORTER_NAMES.values(), VARIANT_COLORS, strict=True))
    present = [name for name in colors if name in set(table.Porter)] if len(table) else []
    both = {"Sin Porter", "Con Porter"} <= set(present)
    methods = table.Modelo.nunique() if len(table) else 0
    # Con un solo modo Porter no hay pares que agrupar: las barras quedan centradas.
    offset = {"yOffset": {"field": "Variante", "type": "nominal",
                          "scale": {"domain": ["Sin Porter", "Con Porter"]}}} if both else {}
    return {
        "data": {"values": values}, "height": max(1, methods) * (52 if both else 44) + 110,  # incluye leyenda y eje
        # Solo se reserva sitio a la etiqueta si el mayor valor llega al borde derecho.
        "padding": {"left": 0, "right": 56 if best > .85 else 8, "top": 4, "bottom": 0},
        "encoding": {
            "y": {"field": "Método", "type": "nominal", "sort": None,
                  "axis": {"title": None, "labelLimit": 120, "labelOverlap": False,
                           "ticks": False, "domain": False}},
            "x": {"field": "Valor", "type": "quantitative", "title": label,
                  "scale": {"domain": [0, 1]}, "axis": {"tickCount": 5, "format": ".1f"}},
            "color": {"field": "Variante", "type": "nominal",
                      "scale": {"domain": present, "range": [colors[name] for name in present]},
                      "legend": {"orient": "top", "title": None, "symbolType": "square",
                                 "labelFontSize": 12, "labelColor": THEME["textColor"],
                                 "symbolSize": 40, "columnPadding": 4,
                                 "labelExpr": f"datum.value == '{NO_PORTER}' ? 'Embeddings' : datum.value"}},
            "tooltip": [
                {"field": "Método", "type": "nominal"},
                {"field": "Variante", "type": "nominal"},
                {"field": "Métrica", "type": "nominal"},
                {"field": "Valor", "type": "quantitative", "format": ".6f"},
            ],
        },
        "layer": [*_chart_layers(f"datum.Variante != '{NO_PORTER}'", offset),
                  *_chart_layers(f"datum.Variante == '{NO_PORTER}'", {})],
        "config": {
            "font": "Schibsted Grotesk", "background": THEME["backgroundColor"],
            "view": {"stroke": None},
            "axis": {"labelColor": THEME["textColor"], "titleColor": THEME["textColor"],
                     "labelFontSize": 13, "titleFontSize": 13},
        },
    }


def _show_table(
    table: pd.DataFrame, labels: dict[str, str], *, description: str, key: str,
    decimals: int = DECIMALS,
) -> None:
    # Los fondos quedan a cargo del grid para conservar su hover transitorio.
    if len(table) >= LARGE_TABLE_ROW_THRESHOLD:
        styled = table
    else:
        styled = table.style.set_properties(color=THEME["textColor"])
        positive_metrics = [name for name in labels if table[name].max() > 0]
        if positive_metrics:
            styled = styled.highlight_max(subset=positive_metrics, props="font-weight: 700;")
    st.dataframe(
        styled, hide_index=True, width="stretch", alt=description, key=key,
        height=min(360, TABLE_ROW_HEIGHT * (len(table) + 1) + 4), row_height=TABLE_ROW_HEIGHT,
        column_config={
            "qid": st.column_config.TextColumn("Consulta (qid)", width=110, pinned=True),
            "Modelo": st.column_config.TextColumn("Método", width=100,
                                                  pinned="qid" not in table.columns),
            "Porter": st.column_config.TextColumn("Variante", width=95),
            **{name: st.column_config.NumberColumn(label, format=f"%.{decimals}f", width=105,
                                                   alignment="right")
               for name, label in labels.items()},
        },
    )


def _provenance(report: MetricsReport) -> None:
    summary = report.summary
    config = summary["config"]
    with st.expander("Procedencia"):
        st.markdown(f"Esquema **{summary['schema_version']}**, profundidad **{config['depth']}**.")
        st.markdown("**Conteos**")
        st.json(summary["counts"], expanded=True)
        st.markdown("**Configuración**")
        st.json(summary["config"], expanded=False)
        st.markdown("**Datos, artefactos y SHA-256**")
        st.json(summary["provenance"], expanded=False)
        st.markdown("**Versiones**")
        st.json(summary["versions"], expanded=True)
        st.markdown("**Resultados y huellas de rankings**")
        st.json(summary["results"], expanded=False)
        st.caption("CSV y JSON coinciden por SHA-256. Los originales conservan todos los decimales.")
        with st.container(horizontal=True, gap="small"):
            st.download_button("CSV", report.csv_bytes, file_name="metrics_per_query.csv",
                               mime="text/csv", key="metrics_csv", icon=":material/download:")
            st.download_button("JSON", report.summary_bytes, file_name="metrics_summary.json",
                               mime="application/json", key="metrics_json",
                               icon=":material/download:")


def _check_report_unchanged(directory: Path, report: MetricsReport) -> None:
    """Reutiliza el par validado solo si ambos archivos siguen idénticos.

    Si cambian o faltan, el rerun completo vuelve a validar y actualiza toda la
    vista, incluida la procedencia; un fragmento nunca muestra un par mezclado.
    """
    try:
        unchanged = (
            (directory / "metrics_per_query.csv").read_bytes() == report.csv_bytes
            and (directory / "metrics_summary.json").read_bytes() == report.summary_bytes
        )
    except OSError:
        unchanged = False
    if not unchanged:
        st.rerun()


def _reset_filters(first_qid: str) -> None:
    for key, value in {
        "metrics_porter": "Comparar", "metrics_methods": list(MODEL_NAMES.values()),
        "metrics_metric": "ap_at_1000", "metrics_qid": first_qid,
        "metrics_qid_filter": "", "metrics_explore": False,
    }.items():
        st.session_state[key] = value


def _restore_methods() -> None:
    st.session_state["metrics_methods"] = list(MODEL_NAMES.values())


def _clear_query_filter() -> None:
    st.session_state["metrics_qid_filter"] = ""


def _show_differences(table: pd.DataFrame, metric: str, *, key: str) -> None:
    with st.expander("Efecto de Porter", key=f"{key}_panel", on_change="rerun") as panel:
        if not panel.open:
            return
        st.caption("Con menos Sin; diferencia absoluta.")
        differences = porter_differences(table, metric)
        if differences.empty:
            st.info("Sin pares con y sin Porter.")
            return
        st.dataframe(
            differences.style.set_properties(color=THEME["textColor"]),
            hide_index=True, width="stretch", key=key, row_height=TABLE_ROW_HEIGHT,
            height=min(360, TABLE_ROW_HEIGHT * (len(differences) + 1) + 4),
            alt="Cambio absoluto entre Con Porter y Sin Porter para cada método.",
            column_config={
                "Modelo": st.column_config.TextColumn("Método", width=100, pinned=True),
                "Sin Porter": st.column_config.NumberColumn("Sin", format="%.6f", width=95),
                "Con Porter": st.column_config.NumberColumn("Con", format="%.6f", width=95),
                "Cambio absoluto": st.column_config.NumberColumn(
                    "Cambio", format="%+.6f", width=100),
            },
        )


def data_line(report: MetricsReport) -> str:
    """Única línea de contexto: tamaño de la evaluación."""
    counts = report.summary["counts"]

    def number(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    return f"FiQA test: {number(counts['n_queries'])} consultas, {number(counts['n_corpus_docs'])} documentos"


@st.fragment
def _query_detail(
    directory: Path, report: MetricsReport, porter: str, methods: list[str], metric: str,
) -> None:
    _check_report_unchanged(directory, report)
    with st.container(key="metrics_query_detail"):
        st.subheader("Detalle por consulta", anchor=False)
        if st.session_state.get("metrics_qid") not in report.qids:
            st.session_state["metrics_qid"] = report.qids[0]
        with st.container(key="metrics_detail_controls", horizontal=True,
                          vertical_alignment="bottom", gap="medium"):
            qid = st.selectbox("Consulta (qid)", report.qids, key="metrics_qid",
                               persist_state="session", width=260)
            explore = st.checkbox("Explorar todas las consultas", key="metrics_explore",
                                  persist_state="session")
        labels = metric_labels(report, per_query=True)
        query_table = filter_table(metric_table(report, qid), porter, methods)
        with st.container(key=f"jab_query_{qid}"):
            st.markdown(f"**Consulta {qid}**")
            _show_table(query_table, labels, description=f"Resultados de la consulta {qid}.",
                        key="metrics_query_table", decimals=6)
        if porter == "Comparar":
            _show_differences(query_table, metric, key="metrics_query_differences")

    if explore:
        search = st.text_input("Filtrar por qid", key="metrics_qid_filter",
                               persist_state="session", live=True,
                               placeholder="Ej.: 100")
        all_rows = _all_query_table(report.rows, report.csv_bytes, porter, tuple(methods), search.strip())
        n_queries, n_rows = all_rows.qid.nunique(), len(all_rows)
        st.caption(f"{n_queries} consulta{'s' if n_queries != 1 else ''}, "
                   f"{n_rows} fila{'s' if n_rows != 1 else ''}")
        if all_rows.empty:
            st.info("No hay consultas que coincidan.")
            st.button("Limpiar filtro", key="metrics_clear_query_filter",
                      on_click=_clear_query_filter)
        else:
            _show_table(all_rows[["qid", "Modelo", "Porter", *labels]], labels,
                        description=f"Consultas filtradas: {porter}.",
                        key="metrics_all_queries", decimals=6)


def _dashboard(directory: Path, report: MetricsReport) -> None:
    labels = metric_labels(report)
    with st.container(key="metrics_controls", horizontal=True, vertical_alignment="bottom", gap="medium"):
        porter = st.segmented_control(
            "Porter", PORTER_MODES,
            default=None if "metrics_porter" in st.session_state else "Comparar", required=True,
            key="metrics_porter", persist_state="session",
        )
        metric = st.segmented_control(
            "Métrica", list(labels), format_func=labels.get, required=True,
            default=None if "metrics_metric" in st.session_state else "ap_at_1000",
            key="metrics_metric", persist_state="session",
        )
        methods = st.pills("Métodos", list(MODEL_NAMES.values()), selection_mode="multi",
                           default=None if "metrics_methods" in st.session_state else list(MODEL_NAMES.values()),
                           key="metrics_methods", persist_state="session", wrap=True)
        st.button("Restablecer", key="metrics_reset", icon=":material/restart_alt:", type="tertiary",
                  on_click=_reset_filters, args=(report.qids[0],))

    table = filter_table(metric_table(report), porter, methods)
    if table.empty:
        st.caption(data_line(report))
        st.info("No hay métodos seleccionados.")
        st.button("Mostrar todos los métodos", key="metrics_restore_methods", on_click=_restore_methods)
        return

    with st.container(key="metrics_heading_row", horizontal=True, gap="medium"):
        st.subheader("Resultados globales", anchor=False)
        st.caption(metric_summary(table, metric))
    left, right = st.columns([9, 10], gap="medium")
    with left:
        st.vega_lite_chart(spec=chart_spec(table, metric, labels[metric]), width="stretch",
                           key="metrics_chart",
                           alt=f"{labels[metric]} por método y variante de Porter; escala de 0 a 1.")
    with right:
        _show_table(table, labels, description=f"Medias globales: {porter}.", key="metrics_global_table")
    st.caption(data_line(report))
    if porter == "Comparar":
        _show_differences(table, metric, key="metrics_global_differences")
    _query_detail(directory, report, porter, methods, metric)


def _interpretation(report: MetricsReport) -> None:
    with st.expander("Definiciones"):
        st.markdown("\n".join(f"- {line}" for line in help_lines(report)))


def render(context: AppContext) -> None:
    """Consume únicamente reports_dir; independiente del buscador y de su motor."""
    with st.container(key="jab_metrics"):
        try:
            with st.spinner("Validando reportes…"):
                report = load_reports(context.reports_dir)
        except ReportError as error:
            st.error("No se pueden mostrar las métricas: reportes ausentes o no válidos.")
            st.write(str(error))
            st.info(
                "Restaure juntos metrics_per_query.csv y metrics_summary.json en JAB_REPORTS_DIR "
                "(reports/ por defecto) y pulse «Reintentar carga»."
            )
            st.button("Reintentar carga", key="metrics_retry")
            return

        _dashboard(context.reports_dir, report)
        _interpretation(report)
        _provenance(report)
