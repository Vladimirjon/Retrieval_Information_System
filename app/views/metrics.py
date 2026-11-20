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
PORTER_NAMES = {"false": "Sin Porter", "true": "Con Porter", "na": "No aplica"}
PORTER_MODES = ("Sin Porter", "Con Porter", "Comparar")
TABLE_ROW_HEIGHT = 36
VARIANT_COLORS = ("#7A2E5E", "#246477", "#5E5560")
METHOD_HELP = (
    "Jaccard: coincidencia de palabras distintas. TF-IDF: coseno de palabras ponderadas. "
    "BM25: coincidencia léxica con ajuste de frecuencia y longitud. "
    "Embeddings: similitud semántica con MiniLM; Porter no aplica."
)
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
    variants = list(PORTER_NAMES.values()) if porter == "Comparar" else [porter, "No aplica"]
    return table[table.Modelo.isin(methods) & table.Porter.isin(variants)].reset_index(drop=True)


def metric_summary(table: pd.DataFrame, metric: str, label: str) -> str:
    """Resumen legible del máximo visible, usando los valores sin redondear."""
    if table.empty:
        return "No hay configuraciones visibles."
    maximum = table[metric].max()
    if maximum == 0:
        return f"Todas las configuraciones visibles tienen {label} = 0."
    leaders = table.loc[table[metric] == maximum, ["Modelo", "Porter"]]
    names = ", ".join(f"{row.Modelo} ({row.Porter})" for row in leaders.itertuples(index=False))
    tie = " · empate" if len(leaders) > 1 else ""
    return f"Mayor **{label}** visible: **{maximum:.6f}** · {names}{tie}."


def porter_differences(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Diferencias absolutas Con − Sin dentro del mismo método y corte."""
    rows = []
    for method in table.Modelo.unique():
        if method == "Embeddings":
            continue
        values = table[table.Modelo == method].set_index("Porter")[metric]
        without, with_porter = values.get("Sin Porter"), values.get("Con Porter")
        delta = None if without is None or with_porter is None else with_porter - without
        status = ("Falta Sin Porter" if without is None else "Falta Con Porter") if delta is None else (
            "Aumenta" if delta > 0 else "Disminuye" if delta < 0 else "Sin cambio"
        )
        rows.append({"Modelo": method, "Sin Porter": without, "Con Porter": with_porter,
                     "Cambio absoluto": delta, "Resultado": status})
    return pd.DataFrame(rows, columns=[
        "Modelo", "Sin Porter", "Con Porter", "Cambio absoluto", "Resultado",
    ])


def metric_help(report: MetricsReport, *, per_query: bool = False) -> dict[str, str]:
    k, cutoff = report.summary["config"]["k"], report.summary["config"]["ap_cutoff"]
    prefix = "" if per_query else "Media sobre todas las consultas de test. "
    return {
        "precision": prefix + f"Relevantes entre los primeros {k}, divididos por {k}. "
                     "Un ranking corto también divide por ese corte. Más es mejor; escala 0–1.",
        "recall": prefix + f"Relevantes entre los primeros {k}, divididos por todos los relevantes "
                  "de la consulta. Más es mejor; escala 0–1.",
        "ap_at_1000": (
            f"AP@{cutoff}: suma la precisión en cada posición relevante hasta {cutoff} "
            "y divide por todos los relevantes. " if per_query else
            f"MAP@{cutoff}: media de AP@{cutoff} sobre todas las consultas test, incluidos los ceros. "
        ) + "Más es mejor; escala 0–1.",
    }


def chart_spec(table: pd.DataFrame, metric: str, label: str) -> dict:
    """Barras por configuración con escala común [0,1]; el mayor valor lleva contorno y la palabra «mayor»."""
    best = table[metric].max() if len(table) else 0
    values = [
        {"Configuración": f"{row['Modelo']} · {row['Porter']}", "Método": row["Modelo"],
         "Variante": row["Porter"], "Métrica": label, "Valor": row[metric],
         "Mayor": bool(best > 0 and row[metric] == best),
         "Etiqueta": f"{row[metric]:.4f}" + (" · mayor" if best > 0 and row[metric] == best else "")}
        for row in table.to_dict("records")
    ]
    return {
        "data": {"values": values}, "height": 30 * len(values) + 64,
        # Solo se reserva sitio a la etiqueta si el mayor valor llega al borde derecho.
        "padding": {"left": 0, "right": 96 if best > .7 else 8, "top": 4, "bottom": 0},
        "encoding": {
            "y": {"field": "Configuración", "type": "nominal", "sort": None,
                  "axis": {"title": None, "labelLimit": 170, "labelOverlap": False,
                           "ticks": False, "domain": False}},
            "x": {"field": "Valor", "type": "quantitative", "title": label,
                  "scale": {"domain": [0, 1]}, "axis": {"tickCount": 5, "format": ".1f"}},
            "color": {"field": "Variante", "type": "nominal",
                      "scale": {"domain": list(PORTER_NAMES.values()), "range": list(VARIANT_COLORS)},
                      "legend": None},
            "tooltip": [
                {"field": "Método", "type": "nominal"},
                {"field": "Variante", "type": "nominal"},
                {"field": "Métrica", "type": "nominal"},
                {"field": "Valor", "type": "quantitative", "format": ".6f"},
            ],
        },
        "layer": [
            {"mark": {"type": "bar", "cornerRadiusEnd": 3, "strokeWidth": 2},
             "encoding": {"stroke": {"condition": {"test": "datum.Mayor", "value": THEME["textColor"]},
                                     "value": "transparent"}}},
            *({"transform": [{"filter": test}],
               "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 6,
                        "fontSize": 13, "fontWeight": weight},
               "encoding": {"text": {"field": "Etiqueta", "type": "nominal"},
                            "color": {"value": THEME["textColor"]}}}
              for test, weight in (("!datum.Mayor", 400), ("datum.Mayor", 700))),
        ],
        "config": {
            "font": "Schibsted Grotesk", "background": THEME["backgroundColor"],
            "view": {"stroke": None},
            "axis": {"labelColor": THEME["textColor"], "titleColor": THEME["textColor"],
                     "labelFontSize": 13, "titleFontSize": 13},
        },
    }


def _show_table(
    table: pd.DataFrame, labels: dict[str, str], *, description: str, key: str,
) -> None:
    # Los fondos quedan a cargo del grid para conservar su hover transitorio.
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
            **{name: st.column_config.NumberColumn(f"{label} ↑", format="%.6f", width=105,
                                                   alignment="right")
               for name, label in labels.items()},
        },
    )


def _provenance(report: MetricsReport) -> None:
    summary = report.summary
    config = summary["config"]
    with st.expander("Procedencia y configuración"):
        st.caption(
            "Datos declarados en la evaluación guardada. Se comprueban los reportes; "
            "esta pantalla no vuelve a verificar el corpus ni los índices."
        )
        st.markdown(
            f"Esquema **{summary['schema_version']}** · P@{config['k']} · R@{config['k']} · "
            f"AP/MAP@{config['ap_cutoff']} · profundidad evaluada **{config['depth']}**."
        )
        st.markdown("**Conteos**")
        st.json(summary["counts"], expanded=True)
        st.markdown("**Configuración de la evaluación y de los modelos**")
        st.json(summary["config"], expanded=False)
        st.markdown("**Datos, artefactos y huellas SHA-256**")
        st.json(summary["provenance"], expanded=False)
        st.markdown("**Versiones de la evaluación**")
        st.json(summary["versions"], expanded=True)
        st.markdown("**Resultados y huellas de los rankings**")
        st.json(summary["results"], expanded=False)
        st.caption(
            "CSV y JSON coinciden por SHA-256; cobertura y medias contrastadas con "
            "tolerancia absoluta de 1e-12 (sin tolerancia relativa)."
        )
        st.download_button("Descargar CSV original", report.csv_bytes,
                           file_name="metrics_per_query.csv", mime="text/csv",
                           key="metrics_csv")
        st.download_button("Descargar JSON original", report.summary_bytes,
                           file_name="metrics_summary.json", mime="application/json",
                           key="metrics_json")
        st.caption("Los originales conservan todos los decimales de la evaluación.")


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


def _show_differences(table: pd.DataFrame, metric: str, label: str, *, key: str) -> None:
    with st.expander(f"Diferencias con Porter · {label}", key=f"{key}_panel", on_change="rerun") as panel:
        if not panel.open:
            return
        st.caption("Con Porter − Sin Porter · diferencia absoluta, no porcentual. + indica aumento.")
        differences = porter_differences(table, metric)
        if differences.empty:
            st.info("No hay pares clásicos que comparar. Porter no se aplica a Embeddings.")
            return
        st.dataframe(
            differences.style.set_properties(color=THEME["textColor"]),
            hide_index=True, width="stretch", key=key, row_height=TABLE_ROW_HEIGHT,
            height=min(360, TABLE_ROW_HEIGHT * (len(differences) + 1) + 4),
            alt=f"Cambio absoluto de {label} entre variantes del mismo método; Embeddings no aplica.",
            column_config={
                "Modelo": st.column_config.TextColumn("Método", width=100, pinned=True),
                **{name: st.column_config.NumberColumn(name, format="%.6f", width=110)
                   for name in ("Sin Porter", "Con Porter")},
                "Cambio absoluto": st.column_config.NumberColumn(
                    "Δ absoluto", format="%+.6f", width=120,
                    help=f"{label} con Porter menos {label} sin Porter. No es un porcentaje ni "
                         "una prueba de significancia.",
                ),
                "Resultado": st.column_config.TextColumn(width=120),
            },
        )


def data_line(report: MetricsReport, n_visible: int) -> str:
    """Única línea de contexto: procedencia y tamaño de la evaluación."""
    counts = report.summary["counts"]

    def number(value: int) -> str:
        return f"{value:,}".replace(",", " ")

    return (
        "FiQA (test), independiente del buscador · "
        f"{number(counts['n_queries'])} consultas · {number(counts['n_corpus_docs'])} documentos · "
        f"{n_visible} configuraciones visibles"
    )


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
            qid = st.selectbox("Consulta evaluada (qid)", report.qids, key="metrics_qid",
                               persist_state="session", width=260)
            explore = st.checkbox("Explorar todas las consultas", key="metrics_explore",
                                  persist_state="session")
        labels = metric_labels(report, per_query=True)
        query_table = filter_table(metric_table(report, qid), porter, methods)
        with st.container(key=f"jab_query_{qid}"):
            st.markdown(f"**Resultados de la consulta {qid}** · {porter}")
            _show_table(query_table, labels, description=f"Resultados de la consulta {qid}: {porter}.",
                        key="metrics_query_table")
        if porter == "Comparar":
            _show_differences(query_table, metric, labels[metric], key="metrics_query_differences")

    if explore:
        search = st.text_input("Filtrar identificadores de consulta", key="metrics_qid_filter",
                               persist_state="session", live=True,
                               placeholder="Coincidencia literal en el qid, por ejemplo: 100")
        all_rows = pd.DataFrame(report.rows)
        all_rows["Modelo"] = all_rows.model.map(MODEL_NAMES)
        all_rows["Porter"] = all_rows.stem.map(PORTER_NAMES)
        all_rows = filter_table(all_rows, porter, methods)
        all_rows = all_rows[all_rows.qid.str.contains(search.strip(), regex=False)]
        n_queries, n_rows = all_rows.qid.nunique(), len(all_rows)
        st.caption(f"{n_queries} consulta{'s' if n_queries != 1 else ''} · "
                   f"{n_rows} fila{'s' if n_rows != 1 else ''} · {porter}.")
        if all_rows.empty:
            st.info("No hay consultas que coincidan con ese filtro. Bórrelo para ver el listado completo.")
            st.button("Limpiar filtro de consultas", key="metrics_clear_query_filter",
                      on_click=_clear_query_filter)
        else:
            _show_table(all_rows[["qid", "Modelo", "Porter", *labels]], labels,
                        description=f"Consultas filtradas: {porter}; ordenar pulsando los encabezados.",
                        key="metrics_all_queries")


def _dashboard(directory: Path, report: MetricsReport) -> None:
    labels = metric_labels(report)
    with st.container(key="metrics_controls", horizontal=True, vertical_alignment="bottom", gap="medium"):
        porter = st.segmented_control(
            "Variante de Porter", PORTER_MODES,
            default=None if "metrics_porter" in st.session_state else "Comparar", required=True,
            key="metrics_porter", persist_state="session",
        )
        metric = st.segmented_control(
            "Métrica destacada", list(labels), format_func=labels.get, required=True,
            default=None if "metrics_metric" in st.session_state else "ap_at_1000",
            key="metrics_metric", persist_state="session",
        )
        methods = st.pills("Métodos visibles", list(MODEL_NAMES.values()), selection_mode="multi",
                           default=None if "metrics_methods" in st.session_state else list(MODEL_NAMES.values()),
                           key="metrics_methods", persist_state="session", wrap=True)
        st.button("Restablecer", key="metrics_reset", icon=":material/restart_alt:", type="tertiary",
                  on_click=_reset_filters, args=(report.qids[0],), help="Vuelve a los filtros iniciales.")

    table = filter_table(metric_table(report), porter, methods)
    if table.empty:
        st.caption(data_line(report, 0))
        st.info("No hay métodos seleccionados. Elija uno arriba o muestre todos los métodos.")
        st.button("Mostrar todos los métodos", key="metrics_restore_methods", on_click=_restore_methods)
        return

    with st.container(key="metrics_heading_row", horizontal=True, gap="medium"):
        st.subheader("Resultados globales", anchor=False)
        st.caption(metric_summary(table, metric, labels[metric]))
    left, right = st.columns([9, 10], gap="medium")
    with left:
        st.vega_lite_chart(spec=chart_spec(table, metric, labels[metric]), width="stretch",
                           key="metrics_chart",
                           alt=f"{labels[metric]}: {porter}; {len(table)} configuraciones; escala de 0 a 1.")
    with right:
        _show_table(table, labels, description=f"Medias globales: {porter}.", key="metrics_global_table")
    st.caption(f"{data_line(report, len(table))} · negrita: mejor valor; pulse un encabezado para ordenar.")
    if porter == "Comparar":
        _show_differences(table, metric, labels[metric], key="metrics_global_differences")
    _query_detail(directory, report, porter, methods, metric)


def _interpretation(report: MetricsReport) -> None:
    labels, helps = metric_labels(report), metric_help(report)
    with st.expander("Cómo interpretar estos resultados"):
        st.markdown("**Métricas · valores de 0 a 1; un valor mayor indica mejor recuperación.**")
        for metric, label in labels.items():
            st.markdown(f"- **{label}**: {helps[metric]}")
        st.write("En el detalle, AP corresponde solo a la consulta elegida; MAP promedia todas las "
                 "consultas evaluadas. Sin relevantes o sin ranking, la consulta aporta cero.")
        st.markdown("**Métodos y variantes**")
        st.write(METHOD_HELP)
        st.write("Porter se aplica al preprocesamiento de documentos y consultas clásicas. No implica "
                 "una mejora automática: puede aumentar o reducir una métrica. Embeddings codifica el "
                 "texto original y se muestra una sola vez, con variante «No aplica».")
        st.markdown("**Comparaciones**")
        st.write("Las diferencias son absolutas (Con Porter − Sin Porter), para el mismo método, "
                 "métrica, corte y conjunto test. No son porcentajes ni pruebas de significancia. "
                 "Embeddings no tiene una variante Porter con la que calcular esa diferencia.")
        st.write("Las medias se leen del reporte completo: filtrar métodos, variantes o qids en el "
                 "listado no vuelve a evaluar ni altera esas medias. Las tablas usan seis decimales; "
                 "los originales descargables conservan toda su precisión. En pantallas pequeñas, "
                 "deslice la tabla horizontalmente o abra su vista de pantalla completa.")


def render(context: AppContext) -> None:
    """Consume únicamente reports_dir; independiente del buscador y de su motor."""
    with st.container(key="jab_metrics"):
        try:
            with st.spinner("Validando los reportes…"):
                report = load_reports(context.reports_dir)
        except ReportError as error:
            st.error("No se pueden mostrar las métricas: los reportes no están disponibles o no son válidos.")
            st.write(str(error))
            st.info(
                "Restaure juntos metrics_per_query.csv y metrics_summary.json de una evaluación "
                "válida en la carpeta indicada por JAB_REPORTS_DIR (reports/ por defecto). "
                "Después pulse «Reintentar carga»."
            )
            st.button("Reintentar carga", key="metrics_retry", help="Vuelve a leer y validar los reportes.")
            return

        _dashboard(context.reports_dir, report)
        _interpretation(report)
        _provenance(report)
