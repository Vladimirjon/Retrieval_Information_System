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
ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THEME = {
    "primaryColor": "#7A2E5E", "backgroundColor": "#FBFAF9",
    "secondaryBackgroundColor": "#F3F1F2", "textColor": "#201A1F",
}
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


def chart_spec(table: pd.DataFrame, metric: str, label: str) -> dict:
    """Siete barras etiquetadas; dominio [0,1] común, sin codificación solo por color."""
    values = [
        {"Configuración": f"{row['Modelo']} · {row['Porter'].lower()}", "Valor": row[metric]}
        for row in table.to_dict("records")
    ]
    return {
        "data": {"values": values}, "height": 240,
        "mark": {"type": "bar", "color": THEME["primaryColor"]},
        "encoding": {
            "y": {"field": "Configuración", "type": "nominal", "sort": None,
                  "axis": {"title": None, "labelLimit": 200, "ticks": False, "domain": False}},
            "x": {"field": "Valor", "type": "quantitative", "title": label,
                  "scale": {"domain": [0, 1]}, "axis": {"tickCount": 5, "format": ".1f"}},
            "tooltip": [
                {"field": "Configuración", "type": "nominal"},
                {"field": "Valor", "type": "quantitative", "title": label, "format": ".8f"},
            ],
        },
        "config": {
            "font": "Schibsted Grotesk", "background": THEME["backgroundColor"],
            "view": {"stroke": None},
            "axis": {"labelColor": THEME["textColor"], "titleColor": THEME["textColor"],
                     "labelFontSize": 13, "titleFontSize": 13},
        },
    }


def _show_table(table: pd.DataFrame, labels: dict[str, str], *, description: str) -> None:
    st.dataframe(
        table, hide_index=True, width="stretch", alt=description,
        column_config={
            "Modelo": st.column_config.TextColumn(width="small"),
            "Porter": st.column_config.TextColumn(width="small"),
            **{key: st.column_config.NumberColumn(label, format="%.6f", width="small")
               for key, label in labels.items()},
        },
    )


def _provenance(report: MetricsReport) -> None:
    summary = report.summary
    with st.expander("Procedencia y configuración"):
        st.caption(
            "Datos declarados en la evaluación guardada. Se comprueban los reportes; "
            "esta pantalla no vuelve a verificar el corpus ni los índices."
        )
        st.markdown(
            f"Esquema **{summary['schema_version']}** · profundidad recuperada "
            f"**{summary['config']['depth']}** · corte de AP **{summary['config']['ap_cutoff']}**."
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


def render(context: AppContext) -> None:
    """Consume únicamente reports_dir; independiente del buscador y de su motor."""
    with st.container(key="jab_metrics"):
        st.header("Métricas")
        st.caption(
            "Evaluación guardada de FiQA. La consulta, el modelo, top-n y Porter "
            "de la barra de búsqueda no cambian estos resultados."
        )
        st.button("Volver a leer reportes", key="metrics_reload", help="Relee y valida ambos archivos.")
        try:
            report = load_reports(context.reports_dir)
        except ReportError as error:
            st.error("No se pueden mostrar las métricas: los reportes no están disponibles o no son válidos.")
            st.write(str(error))
            st.info(
                "Restaure juntos metrics_per_query.csv y metrics_summary.json de una evaluación "
                "válida en la carpeta indicada por JAB_REPORTS_DIR (reports/ por defecto). "
                "Después pulse «Volver a leer reportes»."
            )
            return

        config, counts = report.summary["config"], report.summary["counts"]
        st.subheader("Resumen")
        st.caption(
            f"Split: {config['split']} · P@{config['k']} y R@{config['k']} · "
            f"AP/MAP@{config['ap_cutoff']} · profundidad {config['depth']}."
        )
        for column, (label, value) in zip(st.columns(3), (
            ("Consultas evaluadas", counts["n_queries"]),
            ("Documentos", counts["n_corpus_docs"]),
            ("Configuraciones", len(report.summary["results"])),
        ), strict=True):
            column.metric(label, f"{value:,}".replace(",", " "))

        st.subheader("Comparación de modelos")
        st.caption(
            f"Medias sobre las {counts['n_queries']} consultas, incluidos los ceros. "
            "Embeddings se evalúa una sola vez: Porter no aplica."
        )
        labels = metric_labels(report)
        global_table = metric_table(report)
        metric = st.radio("Métrica del gráfico", list(labels), index=2, horizontal=True,
                          format_func=labels.get, key="metrics_metric")
        st.vega_lite_chart(spec=chart_spec(global_table, metric, labels[metric]), width="stretch",
                           alt=f"{labels[metric]} global de siete configuraciones. Valores en la tabla siguiente.")
        _show_table(global_table, labels, description="Medias globales de las siete configuraciones.")
        st.caption("Escala de 0 a 1 · tabla con seis decimales; originales completos en Procedencia.")

        st.subheader("Detalle por consulta")
        if st.session_state.get("metrics_qid") not in report.qids:
            st.session_state["metrics_qid"] = report.qids[0]
        qid = st.selectbox("Consulta evaluada (qid)", report.qids, key="metrics_qid",
                           help="Busque un identificador como texto. Orden lexicográfico de los qids.")
        st.caption(
            f"qid: {qid} · consulta del split {config['split']}, independiente del texto de la barra. "
            f"AP@{config['ap_cutoff']} corresponde a esta consulta; MAP es la media global."
        )
        _show_table(metric_table(report, qid), metric_labels(report, per_query=True),
                    description=f"Precisión, recall y AP de las siete configuraciones para el qid {qid}.")
        with st.expander("Cómo leer las métricas"):
            st.write(
                f"P@{config['k']}: relevantes en las primeras {config['k']} posiciones, "
                f"divididos por {config['k']}, aunque el ranking sea más corto. "
                f"R@{config['k']}: esos aciertos divididos por todos los relevantes de la consulta. "
                f"AP@{config['ap_cutoff']}: suma de la precisión en cada posición relevante hasta "
                f"{config['ap_cutoff']}, dividida por todos los relevantes. "
                f"MAP@{config['ap_cutoff']}: media de AP de todas las consultas evaluadas. "
                "Sin relevantes o sin ranking, la consulta aporta cero."
            )
        _provenance(report)
