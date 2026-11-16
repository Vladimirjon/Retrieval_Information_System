"""Vista de métricas de evaluación. La implementa #15 (Jorge Torres).

Contrato: `render(context)` recibe un `app.context.AppContext`. Lee las métricas
de `context.reports_dir` (esquema en `reports/README.md`) y valida
`provenance.metrics_per_query_sha256` con `scripts.evaluate.verify_reports` antes
de mostrarlas. No recalcula métricas ni usa `context.engine`.
"""

from __future__ import annotations

import streamlit as st

from app.context import AppContext


def render(context: AppContext) -> None:
    st.info("La vista de métricas se implementa en la issue #15.")
