"""jab: buscador sobre FiQA con Jaccard, TF-IDF, BM25 y embeddings (issue #13).

Ejecutar desde la raíz del repositorio:
    python -m streamlit run app/streamlit_app.py

Variables de entorno opcionales:
    JAB_ARTIFACTS_DIR  carpeta con los índices (por defecto, ./artifacts)
    JAB_REPORTS_DIR    carpeta con las métricas de #11 (por defecto, ./reports)

La interfaz de búsqueda es un componente propio (app/frontend/); las vistas de
comparación (#14) y métricas (#15) se dibujan debajo con Streamlit.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for import_root in (ROOT / "src", ROOT):
    if str(import_root) not in sys.path:  # funciona sin `pip install -e .`
        sys.path.insert(0, str(import_root))

import streamlit as st

from app.context import AppContext
from app.ui import jab_component, parse_request, respond
from app.views import comparison, metrics
from ris.artifacts import ensure_artifacts
from ris.data import load_queries
from ris.engine import CLASSICAL_MODELS, EngineError, SearchEngine

PAGE_CSS = """
<style>
header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.block-container { max-width: none; padding: 0 0 4rem; }
.stApp { font-family: "Schibsted Grotesk", ui-sans-serif, system-ui, sans-serif; }
.st-key-jab_views { width: min(720px, 100% - 32px); margin: 0 auto; }
.st-key-jab_views:has(.st-key-jab_metrics) { width: min(1040px, 100% - 32px); }
.jab[data-view="metricas"] .jab-main { width: min(1040px, 100% - 32px); }
.st-key-jab_metrics [data-testid="stCaptionContainer"] p { color: #5E5560; }
.st-key-metrics_controls { background: #F3F1F2; }
.st-key-metrics_controls button { min-height: 42px; }
.st-key-jab_metrics :is(button, input, [role="radio"], [role="checkbox"], [role="combobox"]):focus-visible {
  outline: 2px solid #7A2E5E; outline-offset: 3px;
}
.st-key-jab_metrics .stTooltipIcon button { min-width: 28px; min-height: 28px; }
@media (max-width: 600px) {
  .st-key-jab_views:has(.st-key-jab_metrics) { width: calc(100% - 24px); }
  .jab[data-view="metricas"] .jab-main { width: calc(100% - 24px); }
  .st-key-jab_metrics [data-testid="stMetricValue"] { font-size: 1.75rem; }
}
/* La navegación del componente es inmediata; no esperar al fin del rerun para
   ocultar la vista anterior que Streamlit conserva durante la transición. */
body:has(.jab[data-view="buscar"]) .st-key-jab_views,
body:has(.jab:not([data-view="metricas"])) .st-key-jab_metrics { display: none; }
/* Solo el buscador tiene su propio indicador de carga. Las vistas nativas
   conservan la atenuación de Streamlit mientras actualizan sus datos. */
[data-testid="stElementContainer"][data-stale="true"]:has(.jab) { opacity: 1 !important; transition: none !important; }
.st-key-jab_metrics [class*="st-key-jab_query_"] { border-radius: 8px; animation: jab-query-update .45s ease-out; }
@keyframes jab-query-update { from { box-shadow: 0 0 0 2px #7A2E5E40; } to { box-shadow: 0 0 0 2px transparent; } }
.jab-loading { min-height: 85vh; display: grid; place-content: center; justify-items: center; gap: 24px; color: #201A1F; text-align: center; }
.jab-loading svg { width: 120px; height: auto; }
.jab-loading .dot { fill: #7A2E5E; }
.jab-loading-bar { width: min(440px, 80vw); height: 56px; border-radius: 28px; background: #F3F1F2; animation: jab-load 1.4s ease-in-out infinite alternate; }
.jab-loading p { margin: 0; max-width: 36ch; color: #5E5560; font-size: 14px; }
@keyframes jab-load { from { opacity: 1; transform: scaleX(1); } to { opacity: .65; transform: scaleX(.98); } }
@media (prefers-reduced-motion: reduce) {
  .jab-loading-bar, .st-key-jab_metrics [class*="st-key-jab_query_"] { animation: none; }
}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap">
"""


def _warm(engine: SearchEngine, models: tuple[str, ...]) -> None:
    """Carga los modelos pedidos y registra sus tiempos; un fallo no detiene la app."""
    first = len(engine.load_events)
    for model in models:
        try:
            engine.warm_up((model,))
        except EngineError as error:
            print(f"[jab] {error}", flush=True)  # el modelo informa su error al usarse
    for event in engine.load_events[first:]:
        rss = "" if event.rss_mib is None else f", RSS {event.rss_mib:.0f} MiB"
        print(f"[jab] carga {event.step}: {event.seconds:.2f} s{rss}", flush=True)


@st.cache_resource(show_spinner=False)
def get_engine() -> SearchEngine:
    """Un único motor por proceso, cargado una sola vez y compartido por todas las sesiones.

    Los clásicos (unos 5 s) se cargan antes de mostrar la interfaz. MiniLM (20 a 60 s,
    sobre todo importar torch) termina en segundo plano; si alguien busca con
    Embeddings antes, esa búsqueda espera a que acabe.
    """
    if "JAB_ARTIFACTS_DIR" in os.environ:
        artifacts_dir = Path(os.environ["JAB_ARTIFACTS_DIR"])
    else:
        artifacts_dir = ROOT / "artifacts"
        try:
            artifacts_dir = ensure_artifacts(artifacts_dir, ROOT / "deploy" / "artifacts.json")
        except Exception as error:  # noqa: BLE001 - el motor presenta artefactos ausentes
            print(f"[jab] no se pudieron preparar los artefactos: {error}", flush=True)
    engine = SearchEngine(artifacts_dir)
    _warm(engine, CLASSICAL_MODELS)
    threading.Thread(
        target=_warm, args=(engine, ("dense",)), name="jab-dense", daemon=True
    ).start()
    return engine


@st.cache_resource(show_spinner=False)
def _engine_status() -> dict[str, bool]:
    """Estado compartido entre ejecuciones: la pantalla de carga solo se pinta una vez."""
    return {"ready": False}


@st.cache_resource(show_spinner=False)
def get_suggestions() -> tuple[str, ...]:
    """Consultas reales de FiQA (train) para el autocompletado; vacío si no cargan."""
    try:
        texts = {" ".join(text.split()) for text in load_queries("train").values()}
    except Exception:  # noqa: BLE001 - sin sugerencias la búsqueda sigue funcionando
        return ()
    return tuple(sorted(text for text in texts if 3 <= len(text) <= 140))


def main() -> None:
    mark = Path(__file__).parent / "static" / "jab-mark.svg"
    st.set_page_config(page_title="jab · búsqueda en FiQA", page_icon=str(mark), layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    status = _engine_status()
    if status["ready"]:
        engine = get_engine()  # ya en caché: sin pantalla de carga en cada interacción
    else:
        loading = st.empty()
        logo = (Path(__file__).parent / "static" / "jab-wordmark.svg").read_text(encoding="utf-8")
        loading.markdown(
            f'<div class="jab-loading" role="status" aria-live="polite">'
            f'{logo}<div class="jab-loading-bar" aria-hidden="true"></div>'
            '<p>Preparando el buscador… la primera visita puede descargar los artefactos (~250 MB) y cargar los modelos.</p></div>',
            unsafe_allow_html=True,
        )
        try:
            engine = get_engine()
        finally:
            loading.empty()
        status["ready"] = True

    # Estado que el navegador envió en la interacción anterior.
    state = st.session_state.get("jab") or {}
    request = parse_request(state.get("request"))
    cache = st.session_state.get("jab_response")
    if cache and request.q and cache.get("key") == request.key and "error" not in cache:
        cache = {**cache, "seq": request.seq}  # misma búsqueda (p. ej. solo cambió la vista)
    else:
        cache = respond(engine, request)
    st.session_state["jab_response"] = cache

    jab_component(get_suggestions())(
        key="jab",
        data={
            "initial": {"q": request.q, "model": request.model, "k": request.k,
                        "stem": request.stem, "view": request.view, "seq": request.seq},
            "response": cache,
        },
        default={"request": None},
        on_request_change=lambda: None,
    )

    if request.view != "buscar":
        context = AppContext(
            engine=engine, query=request.q, model=request.model, k=request.k,
            stem=request.stem,
            reports_dir=Path(os.environ.get("JAB_REPORTS_DIR", ROOT / "reports")),
        )
        with st.container(key="jab_views"):
            (comparison if request.view == "comparar" else metrics).render(context)


main()
