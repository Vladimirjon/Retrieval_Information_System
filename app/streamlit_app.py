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
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # permite `from app...` al lanzar con streamlit run
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.context import AppContext
from app.ui import jab_component, parse_request, respond
from app.views import comparison, metrics
from ris.data import load_queries
from ris.engine import MODELS, EngineError, SearchEngine

PAGE_CSS = """
<style>
header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.block-container { max-width: none; padding: 0 0 4rem; }
.stApp { font-family: "Schibsted Grotesk", ui-sans-serif, system-ui, sans-serif; }
.st-key-jab_views { width: min(720px, 100% - 32px); margin: 0 auto; }
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap">
"""


@st.cache_resource(show_spinner="Preparando jab…")
def get_engine() -> SearchEngine:
    """Un único motor por proceso, con los cuatro modelos y el encoder ya cargados."""
    engine = SearchEngine(os.environ.get("JAB_ARTIFACTS_DIR", ROOT / "artifacts"))
    for model in MODELS:
        try:
            engine.warm_up((model,))
        except EngineError as error:
            print(f"[jab] {error}", flush=True)  # el modelo informa su error al usarse
    for event in engine.load_events:
        rss = "" if event.rss_mib is None else f", RSS {event.rss_mib:.0f} MiB"
        print(f"[jab] carga {event.step}: {event.seconds:.2f} s{rss}", flush=True)
    return engine


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
    engine = get_engine()

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
