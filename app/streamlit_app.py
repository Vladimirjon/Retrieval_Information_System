"""jab: buscador sobre FiQA con Jaccard, TF-IDF, BM25 y embeddings (issue #13).

Ejecutar desde la raíz del repositorio:
    python -m streamlit run app/streamlit_app.py

Rutas: / (inicio), /search, /comparison y /metrics; las opciones de la búsqueda van en
los parámetros de la URL (ver app/ui.py).

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
from app.ui import (
    VOCABULARY_SIZE,
    jab_component,
    page_for,
    parse_request,
    resolve_state,
    respond,
    url_params,
)
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
/* Streamlit atenúa la página mientras vuelve a ejecutar el script en cada interacción;
   jab ya muestra su propio estado de carga, así que se anula ese fundido gris. */
[data-testid="stElementContainer"][data-stale="true"] { opacity: 1 !important; transition: none !important; }
.jab-loading { min-height: 85vh; display: grid; place-content: center; justify-items: center; gap: 24px; color: #201A1F; text-align: center; }
.jab-loading svg { width: 120px; height: auto; }
.jab-loading .dot { fill: #7A2E5E; }
.jab-loading-bar { width: min(440px, 80vw); height: 56px; border-radius: 28px; background: #F3F1F2; animation: jab-load 1.4s ease-in-out infinite alternate; }
.jab-loading p { margin: 0; max-width: 36ch; color: #5E5560; font-size: 14px; }
@keyframes jab-load { from { opacity: 1; transform: scaleX(1); } to { opacity: .65; transform: scaleX(.98); } }
@media (prefers-reduced-motion: reduce) { .jab-loading-bar { animation: none; } }
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


@st.cache_resource(show_spinner=False)
def get_vocabulary(_engine: SearchEngine) -> tuple[str, ...]:
    """Palabras frecuentes de FiQA para completar en gris; vacío si el índice no carga."""
    try:
        return _engine.top_terms(VOCABULARY_SIZE)
    except Exception:  # noqa: BLE001 - sin vocabulario se completa solo con sugerencias
        return ()


# Cada ruta es una página de Streamlit sin contenido propio: la interfaz se dibuja en
# main() para que el componente siga montado (y con el foco) al cambiar de ruta.
def _home() -> None: ...
def _search() -> None: ...
def _comparison() -> None: ...
def _metrics() -> None: ...


def _pages() -> dict[str, st.Page]:
    return {
        "home": st.Page(_home, title="jab · búsqueda en FiQA", default=True),
        "search": st.Page(_search, title="jab · Buscar", url_path="search"),
        "comparison": st.Page(_comparison, title="jab · Comparar", url_path="comparison"),
        "metrics": st.Page(_metrics, title="jab · Métricas", url_path="metrics"),
    }


def _current_request(page: str):
    """Estado vigente y si llegó desde la URL (nav) en vez de desde la interfaz.

    Gana lo último que ocurrió: una acción del navegador sube su seq; un cambio de la
    URL que no hizo la app (recargar, atrás, adelante, un enlace) cambia la ruta o los
    parámetros sin cambiar el seq, y entonces se adopta la URL con un seq nuevo.

    Solo cuenta como cambio una URL distinta de la que se vio en la ejecución anterior:
    una ejecución que el navegador pidió antes de procesar un `switch_page` todavía
    trae la URL vieja, y adoptarla devolvería la interfaz a un estado ya abandonado.
    """
    sent = (st.session_state.get("jab") or {}).get("request")
    current, nav, seen = resolve_state(
        st.session_state.get("jab_current"),
        None if sent is None else parse_request(sent),
        page, st.query_params.to_dict(),
        st.session_state.get("jab_seen_url"), st.session_state.get("jab_nav", 0),
    )
    st.session_state["jab_seen_url"] = seen
    st.session_state["jab_current"] = current
    st.session_state["jab_nav"] = nav
    return current, nav


def main() -> None:
    mark = Path(__file__).parent / "static" / "jab-mark.svg"
    st.set_page_config(page_title="jab · búsqueda en FiQA", page_icon=str(mark), layout="wide")
    pages = _pages()
    active = st.navigation(list(pages.values()), position="hidden")
    page = next((key for key, value in pages.items() if value.url_path == active.url_path),
                "home")
    st.set_page_config(page_title=active.title)  # la pestaña nombra la vista
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
            get_vocabulary(engine)  # unos segundos la primera vez: dentro de la pantalla de carga
        finally:
            loading.empty()
        status["ready"] = True

    request, nav = _current_request(page)
    cache = st.session_state.get("jab_response")
    if cache and request.q and cache.get("key") == request.key and "error" not in cache:
        cache = {**cache, "seq": request.seq}  # misma búsqueda (p. ej. solo cambió la vista)
    else:
        cache = respond(engine, request)
    st.session_state["jab_response"] = cache

    jab_component(get_suggestions(), get_vocabulary(engine))(
        key="jab",
        data={
            "initial": {"q": request.q, "model": request.model, "k": request.k,
                        "stem": request.stem, "graph": request.graph, "view": request.view,
                        "route": request.route, "seq": request.seq, "nav": nav},
            "response": cache,
        },
        default={"request": None},
        on_request_change=lambda: None,
    )

    # La URL sigue al estado: ruta según la vista y parámetros con las opciones.
    target, params = page_for(request), url_params(request)
    if target != page:
        st.switch_page(pages[target], query_params=params)
    if st.query_params.to_dict() != params:
        st.query_params.from_dict(params)

    if request.view != "buscar":
        context = AppContext(
            engine=engine, query=request.q, model=request.model, k=request.k,
            stem=request.stem,
            reports_dir=Path(os.environ.get("JAB_REPORTS_DIR", ROOT / "reports")),
        )
        with st.container(key="jab_views"):
            (comparison if request.view == "comparar" else metrics).render(context)


main()
