"""Puente entre la interfaz de jab (app/frontend/) y el motor.

El navegador envía un único estado {q, model, k, stem, graph, view, seq}; cada cambio sube
`seq`, así ningún cambio se pierde aunque llegue durante otra ejecución. Python
responde con el `seq` atendido y una `key` de la búsqueda, para que la interfaz
descarte respuestas viejas y no repinte al cambiar solo de vista. Todo lo que
llega del navegador se trata como no confiable.

El mismo estado vive en la URL: la vista es la ruta (/search, /comparison, /metrics;
"/" es el inicio) y las opciones van en los parámetros q, model, results, stemming y
graph. Así una búsqueda se puede recargar, compartir o recorrer con atrás y adelante.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import streamlit as st
import streamlit.components.v2 as components

from ris.engine import MODELS, EngineError, SearchEngine
from ris.preprocessing import preprocess

FRONTEND = Path(__file__).parent / "frontend"
WORDMARK = Path(__file__).parent / "static" / "jab-wordmark.svg"
TOP_K = (5, 10, 20, 50)
VIEWS = ("buscar", "comparar", "metricas")
MAX_QUERY = 300
GRAPH_CONTEXT = 50  # el grafo sitúa también, tenues, los resultados hasta este puesto
VOCABULARY_SIZE = 12_000  # palabras frecuentes para completar en gris la última palabra

# Rutas y parámetros de la URL. El modelo denso se llama "embeddings" en la URL.
PAGES = {"home": "buscar", "search": "buscar", "comparison": "comparar", "metrics": "metricas"}
URL_MODELS = {"jaccard": "jaccard", "tfidf": "tfidf", "bm25": "bm25", "dense": "embeddings"}
_MODEL_FROM_URL = {value: key for key, value in URL_MODELS.items()}


@dataclass(frozen=True)
class Request:
    """Petición validada del navegador."""

    q: str
    model: str
    k: int
    stem: bool
    view: str
    seq: int
    graph: bool = False
    # En Buscar, "/" o "/search". Escribir no cambia la ruta: cambiar de página vuelve a
    # montar el componente y se perdería el foco. Solo una acción explícita (Enter, una
    # sugerencia, un ejemplo) pasa a /search, y el logo vuelve al inicio.
    route: str = "home"

    @property
    def key(self) -> str:
        """Identifica la búsqueda (sin la vista ni el seq); el grafo cambia la respuesta."""
        return f"{self.model}|{self.k}|{int(self.stem)}|{int(self.graph)}|{self.q}"


DEFAULT_REQUEST = Request(q="", model="bm25", k=10, stem=False, view="buscar", seq=0)


def parse_request(raw: object) -> Request:
    """Normaliza la petición; cualquier valor inválido vuelve al predeterminado."""
    if not isinstance(raw, dict):
        return DEFAULT_REQUEST
    q = raw.get("q")
    model = raw.get("model")
    k = raw.get("k")
    seq = raw.get("seq")
    return Request(
        q=" ".join(q.split())[:MAX_QUERY] if isinstance(q, str) else "",
        model=model if model in MODELS else DEFAULT_REQUEST.model,
        k=k if isinstance(k, int) and not isinstance(k, bool) and k in TOP_K else 10,
        stem=raw.get("stem") is True and model != "dense",
        view=parse_view(raw.get("view")),
        seq=seq if isinstance(seq, int) and not isinstance(seq, bool) and seq >= 0 else 0,
        graph=raw.get("graph") is True,
        route=raw.get("route") if raw.get("route") in ("home", "search") else "home",
    )


def parse_view(raw: object) -> str:
    return raw if raw in VIEWS else "buscar"


def same_state(a: Request, b: Request) -> bool:
    """Misma búsqueda, vista y ruta, sin mirar el seq."""
    return a.key == b.key and a.view == b.view and page_for(a) == page_for(b)


def _flag(raw: object) -> bool:
    return isinstance(raw, str) and raw.strip().lower() == "true"


def request_from_url(page: str, params: Mapping[str, object], seq: int = 0) -> Request:
    """Estado a partir de la ruta y los parámetros; lo inválido vuelve al predeterminado."""
    model = params.get("model")
    model = _MODEL_FROM_URL.get(model.strip().lower()) if isinstance(model, str) else None
    results = params.get("results")
    # Se compara con las opciones antes de convertir: "²".isdigit() es True e int() fallaría.
    k = next((option for option in TOP_K if results == str(option)), 10)
    return parse_request({
        "q": params.get("q"), "model": model, "k": k, "stem": _flag(params.get("stemming")),
        "graph": _flag(params.get("graph")), "view": PAGES.get(page, "buscar"), "seq": seq,
        "route": "search" if page == "search" else "home",
    })


def resolve_state(
    current: Request | None, sent: Request | None, page: str, params: Mapping[str, object],
    seen: tuple | None, nav: int,
) -> tuple[Request, int, tuple]:
    """Estado vigente entre lo que envió el navegador y lo que dice la URL.

    Devuelve (estado, nav, url_vista). Gana lo último que ocurrió: una acción de la
    interfaz llega con un seq mayor; un cambio de la URL que no hizo la app (recargar,
    atrás, adelante, un enlace) llega sin cambiar el seq y se adopta con un seq nuevo y
    `nav` + 1, para que la interfaz lo aplique. Solo cuenta como cambio una URL distinta
    de la vista en la ejecución anterior (`seen`): una ejecución que el navegador pidió
    antes de procesar un `switch_page` todavía trae la URL vieja, y adoptarla devolvería
    la interfaz a un estado ya abandonado.
    """
    from_url = request_from_url(page, params)
    url_now = (page, from_url.key, from_url.view, from_url.route)
    if current is None:
        return from_url, nav + 1, url_now
    if sent is not None and sent.seq > current.seq:
        return sent, nav, url_now
    if not same_state(from_url, current) and url_now != seen:
        return replace(from_url, seq=current.seq + 1), nav + 1, url_now
    return current, nav, url_now


def page_for(request: Request) -> str:
    """Ruta de un estado. En Buscar la elige la interfaz (`route`), no la consulta."""
    if request.view == "comparar":
        return "comparison"
    if request.view == "metricas":
        return "metrics"
    return request.route


def url_params(request: Request) -> dict[str, str]:
    """Parámetros de la URL. En el inicio, solo los que difieren de lo predeterminado."""
    params = {"q": request.q, "model": URL_MODELS[request.model], "results": str(request.k),
              "stemming": "true" if request.stem else "false",
              "graph": "true" if request.graph else "false"}
    if request.model == "dense":
        del params["stemming"]  # Porter no se aplica a Embeddings
    if not request.q:
        del params["q"]
    if page_for(request) == "home":
        params = {key: value for key, value in params.items()
                  if _HOME_DEFAULTS.get(key) != value}
    return params


_HOME_DEFAULTS = {"model": "bm25", "results": "10", "stemming": "false", "graph": "false"}


_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)\s]*\)")
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!>~|^])")
_MD_LINE_MARK = re.compile(r"^[ \t]*(?:#{1,6}|>+)[ \t]*", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"\*\*|__|~~|(?<!\w)\^+")
_MD_ITALIC = re.compile(r"(?<![\w*])\*(?=\w)([^*\n]*?\w)\*(?![\w*])")
_DISPLAY_WORD = re.compile(r"\b\w+(?:[-'’.]\w+)*\b", re.UNICODE)


def clean_text(text: str) -> str:
    """Texto legible a partir del Markdown de Reddit/StackExchange que trae FiQA.

    Quita marcas (negrita, títulos, citas, superíndices, escapes), deja solo la
    etiqueta de los enlaces y decodifica entidades como `&amp;`. El resultado se
    sigue tratando como texto plano: el navegador lo escapa al pintarlo.
    """
    text = html.unescape(text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_LINE_MARK.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    text = _MD_ITALIC.sub(r"\1", text)
    return _MD_ESCAPE.sub(r"\1", text).strip()


@lru_cache(maxsize=50_000)
def _word_terms(word: str, stem: bool) -> tuple[str, ...]:
    return tuple(preprocess(word, stem=stem))


def highlight_spans(text: str, matched: tuple[str, ...] | list[str], *, stem: bool) -> list[list[int]]:
    """Tramos UTF-16 de palabras visibles que producen un término coincidente."""
    terms = set(matched)
    spans = []
    cursor = units = 0
    for word in _DISPLAY_WORD.finditer(text):
        start = units + len(text[cursor:word.start()].encode("utf-16-le")) // 2
        end = start + len(word.group().encode("utf-16-le")) // 2
        if terms.intersection(_word_terms(word.group().lower(), stem)):
            spans.append([start, end])
        cursor, units = word.end(), end
    return spans


def highlight_html(text: str, matched: tuple[str, ...] | list[str], *, stem: bool) -> str:
    """Texto escapado con `<mark>` en las mismas palabras que marca `highlight_spans`.

    Trabaja con índices de `str` de Python (no UTF-16) y escapa todo lo demás, así el
    resultado se puede insertar tal cual en HTML.
    """
    terms = set(matched)
    if not terms:
        return html.escape(text)
    parts = []
    cursor = 0
    for word in _DISPLAY_WORD.finditer(text):
        if terms.intersection(_word_terms(word.group().lower(), stem)):
            parts.append(html.escape(text[cursor:word.start()]))
            parts.append(f"<mark>{html.escape(word.group())}</mark>")
            cursor = word.end()
    parts.append(html.escape(text[cursor:]))
    return "".join(parts)


def graph_payload(engine: SearchEngine, query: str, hits: tuple, shown: int) -> dict:
    """Grafo 3D de la consulta para la interfaz, o el motivo legible si no se pudo.

    `nodes[i]` es [x, y, z, coseno con la consulta] del resultado i del ranking; la
    consulta está en el origen. Los `shown` primeros son los resultados visibles y los
    únicos unidos por `edges` ([i, j, coseno]); el resto llega en `context` para
    dibujarlos tenues alrededor.
    """
    started = perf_counter()
    try:
        graph = engine.query_graph(query, [hit.doc_id for hit in hits], linked=shown)
    except EngineError as error:
        return {"error": str(error)}
    return {
        "nodes": [[*(round(value, 4) for value in node.position), round(node.similarity, 4)]
                  for node in graph.nodes],
        "edges": [[i, j, round(similarity, 4)] for i, j, similarity in graph.edges],
        "shown": shown,
        "context": [{"rank": hit.rank, "doc_id": hit.doc_id, "text": clean_text(hit.text)[:240]}
                    for hit in hits[shown:]],
        "variance": round(graph.explained_variance, 4),
        "elapsed_ms": round((perf_counter() - started) * 1000, 1),
    }


def suggestion_for(engine: SearchEngine, query: str) -> str | None:
    """«Quizás quisiste decir»: la consulta corregida, o None. Nunca rompe la búsqueda."""
    try:
        return engine.correct_query(query)
    except Exception:  # noqa: BLE001 - la sugerencia es opcional
        return None


def respond(engine: SearchEngine, request: Request) -> dict | None:
    """Respuesta para la interfaz; None si no hay consulta.

    Para los modelos clásicos incluye los términos con los que se buscó (`terms`) y,
    en cada resultado, cuáles aparecen en el documento (`matched`): así se entiende
    por qué salió y qué palabras descartó el preprocesamiento. Con `request.graph`
    añade `graph` (ver `graph_payload`), calculado para esta búsqueda con los
    `GRAPH_CONTEXT` primeros; si el grafo falla, los resultados se devuelven igual.
    `suggestion` es la consulta corregida si alguna palabra parece mal escrita.
    """
    if not request.q:
        return None
    suggestion = suggestion_for(engine, request.q)
    try:
        wanted = max(request.k, GRAPH_CONTEXT) if request.graph else request.k
        response = engine.search(request.model, request.q, wanted,
                                 stem=request.stem, skip_empty=True)
        shown = response.hits[:request.k]
        classical = request.model != "dense"
        terms = engine.query_terms(request.q, stem=request.stem) if classical else ()
        hits = []
        for hit in shown:
            text = clean_text(hit.text)
            matched = (engine.matched_terms(hit.doc_id, terms, stem=request.stem)
                       if terms else ())
            hits.append({
                "rank": hit.rank, "doc_id": hit.doc_id, "text": text,
                "score": hit.score, "matched": list(matched),
                "spans": highlight_spans(text, matched, stem=request.stem) if matched else [],
            })
    except EngineError as error:
        return {"seq": request.seq, "key": request.key, "error": str(error),
                "suggestion": suggestion}
    payload = {
        "seq": request.seq,
        "key": request.key,
        "model": response.model,
        "stem": response.stem,
        "elapsed_ms": round(response.elapsed_ms, 1),
        "terms": list(terms),
        "hits": hits,
        "suggestion": suggestion,
    }
    if request.graph and hits:
        payload["graph"] = graph_payload(engine, request.q, response.hits, len(hits))
    return payload


def build_assets(
    suggestions: tuple[str, ...], vocabulary: tuple[str, ...] = ()
) -> tuple[str, str, str]:
    """HTML (con el logo en línea), CSS y JS del componente."""
    logo = WORDMARK.read_text(encoding="utf-8")
    html = (FRONTEND / "index.html").read_text(encoding="utf-8").replace("{{LOGO}}", logo)
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    js = (FRONTEND / "app.js").read_text(encoding="utf-8").replace(
        "__SUGGESTIONS__", json.dumps(list(suggestions), ensure_ascii=False)
    ).replace("__VOCABULARY__", json.dumps(list(vocabulary), ensure_ascii=False))
    return html, css, js


@st.cache_resource(show_spinner=False)
def jab_component(suggestions: tuple[str, ...], vocabulary: tuple[str, ...] = ()):
    """Registra el componente una vez por proceso. Sin aislamiento de estilos: el
    CSS está acotado a `.jab` y así el logo SVG y las transiciones no se sanean."""
    html, css, js = build_assets(suggestions, vocabulary)
    return components.component("jab_ui", html=html, css=css, js=js, isolate_styles=False)
