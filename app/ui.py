"""Puente entre la interfaz de jab (app/frontend/) y el motor.

El navegador envía un único estado {q, model, k, stem, view, seq}; cada cambio sube
`seq`, así ningún cambio se pierde aunque llegue durante otra ejecución. Python
responde con el `seq` atendido y una `key` de la búsqueda, para que la interfaz
descarte respuestas viejas y no repinte al cambiar solo de vista. Todo lo que
llega del navegador se trata como no confiable.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from ris.engine import MODELS, EngineError, SearchEngine
from ris.preprocessing import preprocess

FRONTEND = Path(__file__).parent / "frontend"
WORDMARK = Path(__file__).parent / "static" / "jab-wordmark.svg"
TOP_K = (5, 10, 20, 50)
VIEWS = ("buscar", "comparar", "metricas")
MAX_QUERY = 300


@dataclass(frozen=True)
class Request:
    """Petición validada del navegador."""

    q: str
    model: str
    k: int
    stem: bool
    view: str
    seq: int

    @property
    def key(self) -> str:
        """Identifica la búsqueda (sin la vista ni el seq)."""
        return f"{self.model}|{self.k}|{int(self.stem)}|{self.q}"


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
    )


def parse_view(raw: object) -> str:
    return raw if raw in VIEWS else "buscar"


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


def respond(engine: SearchEngine, request: Request) -> dict | None:
    """Respuesta para la interfaz; None si no hay consulta.

    Para los modelos clásicos incluye los términos con los que se buscó (`terms`) y,
    en cada resultado, cuáles aparecen en el documento (`matched`): así se entiende
    por qué salió y qué palabras descartó el preprocesamiento.
    """
    if not request.q:
        return None
    try:
        response = engine.search(request.model, request.q, request.k,
                                 stem=request.stem, skip_empty=True)
        classical = request.model != "dense"
        terms = engine.query_terms(request.q, stem=request.stem) if classical else ()
        hits = []
        for hit in response.hits:
            text = clean_text(hit.text)
            matched = (engine.matched_terms(hit.doc_id, terms, stem=request.stem)
                       if terms else ())
            hits.append({
                "rank": hit.rank, "doc_id": hit.doc_id, "text": text,
                "score": hit.score, "matched": list(matched),
                "spans": highlight_spans(text, matched, stem=request.stem) if matched else [],
            })
    except EngineError as error:
        return {"seq": request.seq, "key": request.key, "error": str(error)}
    return {
        "seq": request.seq,
        "key": request.key,
        "model": response.model,
        "stem": response.stem,
        "elapsed_ms": round(response.elapsed_ms, 1),
        "terms": list(terms),
        "hits": hits,
    }


def build_assets(suggestions: tuple[str, ...]) -> tuple[str, str, str]:
    """HTML (con el logo en línea), CSS y JS del componente."""
    logo = WORDMARK.read_text(encoding="utf-8")
    html = (FRONTEND / "index.html").read_text(encoding="utf-8").replace("{{LOGO}}", logo)
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    js = (FRONTEND / "app.js").read_text(encoding="utf-8").replace(
        "__SUGGESTIONS__", json.dumps(list(suggestions), ensure_ascii=False)
    )
    return html, css, js


@st.cache_resource(show_spinner=False)
def jab_component(suggestions: tuple[str, ...]):
    """Registra el componente una vez por proceso. Sin aislamiento de estilos: el
    CSS está acotado a `.jab` y así el logo SVG y las transiciones no se sanean."""
    html, css, js = build_assets(suggestions)
    return components.component("jab_ui", html=html, css=css, js=js, isolate_styles=False)
