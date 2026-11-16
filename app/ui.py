"""Puente entre la interfaz de jab (app/frontend/) y el motor.

El navegador envía un único estado {q, model, k, stem, view, seq}; cada cambio sube
`seq`, así ningún cambio se pierde aunque llegue durante otra ejecución. Python
responde con el `seq` atendido y una `key` de la búsqueda, para que la interfaz
descarte respuestas viejas y no repinte al cambiar solo de vista. Todo lo que
llega del navegador se trata como no confiable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from ris.engine import MODELS, EngineError, SearchEngine

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


def respond(engine: SearchEngine, request: Request) -> dict | None:
    """Respuesta para la interfaz; None si no hay consulta."""
    if not request.q:
        return None
    try:
        response = engine.search(request.model, request.q, request.k, stem=request.stem)
    except EngineError as error:
        return {"seq": request.seq, "key": request.key, "error": str(error)}
    return {
        "seq": request.seq,
        "key": request.key,
        "model": response.model,
        "stem": response.stem,
        "elapsed_ms": round(response.elapsed_ms, 1),
        "hits": [
            {"rank": h.rank, "doc_id": h.doc_id, "text": h.text, "score": h.score}
            for h in response.hits
        ],
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
