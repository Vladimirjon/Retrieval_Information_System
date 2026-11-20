"""Vista de comparación lado a lado (#14): los cuatro rankings de una misma consulta.

Usa `context.search(modelo, skip_empty=True)` por cada modelo de `ris.engine.MODELS`,
igual que la vista Buscar, y `app.ui.clean_text` para el Markdown de Reddit. El
solapamiento está en `app.overlap`. Colores y medidas salen de los tokens de
`app/frontend/styles.css` (`:root`); cada uno lleva su valor de respaldo.

Los documentos compartidos se enlazan con chips y etiquetas de posición.
Como en Buscar, los modelos clásicos marcan con `<mark>` las palabras de la
consulta que coinciden; Embeddings no.

Sin consulta la vista no dibuja nada: el componente de la barra muestra el estado
vacío con los mismos ejemplos que el inicio de Buscar.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass

import streamlit as st

from app.context import AppContext
from app.overlap import appearances
from app.ui import clean_text, highlight_html
from ris.engine import MODELS, EngineError, Hit

MODEL_NAMES = {"jaccard": "Jaccard", "tfidf": "TF-IDF", "bm25": "BM25", "dense": "Embeddings"}
MODEL_KINDS = {
    "jaccard": "Binario", "tfidf": "Vectorial", "bm25": "Probabilístico", "dense": "Semántico",
}
SNIPPET_CHARS = 200
# Cada cuánto se comprueba si la búsqueda de un modelo terminó. Mientras espera, la
# vista actualiza su indicador de carga: cada actualización es un punto en el que
# Streamlit atiende una petición nueva (p. ej. volver a Buscar) sin esperar al modelo.
WAIT_STEP_S = 0.25
_SEARCHES = ThreadPoolExecutor(max_workers=len(MODELS), thread_name_prefix="jab-compare")

HELP_TEXT = (
    "Pasa el puntero (o toca) un documento compartido para ver dónde queda en los otros modelos. "
    "«Ver en…» te lleva a la siguiente columna donde aparece. "
    "Los scores de cada modelo no se comparan entre sí."
)

CSS = """
<style>
.st-key-jab_views:has(.st-key-jab_view_comparar) { width: min(1160px, 100% - 32px); }
/* Con la vista ya pintada sobra la línea «Buscando…» del componente (styles.css). */
.jab .cmp-pending { display: none !important; }
.cmp {
  --cmp-ink: var(--ink, #201A1F); --cmp-ink-2: var(--ink-2, #5E5560); --cmp-ink-3: var(--ink-3, #756B77);
  --cmp-line: var(--line, #E6E1E4); --cmp-surface: var(--surface, #F3F1F2);
  --cmp-accent: var(--accent-ink, #7A2E5E);
  --cmp-ochre: color-mix(in srgb, var(--ochre, #A86A12) 72%, #000); --cmp-ochre-soft: var(--ochre-soft, #F8EBD2);
  position: relative; color: var(--cmp-ink); font-size: .9375rem; line-height: 1.5;
}
.cmp, .cmp *, .cmp *::before, .cmp *::after { box-sizing: border-box; }
.cmp p { margin: 0; }
.cmp :focus-visible { outline: 2px solid var(--cmp-accent); outline-offset: 2px; border-radius: 6px; }
.cmp ::selection { background: var(--accent-soft, #F3E7EF); color: var(--cmp-ink); }
.cmp-sr { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0; }

/* Contexto: «Top K» y el interruptor de coincidencias, alineados con la rejilla. */
.cmp-band { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
  gap: 12px; margin: 8px 0 24px; }
.cmp-view { display: inline-flex; gap: 2px; padding: 3px; border-radius: 999px; background: var(--cmp-surface); }
.cmp-seg { border: 0; border-radius: 999px; background: transparent; color: var(--cmp-ink-2); font: inherit;
  font-size: .8125rem; padding: 4px 12px; cursor: pointer; white-space: nowrap;
  transition: background-color 120ms ease-out, color 120ms ease-out; }
.cmp-seg:hover { color: var(--cmp-ink); }
.cmp-seg[aria-pressed="true"] { background: #fff; color: var(--cmp-ink); font-weight: 600;
  box-shadow: 0 1px 2px rgb(32 26 31 / .12); }
/* El mismo documento en las otras columnas se marca con un fondo cálido; nada se atenúa ni se oculta. */
.cmp-doc { position: relative; isolation: isolate; }
.cmp-doc::before { content: ""; position: absolute; z-index: -1; inset: 8px -10px; border-radius: 12px;
  background: var(--cmp-ochre-soft); opacity: 0; transition: opacity 140ms ease-out; }
.cmp-doc[data-active="true"]::before { opacity: 1; }
.cmp-doc[data-active="true"] .cmp-rank { background: #fff; color: var(--cmp-ink); }
.cmp[data-only-shared="true"] .cmp-doc[data-coverage="1"] { display: none; }
.cmp-foot { margin-top: var(--s2, 8px); }
.cmp-jump { display: inline-flex; align-items: center; gap: 5px; padding: 0; border: 0; background: none;
  color: var(--cmp-ink-2); font: inherit; font-size: .8125rem; font-weight: 600; cursor: pointer; }
.cmp-jump svg { width: 13px; height: 13px; fill: none; stroke: currentColor; stroke-width: 1.8;
  stroke-linecap: round; stroke-linejoin: round; transition: transform 140ms ease-out; }
.cmp-jump:hover { color: var(--cmp-ink); }
.cmp-jump:hover svg { transform: translateX(2px); }
.cmp-jump:focus-visible { outline: 2px solid var(--cmp-accent); outline-offset: 3px; border-radius: 4px; }
.cmp-pills { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2, 8px); }
.cmp-pill { display: inline-flex; align-items: center; height: 26px; padding: 0 10px; border-radius: 999px;
  background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .8125rem; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.cmp-pill[data-kind="porter"] { background: var(--cmp-ochre-soft); color: var(--cmp-ochre); }
.cmp-help { display: inline-grid; place-items: center; width: 26px; height: 26px; padding: 0;
  border: 1px solid var(--cmp-line); border-radius: 50%; background: transparent; color: var(--cmp-ink-2);
  font: inherit; font-size: .8125rem; font-weight: 700; line-height: 1; cursor: help;
  transition: background-color 120ms ease-out, color 120ms ease-out; }
.cmp-help:hover, .cmp-help:focus-visible { background: var(--cmp-surface); color: var(--cmp-ink); }
.cmp-help:focus-visible { border-radius: 50%; }
/* Columnas */
.cmp-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--s4, 16px); align-items: start; }
.cmp-col { min-width: 0; overflow: hidden; background: #fff; border: 1px solid var(--cmp-line);
  border-radius: var(--r-menu, 18px); corner-shape: squircle; }
.cmp-col-head { display: flex; justify-content: space-between; align-items: flex-start; gap: var(--s2, 8px);
  padding: var(--s3, 12px) var(--s4, 16px); background: var(--cmp-surface); border-bottom: 1px solid var(--cmp-line); }
.cmp-model { font-size: 1.0625rem; font-weight: 650; line-height: 1.3; }
.cmp-kind { color: var(--cmp-ink-2); font-size: .75rem; }
.cmp-short { margin-left: var(--s2, 8px); color: var(--cmp-ochre); font-weight: 600; font-variant-numeric: tabular-nums; }
.cmp-time { flex: none; display: inline-flex; align-items: center; gap: 5px; height: 24px; padding: 0 9px 0 7px;
  border-radius: 999px; background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .75rem; font-weight: 600;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.cmp-time svg { width: 12px; height: 12px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; }
.cmp-empty { margin: var(--s4, 16px); padding: var(--s3, 12px); border-radius: var(--r-item, 10px);
  background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .875rem; }
.cmp .cmp-list { list-style: none; margin: 0 0 -1px; padding: 0 var(--s4, 16px); }
.cmp .cmp-list > li { margin: 0; }
.cmp-doc { display: grid; grid-template-columns: 30px minmax(0, 1fr); gap: var(--s3, 12px);
  padding: 20px 0; border-bottom: 1px solid var(--cmp-line); }
.cmp-rank { display: grid; place-items: center; width: 30px; height: 30px; border-radius: 8px;
  font-size: .875rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.cmp-rank { background: var(--cmp-surface); color: var(--cmp-ink-2); }
.cmp-doc-top { display: flex; align-items: baseline; gap: var(--s2, 8px); min-height: 30px; padding-top: 5px;
  color: var(--cmp-ink-3); font-size: .75rem; flex-wrap: wrap; overflow-wrap: anywhere; }
.cmp-score { margin-left: auto; font-variant-numeric: tabular-nums; }
.cmp-snippet { color: var(--cmp-ink); line-height: 1.6; overflow-wrap: anywhere; }
.cmp mark { padding: 0 1px; border-radius: 3px; background: var(--accent-soft, #F3E7EF); color: var(--cmp-accent);
  box-decoration-break: clone; -webkit-box-decoration-break: clone; }
.cmp-doc details { display: flex; flex-direction: column; margin-top: var(--s2, 8px); }
.cmp-doc summary { order: 1; width: fit-content; margin-top: var(--s1, 4px); color: var(--cmp-accent);
  font-size: .8125rem; font-weight: 600; cursor: pointer; list-style: none; }
.cmp-doc summary::-webkit-details-marker { display: none; }
.cmp-doc summary:hover { text-decoration: underline; text-underline-offset: 3px; }
.cmp-doc details[open] .cmp-more, .cmp-doc details:not([open]) .cmp-less { display: none; }
.cmp-doc:has(details[open]) .cmp-snippet { display: none; }
.cmp-doc details p { max-height: 320px; overflow-y: auto; white-space: pre-line; line-height: 1.6;
  color: var(--cmp-ink); overflow-wrap: anywhere; scrollbar-width: thin; scrollbar-color: var(--cmp-line) transparent; }

/* Aviso sin resultados y carga */
.cmp-notice { margin-top: var(--s4, 16px); padding: var(--s3, 12px) var(--s4, 16px); border-radius: var(--r-item, 10px);
  background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .875rem; }
.cmp-skel { height: 260px; margin-top: 66px; background: var(--cmp-surface); border-color: transparent;
  animation: cmp-pulse 1.2s ease-in-out infinite alternate; }
@keyframes cmp-pulse { from { opacity: 1; } to { opacity: .55; } }

@media (max-width: 1099px) { .cmp-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 719px) {
  .cmp-grid { grid-template-columns: minmax(0, 1fr); }
  .cmp-skel { margin-top: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .cmp-skel { animation: none; } .cmp *, .cmp .cmp-doc, .cmp .cmp-doc::before { transition: none; }
  .cmp-jump:hover svg { transform: none; }
}
</style>
"""

ARROW = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5"/></svg>'
CLOCK = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2"/><path d="M8 4.6V8l2.2 1.4"/></svg>'


@dataclass(frozen=True)
class ModelRun:
    """Resultado de un modelo: sus documentos, el tiempo, o el motivo del fallo."""

    hits: tuple[Hit, ...] = ()
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def ids(self) -> list[str]:
        return [hit.doc_id for hit in self.hits]


def run_models(
    context: AppContext, on_wait: Callable[[str], None] | None = None,
) -> dict[str, ModelRun]:
    """Una búsqueda por modelo, en orden; si uno falla, los demás se muestran igual.

    Cada búsqueda corre en un hilo aparte. Si tarda más de `WAIT_STEP_S` (Embeddings
    todavía cargándose, el índice con Porter la primera vez), se llama a `on_wait`
    con el modelo que falta en cada intervalo, sin bloquear la ejecución.
    """
    runs = {}
    for model in MODELS:
        future = _SEARCHES.submit(context.search, model, skip_empty=True)
        # `wait` no relanza la excepción de la búsqueda: un TimeoutError propio del motor
        # llega a `result()` y se propaga, en vez de confundirse con la espera.
        while not wait([future], timeout=WAIT_STEP_S).done:
            if on_wait is not None:
                on_wait(model)
        try:
            response = future.result()
        except EngineError as error:
            runs[model] = ModelRun(error=str(error))
        else:
            runs[model] = ModelRun(response.hits, response.elapsed_ms)
    return runs


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Primeros `limit` caracteres en una sola línea, cortados en palabra entera."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") + "…"


def format_ms(ms: float) -> str:
    return f"{ms:.1f} ms" if ms < 10 else f"{ms:.0f} ms"


def _doc_html(hit: Hit, found_in: tuple[str, ...], model: str,
              positions: dict[str, dict[str, int]], matched: tuple[str, ...], stem: bool) -> str:
    text = clean_text(hit.text)
    short = snippet(text)
    tip = f"Puesto {hit.rank}"
    others = [other for other in found_in if other != model]
    labels = ""
    if others:
        # La flecha lleva a la siguiente columna (en orden, dando la vuelta) donde sale este documento.
        later = [other for other in others if MODELS.index(other) > MODELS.index(model)]
        target = (later or others)[0]
        also = html.escape(", ".join(f"{MODEL_NAMES[other]} #{positions[other][hit.doc_id]}" for other in others))
        labels = (f'<div class="cmp-foot"><button type="button" class="cmp-jump" data-jump-model="{target}" '
                  f'data-tip="También en {also}">Ver en {html.escape(MODEL_NAMES[target])} '
                  f'#{positions[target][hit.doc_id]}{ARROW}</button></div>')
    full = ""
    if len(" ".join(text.split())) > len(short):
        body = highlight_html(re.sub(r"\n\s*\n+", "\n\n", text), matched, stem=stem)
        full = ('<details><summary><span class="cmp-more">Ver más</span>'
                f'<span class="cmp-less">Ver menos</span></summary><p>{body}</p></details>')
    shown = highlight_html(short, matched, stem=stem) if short else "(documento sin texto)"
    return (
        f'<li class="cmp-doc" data-doc-id="{html.escape(hit.doc_id)}" '
        f'data-model="{model}" data-coverage="{len(found_in)}"{" tabindex=0" if len(found_in) > 1 else ""}>'
        f'<span class="cmp-rank" role="img" aria-label="{html.escape(tip)}" '
        f'data-tip="{html.escape(tip)}">{hit.rank}</span><div>'
        f'<div class="cmp-doc-top"><span>Doc. {html.escape(hit.doc_id)}</span>'
        f'<span class="cmp-score" title="Score propio del modelo">{hit.score:.3f}</span></div>'
        f'<p class="cmp-snippet">{shown}</p>{full}{labels}</div></li>'
    )


def _column_html(model: str, run: ModelRun, seen: dict[str, tuple[str, ...]], k: int, positions: dict[str, dict[str, int]],
                 matched: dict[str, tuple[str, ...]], stem: bool) -> str:
    short = ""
    if run.error:
        body = f'<p class="cmp-empty">{html.escape(run.error)}</p>'
    elif not run.hits:
        body = '<p class="cmp-empty">Sin resultados para esta consulta.</p>'
    else:
        if len(run.hits) < k:
            short = f'<span class="cmp-short">{len(run.hits)} de {k}</span>'
        items = "".join(
            _doc_html(hit, seen[hit.doc_id], model, positions, matched.get(hit.doc_id, ()), stem)
            for hit in run.hits
        )
        body = f'<ol class="cmp-list">{items}</ol>'
    time = "" if run.error else f'<span class="cmp-time">{CLOCK}{format_ms(run.elapsed_ms)}</span>'
    return (
        '<section class="cmp-col">'
        '<div class="cmp-col-head"><div>'
        f'<div class="cmp-model" role="heading" aria-level="3">{html.escape(MODEL_NAMES[model])}</div>'
        f'<div class="cmp-kind">{html.escape(MODEL_KINDS[model])}{short}</div></div>{time}</div>'
        f"{body}</section>"
    )


def help_html() -> str:
    return (f'<button type="button" class="cmp-help" data-tip="{html.escape(HELP_TEXT)}" '
            'aria-label="Cómo leer la comparación">?</button>')


def head_html(k: int, stem: bool, seen: dict[str, tuple[str, ...]]) -> str:
    porter = '<span class="cmp-pill" data-kind="porter">Porter</span>' if stem else ""
    shared = sum(len(models) > 1 for models in seen.values())
    only = ('<div class="cmp-view" role="group" aria-label="Documentos mostrados">'
            '<button type="button" class="cmp-seg" data-only="false" aria-pressed="true">Todos</button>'
            f'<button type="button" class="cmp-seg" data-only="true" aria-pressed="false">Coincidencias ({shared})</button>'
            '</div>' if shared else "")
    return (f'<div class="cmp-band"><div class="cmp-pills"><span class="cmp-pill">Top {k}</span>{porter}'
            f'{help_html()}</div>{only}</div>')


def loading_html(model: str) -> str:
    """Esqueleto mientras falta un modelo: mismo ancho y rejilla que el resultado."""
    skeleton = '<div class="cmp-col cmp-skel" aria-hidden="true"></div>' * len(MODELS)
    return (f'{CSS}<div class="cmp"><p class="cmp-sr" role="status">Buscando, falta '
            f'{html.escape(MODEL_NAMES[model])}</p><div class="cmp-grid">{skeleton}</div></div>')


def matched_by_doc(context: AppContext, runs: dict[str, ModelRun]) -> dict[str, tuple[str, ...]]:
    """Términos de la consulta presentes en cada documento de los modelos clásicos."""
    engine = context.engine
    terms = engine.query_terms(context.query, stem=context.stem)
    if not terms:
        return {}
    ids = {hit.doc_id for model, run in runs.items() if model != "dense" for hit in run.hits}
    try:
        return {doc_id: engine.matched_terms(doc_id, terms, stem=context.stem) for doc_id in ids}
    except EngineError:
        return {}  # sin resaltado, los resultados se muestran igual


def render(context: AppContext) -> None:
    if not context.query.strip():
        return  # el componente de la barra muestra el estado vacío con ejemplos
    slot = st.empty()
    runs = run_models(context, on_wait=lambda model: slot.html(loading_html(model)))
    rankings = {model: run.ids for model, run in runs.items()}
    if not any(rankings.values()):
        slot.html(f'{CSS}<div class="cmp"><p class="cmp-notice" role="status">Ningún modelo '
                  "devolvió resultados para esta consulta. Prueba con otras palabras.</p></div>")
        return
    positions = {model: {hit.doc_id: hit.rank for hit in run.hits} for model, run in runs.items()}
    seen, matched = appearances(rankings), matched_by_doc(context, runs)
    columns = "".join(
        _column_html(model, run, seen, context.k, positions,
                     matched if model != "dense" else {}, context.stem)
        for model, run in runs.items()
    )
    slot.html(
        f'{CSS}<div class="cmp">{head_html(context.k, context.stem, seen)}'
        f'<div class="cmp-grid">{columns}</div></div>'
    )
