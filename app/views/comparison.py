"""Vista de comparación lado a lado (#14): los cuatro rankings de una misma consulta.

Usa `context.search(modelo, skip_empty=True)` por cada modelo de `ris.engine.MODELS`,
igual que la vista Buscar, y `app.ui.clean_text` para el Markdown de Reddit. El
solapamiento está en `app.overlap`. Colores y medidas salen de los tokens de
`app/frontend/styles.css` (`:root`); los valores de respaldo son la paleta ciruela
de `.streamlit/config.toml`.

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
from app.overlap import (
    PairOverlap,
    appearances,
    exclusive_counts,
    mean_percent,
    overlaps,
)
from app.ui import clean_text
from ris.engine import MODELS, EngineError, Hit

MODEL_NAMES = {"jaccard": "Jaccard", "tfidf": "TF-IDF", "bm25": "BM25", "dense": "Embeddings"}
MODEL_KINDS = {
    "jaccard": "Binario", "tfidf": "Vectorial", "bm25": "Probabilístico", "dense": "Semántico",
}
LEXICAL_PAIRS = (("jaccard", "tfidf"), ("jaccard", "bm25"), ("tfidf", "bm25"))
SEMANTIC_PAIRS = (("jaccard", "dense"), ("tfidf", "dense"), ("bm25", "dense"))
SNIPPET_CHARS = 200
# Cada cuánto se comprueba si la búsqueda de un modelo terminó. Mientras espera, la
# vista actualiza su indicador de carga: cada actualización es un punto en el que
# Streamlit atiende una petición nueva (p. ej. volver a Buscar) sin esperar al modelo.
WAIT_STEP_S = 0.25
_SEARCHES = ThreadPoolExecutor(max_workers=len(MODELS), thread_name_prefix="jab-compare")

CSS = """
<style>
.st-key-jab_views:has(.st-key-jab_view_comparar) { width: min(1120px, 100% - 32px); }
/* Con la vista ya pintada sobra la línea «Buscando…» del componente (styles.css). */
.jab .cmp-pending { display: none !important; }
.cmp {
  --cmp-ink: var(--ink, #201A1F); --cmp-ink-2: var(--ink-2, #5E5560); --cmp-ink-3: var(--ink-3, #756B77);
  --cmp-line: var(--line, #E6E1E4); --cmp-surface: var(--surface, #F3F1F2);
  --cmp-accent: var(--accent, #7A2E5E); --cmp-soft: var(--accent-soft, #F3E7EF);
  color: var(--cmp-ink); font-size: .9375rem; line-height: 1.5;
}
.cmp, .cmp *, .cmp *::before, .cmp *::after { box-sizing: border-box; }
.cmp p { margin: 0; }
.cmp :focus-visible { outline: 2px solid var(--cmp-accent); outline-offset: 2px; border-radius: 6px; }

/* Línea de contexto: top-n, Porter, leyenda y ayuda desplegable */
.cmp-head { position: relative; display: flex; flex-wrap: wrap; align-items: center;
  gap: var(--s2, 8px) var(--s4, 16px); margin: var(--s2, 8px) 0 var(--s4, 16px);
  color: var(--cmp-ink-2); font-size: .8125rem; }
.cmp .cmp-context { font-size: .875rem; }
.cmp-context b { color: var(--cmp-ink); font-weight: 600; }
.cmp-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.cmp-help { margin-left: auto; }
.cmp-help summary { display: inline-flex; align-items: center; gap: 6px; min-height: 32px; padding: 0 var(--s3, 12px);
  border-radius: 999px; color: var(--cmp-accent); font-weight: 600; cursor: pointer; list-style: none; }
.cmp-help summary::-webkit-details-marker { display: none; }
.cmp-help summary:hover, .cmp-help[open] summary { background: var(--cmp-soft); }
.cmp-help summary svg { width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; }
.cmp-help ul { position: absolute; right: 0; top: calc(100% + 4px); z-index: 5; width: min(440px, 100%);
  margin: 0; padding: var(--s3, 12px) var(--s4, 16px) var(--s3, 12px) var(--s6, 32px);
  background: #fff; border: 1px solid var(--cmp-line); border-radius: var(--r-item, 10px);
  box-shadow: var(--shadow-menu, 0 16px 40px rgb(32 26 31 / .14)); color: var(--cmp-ink-2); font-size: .8125rem; }
.cmp-help li + li { margin-top: var(--s1, 4px); }

/* Solapamiento: dos grupos de tres pares, una fila por par */
.cmp-panel { margin-bottom: var(--s5, 24px); padding: var(--s4, 16px) var(--s5, 24px);
  background: #fff; border: 1px solid var(--cmp-line); border-radius: var(--r-menu, 18px);
  corner-shape: squircle; box-shadow: var(--shadow-bar, 0 1px 2px rgb(32 26 31 / .05)); }
.cmp-title { margin: 0 0 var(--s2, 8px); font-size: .9375rem; font-weight: 600; }
.cmp-groups { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s2, 8px) var(--s6, 32px); }
.cmp-group-head { display: flex; justify-content: space-between; gap: var(--s3, 12px); padding-bottom: var(--s1, 4px);
  border-bottom: 1px solid var(--cmp-line); color: var(--cmp-ink-2); font-size: .8125rem; font-weight: 600; }
.cmp-group-head span { color: var(--cmp-ink-2); font-weight: 400; font-variant-numeric: tabular-nums; }
.cmp .cmp-pairs { list-style: none; margin: 0; padding: 0; }
.cmp .cmp-pair { display: grid; grid-template-columns: minmax(0, 11rem) minmax(40px, 1fr) 3.25rem 3rem;
  align-items: center; gap: var(--s3, 12px); min-height: 32px; margin: 0; padding: 0;
  font-size: .875rem; font-variant-numeric: tabular-nums; }
.cmp-pair-names { min-width: 0; overflow: hidden; color: var(--cmp-ink-2); text-overflow: ellipsis; white-space: nowrap; }
.cmp-pair-names i { padding: 0 4px; color: var(--cmp-ink-3); font-style: normal; }
.cmp-count { color: var(--cmp-ink-2); text-align: right; white-space: nowrap; }
.cmp-count b { color: var(--cmp-ink); font-weight: 600; }
.cmp-pct { color: var(--cmp-accent); font-weight: 600; text-align: right; white-space: nowrap; }
.cmp-pair[data-empty="true"] .cmp-pct { color: var(--cmp-ink-3); }
.cmp-bar { height: 4px; border-radius: 2px; background: var(--cmp-line); overflow: hidden; }
.cmp-bar i { display: block; height: 100%; border-radius: 2px; background: var(--cmp-accent); }

/* Etiquetas de cobertura */
.cmp-badge {
  display: inline-flex; align-items: center; gap: 6px; padding: 1px 9px 1px 8px; border: 1px solid transparent;
  border-radius: 999px; font-size: .75rem; font-weight: 600; line-height: 1.5; white-space: nowrap;
}
.cmp-badge[data-level="todos"] { background: var(--cmp-accent); color: var(--on-accent, #fff); }
.cmp-badge[data-level="varios"] { background: var(--cmp-soft); color: var(--cmp-accent); }
.cmp-badge[data-level="solo"] { border-color: var(--cmp-line); color: var(--cmp-ink-2); }
.cmp-dots { display: inline-flex; gap: 2px; }
.cmp-dots i { width: 5px; height: 5px; border-radius: 50%; background: currentColor; opacity: .3; }
.cmp-dots i.on { opacity: 1; }

/* Columnas */
.cmp-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--s4, 16px); align-items: start; }
.cmp-col { min-width: 0; padding: var(--s4, 16px); background: #fff; border: 1px solid var(--cmp-line);
  border-radius: var(--r-menu, 18px); corner-shape: squircle; }
.cmp-col-head { display: flex; justify-content: space-between; align-items: flex-start; gap: var(--s2, 8px); }
.cmp-model { font-size: 1.0625rem; font-weight: 600; line-height: 1.3; }
.cmp-kind { color: var(--cmp-ink-3); font-size: .8125rem; }
.cmp-time { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 999px;
  background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .75rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
.cmp-time svg { width: 12px; height: 12px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; }
.cmp-sub { margin: var(--s2, 8px) 0 var(--s1, 4px); color: var(--cmp-ink-3); font-size: .8125rem; }
.cmp-empty { margin-top: var(--s3, 12px); padding: var(--s3, 12px); border-radius: var(--r-item, 10px);
  background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .875rem; }
.cmp .cmp-list { list-style: none; margin: 0; padding: 0; }
.cmp .cmp-list > li { margin: 0; padding-left: 0; }
.cmp-doc { padding: var(--s4, 16px) 0; border-top: 1px solid var(--cmp-line); }
.cmp-doc:first-child { border-top: 0; }
.cmp-doc-top { display: flex; align-items: baseline; gap: var(--s2, 8px); color: var(--cmp-ink-3); font-size: .8125rem; }
.cmp-rank { color: var(--cmp-ink-2); font-weight: 600; font-variant-numeric: tabular-nums; }
.cmp-doc[data-top="true"] .cmp-rank { color: var(--cmp-accent); font-weight: 700; }
.cmp-score { margin-left: auto; font-variant-numeric: tabular-nums; }
.cmp-snippet { margin-top: var(--s1, 4px); color: var(--cmp-ink); overflow-wrap: anywhere; }
.cmp-doc .cmp-badge { margin-top: var(--s2, 8px); }
.cmp-doc details { margin-top: var(--s2, 8px); font-size: .8125rem; }
.cmp-doc summary { width: fit-content; color: var(--cmp-accent); font-weight: 600; cursor: pointer; }
.cmp-doc summary:hover { text-decoration: underline; text-underline-offset: 3px; }
.cmp-doc details p { max-height: 260px; margin-top: var(--s2, 8px); overflow-y: auto; white-space: pre-line;
  color: var(--cmp-ink-2); overflow-wrap: anywhere; scrollbar-width: thin; }

/* Aviso sin resultados y carga */
.cmp-notice { margin-top: var(--s4, 16px); padding: var(--s3, 12px) var(--s4, 16px); border-radius: var(--r-item, 10px);
  background: var(--cmp-surface); color: var(--cmp-ink-2); font-size: .875rem; }
.cmp-loading { display: flex; align-items: center; gap: var(--s2, 8px); margin: var(--s2, 8px) 0 var(--s4, 16px);
  color: var(--cmp-ink-2); font-size: .8125rem; }
.cmp-loading::before { content: ""; width: 14px; height: 14px; border: 2px solid var(--cmp-line);
  border-top-color: var(--cmp-accent); border-radius: 50%; animation: cmp-spin .8s linear infinite; }
.cmp-skel { height: 220px; background: var(--cmp-surface); border-color: transparent; animation: cmp-pulse 1.2s ease-in-out infinite alternate; }
@keyframes cmp-spin { to { transform: rotate(360deg); } }
@keyframes cmp-pulse { from { opacity: 1; } to { opacity: .55; } }

@media (max-width: 1099px) { .cmp-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 719px) {
  .cmp-grid, .cmp-groups { grid-template-columns: minmax(0, 1fr); }
  .cmp-panel { padding: var(--s3, 12px) var(--s4, 16px); }
  .cmp-help { margin-left: 0; }
}
@media (max-width: 400px) {
  .cmp .cmp-pair { grid-template-columns: minmax(0, 1fr) 3.25rem 3rem; }
  .cmp-bar { display: none; }
}
@media (prefers-reduced-motion: reduce) { .cmp-loading::before, .cmp-skel { animation: none; } }
</style>
"""

CLOCK = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2"/><path d="M8 4.6V8l2.2 1.4"/></svg>'
HELP_ICON = ('<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.2"/>'
             '<path d="M6.4 6.3a1.7 1.7 0 1 1 2.3 1.6c-.5.2-.7.6-.7 1.1v.3M8 11.3v.1"/></svg>')


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


def format_percent(percent: float | None) -> str:
    return "—" if percent is None else f"{percent:.0f} %"


def coverage_badge(found_in: tuple[str, ...], own: str, total: int) -> tuple[str, str, str] | None:
    """(texto, nivel, ayuda) de la etiqueta de un documento, o None si no hay con qué comparar."""
    if total < 2:
        return None
    if len(found_in) == 1:
        return "Solo aquí", "solo", "Solo este modelo recuperó este documento"
    others = ", ".join(MODEL_NAMES[m] for m in found_in if m != own)
    level = "todos" if len(found_in) == total else "varios"
    return f"En {len(found_in)} de {total}", level, f"También lo recuperó: {others}"


def _badge_html(text: str, level: str, count: int, total: int, title: str = "") -> str:
    dots = "".join('<i class="on"></i>' if n < count else "<i></i>" for n in range(total))
    return (f'<span class="cmp-badge" data-level="{level}" title="{html.escape(title)}">'
            f'{html.escape(text)}<span class="cmp-dots" aria-hidden="true">{dots}</span></span>')


def _doc_html(hit: Hit, found_in: tuple[str, ...], model: str, total: int) -> str:
    text = clean_text(hit.text)
    short = snippet(text)
    badge = coverage_badge(found_in, model, total)
    badge_html = _badge_html(*badge[:2], len(found_in), total, badge[2]) if badge else ""
    full = ""
    if len(" ".join(text.split())) > len(short):
        body = html.escape(re.sub(r"\n\s*\n+", "\n\n", text))
        full = f"<details><summary>Ver texto completo</summary><p>{body}</p></details>"
    return (
        f'<li class="cmp-doc" data-top="{str(hit.rank == 1).lower()}">'
        f'<div class="cmp-doc-top"><span class="cmp-rank">#{hit.rank}</span>'
        f'<span>Doc. {html.escape(hit.doc_id)}</span>'
        f'<span class="cmp-score" title="Score propio del modelo">{hit.score:.3f}</span></div>'
        f'<p class="cmp-snippet">{html.escape(short) or "(documento sin texto)"}</p>'
        f"{badge_html}{full}</li>"
    )


def _column_html(model: str, run: ModelRun, seen: dict[str, tuple[str, ...]],
                 exclusive: dict[str, int], k: int, total: int) -> str:
    if run.error:
        sub, body = "No disponible", f'<p class="cmp-empty">{html.escape(run.error)}</p>'
    elif not run.hits:
        sub, body = "0 resultados", '<p class="cmp-empty">Sin resultados para esta consulta.</p>'
    else:
        count = f"{len(run.hits)} de {k} resultados" if len(run.hits) < k else f"{k} resultados"
        only = exclusive[model]
        sub = count + (f" · {only} solo aquí" if only and total > 1 else "")
        items = "".join(_doc_html(hit, seen[hit.doc_id], model, total) for hit in run.hits)
        body = f'<ol class="cmp-list">{items}</ol>'
    time = "" if run.error else f'<span class="cmp-time">{CLOCK}{format_ms(run.elapsed_ms)}</span>'
    return (
        '<section class="cmp-col">'
        '<div class="cmp-col-head"><div>'
        f'<div class="cmp-model" role="heading" aria-level="3">{MODEL_NAMES[model]}</div>'
        f'<div class="cmp-kind">{MODEL_KINDS[model]}</div></div>{time}</div>'
        f'<p class="cmp-sub">{sub}</p>{body}</section>'
    )


def _pair_html(pair: PairOverlap) -> str:
    names = f"{MODEL_NAMES[pair.a]}<i>∩</i>{MODEL_NAMES[pair.b]}"
    if pair.percent is None:
        count, bar, label = "—", 0.0, f"{names}: sin documentos que comparar"
    else:
        count, bar = f"<b>{pair.shared}</b> / {pair.total}", pair.percent
        label = (f"{names}: {pair.shared} de {pair.total} documentos en común, "
                 f"{format_percent(pair.percent)}")
    empty = str(pair.percent is None).lower()
    return (f'<li class="cmp-pair" data-empty="{empty}" title="{html.escape(re.sub("<[^>]+>", " ", label))}">'
            f'<span class="cmp-pair-names">{names}</span>'
            f'<span class="cmp-bar" aria-hidden="true"><i style="width:{bar:.0f}%"></i></span>'
            f'<span class="cmp-count">{count}</span>'
            f'<span class="cmp-pct">{format_percent(pair.percent)}</span></li>')


def _group_html(title: str, pairs: list[PairOverlap]) -> str:
    mean = mean_percent(pairs)
    average = "" if mean is None else f"media {format_percent(mean)}"
    return (f'<div class="cmp-group"><div class="cmp-group-head">{title}<span>{average}</span></div>'
            f'<ul class="cmp-pairs">{"".join(_pair_html(p) for p in pairs)}</ul></div>')


def overlap_html(pairs: list[PairOverlap]) -> str:
    by_pair = {(p.a, p.b): p for p in pairs}
    return (
        '<div class="cmp-panel">'
        '<div class="cmp-title" role="heading" aria-level="2">Solapamiento entre modelos</div>'
        '<div class="cmp-groups">'
        f'{_group_html("Léxicos entre sí", [by_pair[p] for p in LEXICAL_PAIRS])}'
        f'{_group_html("Léxicos frente a Embeddings", [by_pair[p] for p in SEMANTIC_PAIRS])}'
        "</div></div>"
    )


def legend_html(total: int) -> str:
    if total < 2:
        return ""
    samples = [_badge_html(f"En {total} de {total}", "todos", total, total)]
    if total > 2:
        middle = "2" if total == 3 else f"2–{total - 1}"
        samples.append(_badge_html(f"En {middle} de {total}", "varios", 2, total))
    samples.append(_badge_html("Solo aquí", "solo", 1, total))
    return ('<div class="cmp-legend" role="group" '
            'aria-label="En cuántos modelos aparece cada documento">'
            f'Cobertura: {"".join(samples)}</div>')


def help_html(k: int) -> str:
    """Ayuda desplegable: lo que antes eran párrafos bajo el título."""
    items = (
        "Cada modelo puntúa a su manera: los scores no se comparan entre columnas.",
        "Porter solo cambia Jaccard, TF-IDF y BM25; Embeddings no lo usa.",
        (
            "Solapamiento: documentos en común (mismo <code>doc_id</code>) sobre el menor de "
            f"los dos rankings, {k} si ambos están completos. «media» promedia los pares con "
            "documentos; «—», un ranking vacío."
        ),
        "Cobertura: en cuántos de los modelos disponibles aparece cada documento.",
    )
    return (f'<details class="cmp-help"><summary>{HELP_ICON}Cómo leerla</summary>'
            f'<ul>{"".join(f"<li>{item}</li>" for item in items)}</ul></details>')


def head_html(k: int, stem: bool, total: int) -> str:
    porter = "Porter activado" if stem else "Porter desactivado"
    return (f'<div class="cmp-head"><p class="cmp-context"><b>Top-{k}</b> por modelo · {porter}</p>'
            f"{legend_html(total)}{help_html(k)}</div>")


def loading_html(model: str) -> str:
    """Indicador mientras falta un modelo: mismo ancho y rejilla que el resultado."""
    skeleton = '<div class="cmp-col cmp-skel" aria-hidden="true"></div>' * len(MODELS)
    return (f'{CSS}<div class="cmp"><p class="cmp-loading" role="status">Buscando con los '
            f"cuatro modelos… esperando a {MODEL_NAMES[model]}</p>"
            f'<div class="cmp-grid">{skeleton}</div></div>')


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
    total = sum(1 for run in runs.values() if not run.error)
    seen, exclusive = appearances(rankings), exclusive_counts(rankings)
    columns = "".join(
        _column_html(model, run, seen, exclusive, context.k, total) for model, run in runs.items()
    )
    slot.html(
        f'{CSS}<div class="cmp">{head_html(context.k, context.stem, total)}'
        f'{overlap_html(overlaps(rankings))}<div class="cmp-grid">{columns}</div></div>'
    )
