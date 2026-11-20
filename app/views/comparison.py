"""Vista de comparación lado a lado (#14): los cuatro rankings de una misma consulta.

Usa `context.search(modelo, skip_empty=True)` por cada modelo de `ris.engine.MODELS`,
igual que la vista Buscar, y `app.ui.clean_text` para el Markdown de Reddit. El
solapamiento está en `app.overlap`. Colores y medidas salen de los tokens de
`app/frontend/styles.css` (`:root`); los valores de respaldo son la paleta ciruela
de `.streamlit/config.toml`.
"""

from __future__ import annotations

import html
import re
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

CSS = """
<style>
.st-key-jab_views { width: min(1320px, 100% - 32px); }
.cmp {
  --cmp-ink: var(--ink, #201A1F); --cmp-ink-2: var(--ink-2, #5E5560); --cmp-ink-3: var(--ink-3, #756B77);
  --cmp-line: var(--line, #E6E1E4); --cmp-surface: var(--surface, #F3F1F2);
  --cmp-accent: var(--accent, #7A2E5E); --cmp-soft: var(--accent-soft, #F3E7EF);
  color: var(--cmp-ink); font-size: .9375rem; line-height: 1.5;
}
.cmp, .cmp *, .cmp *::before, .cmp *::after { box-sizing: border-box; }
.cmp p { margin: 0; }
.cmp-title { margin: 0 0 var(--s3, 12px); font-size: 1.0625rem; font-weight: 600; }
.cmp-note { margin-top: var(--s3, 12px); color: var(--cmp-ink-3); font-size: .8125rem; }
.cmp-panel {
  margin-bottom: var(--s6, 32px); padding: var(--s5, 24px);
  background: #fff; border: 1px solid var(--cmp-line); border-radius: var(--r-menu, 18px);
  corner-shape: squircle; box-shadow: var(--shadow-bar, 0 1px 2px rgb(32 26 31 / .05));
}

/* Solapamiento: tarjetas por pares */
.cmp-group + .cmp-group { margin-top: var(--s5, 24px); }
.cmp-group-head { display: flex; justify-content: space-between; gap: var(--s3, 12px); margin-bottom: var(--s2, 8px);
  color: var(--cmp-ink-2); font-size: .8125rem; font-weight: 600; }
.cmp-group-head span { color: var(--cmp-ink-3); font-weight: 400; }
.cmp-pairs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--s3, 12px); }
.cmp-pair { padding: var(--s3, 12px) var(--s4, 16px); background: var(--cmp-surface); border-radius: var(--r-item, 10px); }
.cmp-pair-names { color: var(--cmp-ink-2); font-size: .8125rem; }
.cmp-pair-names i { padding: 0 4px; color: var(--cmp-ink-3); font-style: normal; }
.cmp-pair-val { display: flex; align-items: baseline; gap: var(--s2, 8px); margin: var(--s1, 4px) 0 var(--s2, 8px);
  font-variant-numeric: tabular-nums; }
.cmp-pair-val b { font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
.cmp-pair-val em { color: var(--cmp-ink-3); font-style: normal; }
.cmp-pair-val span { margin-left: auto; color: var(--cmp-accent); font-weight: 600; }
.cmp-bar { height: 4px; border-radius: 2px; background: var(--cmp-line); overflow: hidden; }
.cmp-bar i { display: block; height: 100%; border-radius: 2px; background: var(--cmp-accent); }

/* Etiquetas de cobertura */
.cmp-legend { display: flex; flex-wrap: wrap; align-items: center; gap: var(--s2, 8px) var(--s3, 12px);
  margin-bottom: var(--s4, 16px); color: var(--cmp-ink-3); font-size: .8125rem; }
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

@media (max-width: 1099px) { .cmp-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 719px) {
  .cmp-grid, .cmp-pairs { grid-template-columns: minmax(0, 1fr); }
  .cmp-panel { padding: var(--s4, 16px); }
}
</style>
"""

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


def run_models(context: AppContext) -> dict[str, ModelRun]:
    """Una búsqueda por modelo; si uno falla, los demás se muestran igual."""
    runs = {}
    for model in MODELS:
        try:
            response = context.search(model, skip_empty=True)
            runs[model] = ModelRun(response.hits, response.elapsed_ms)
        except EngineError as error:
            runs[model] = ModelRun(error=str(error))
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
        value = "<b>—</b><span>sin documentos que comparar</span>"
        bar = 0
    else:
        value = (f"<b>{pair.shared}</b><em>/ {pair.total}</em>"
                 f"<span>{format_percent(pair.percent)}</span>")
        bar = pair.percent
    return (f'<div class="cmp-pair"><div class="cmp-pair-names">{names}</div>'
            f'<div class="cmp-pair-val">{value}</div>'
            f'<div class="cmp-bar"><i style="width:{bar:.0f}%"></i></div></div>')


def _group_html(title: str, pairs: list[PairOverlap]) -> str:
    mean = mean_percent(pairs)
    average = "" if mean is None else f"media {format_percent(mean)}"
    return (f'<div class="cmp-group"><div class="cmp-group-head">{title}<span>{average}</span></div>'
            f'<div class="cmp-pairs">{"".join(_pair_html(p) for p in pairs)}</div></div>')


def overlap_html(pairs: list[PairOverlap], k: int) -> str:
    by_pair = {(p.a, p.b): p for p in pairs}
    return (
        '<div class="cmp-panel">'
        '<div class="cmp-title" role="heading" aria-level="2">Solapamiento entre modelos</div>'
        f'{_group_html("Léxicos entre sí", [by_pair[p] for p in LEXICAL_PAIRS])}'
        f'{_group_html("Léxicos frente a Embeddings", [by_pair[p] for p in SEMANTIC_PAIRS])}'
        '<p class="cmp-note">Documentos en común (mismo <code>doc_id</code>) sobre el menor de los dos '
        f"rankings: es <b>{k}</b> cuando ambos están completos. «media» promedia los porcentajes "
        "de los pares con documentos.</p></div>"
    )


def legend_html(total: int) -> str:
    if total < 2:
        return ""
    samples = [_badge_html(f"En {total} de {total}", "todos", total, total)]
    if total > 2:
        middle = "2" if total == 3 else f"2–{total - 1}"
        samples.append(_badge_html(f"En {middle} de {total}", "varios", 2, total))
    samples.append(_badge_html("Solo aquí", "solo", 1, total))
    return ('<div class="cmp-legend">En cuántos modelos aparece cada documento: '
            f'{"".join(samples)}</div>')


def render(context: AppContext) -> None:
    st.header("Comparación")
    if not context.query.strip():
        st.info("Escribe una consulta en la barra para comparar los cuatro modelos lado a lado.")
        return
    st.caption(
        f"Top-{context.k} de cada modelo para la misma consulta. Porter "
        f"{'activado' if context.stem else 'desactivado'} en los clásicos; Embeddings no lo usa. "
        "Los scores son de cada modelo y no se pueden comparar entre ellos."
    )
    with st.spinner("Buscando con los cuatro modelos…"):
        runs = run_models(context)
    rankings = {model: run.ids for model, run in runs.items()}
    if not any(rankings.values()):
        st.info("Ningún modelo devolvió resultados para esta consulta. Prueba con otras palabras.")
        return
    total = sum(1 for run in runs.values() if not run.error)
    seen, exclusive = appearances(rankings), exclusive_counts(rankings)
    columns = "".join(
        _column_html(model, run, seen, exclusive, context.k, total) for model, run in runs.items()
    )
    st.html(
        f'{CSS}<div class="cmp">{overlap_html(overlaps(rankings), context.k)}'
        f'{legend_html(total)}<div class="cmp-grid">{columns}</div></div>'
    )
