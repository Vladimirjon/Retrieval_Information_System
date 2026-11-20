"""Solapamiento y vista de comparación lado a lado (#14), sin motor ni red."""

import html
import re
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.context import AppContext
from app.overlap import (
    PairOverlap,
    appearances,
    exclusive_counts,
    mean_percent,
    overlaps,
    pair_overlap,
)
from app.ui import highlight_html, highlight_spans
from app.views import comparison
from ris.engine import MODELS, Hit, ModelUnavailableError, SearchEngine, SearchResponse
from ris.preprocessing import preprocess

# ------------------------------------------------------------------ solapamiento


def test_overlap_counts_exact_intersection_over_k():
    pair = pair_overlap("bm25", ["a", "b", "c", "d"], "dense", ["c", "d", "e", "f"])
    assert (pair.shared, pair.total) == (2, 4)
    assert pair.percent == 50.0


def test_overlap_is_symmetric_and_ignores_order():
    a, b = ["a", "b", "c"], ["c", "x", "a"]
    one, other = pair_overlap("m1", a, "m2", b), pair_overlap("m2", b, "m1", a)
    assert (one.shared, one.total) == (other.shared, other.total) == (2, 3)


def test_identical_and_disjoint_rankings():
    assert pair_overlap("a", ["1", "2"], "b", ["2", "1"]).percent == 100.0
    disjoint = pair_overlap("a", ["1", "2"], "b", ["3", "4"])
    assert (disjoint.shared, disjoint.total, disjoint.percent) == (0, 2, 0.0)


def test_short_ranking_uses_the_smaller_size_as_denominator():
    # el modelo corto solo dio 3 documentos y los 3 están en el largo: 3 de 3, no 3 de 10
    long = [str(n) for n in range(10)]
    pair = pair_overlap("jaccard", ["0", "1", "2"], "bm25", long)
    assert (pair.shared, pair.total, pair.percent) == (3, 3, 100.0)


def test_empty_ranking_has_zero_denominator_and_no_percentage():
    pair = pair_overlap("jaccard", [], "bm25", ["1", "2"])
    assert (pair.shared, pair.total, pair.percent) == (0, 0, None)
    assert pair_overlap("a", [], "b", []).percent is None


def test_duplicates_do_not_inflate_the_count():
    pair = pair_overlap("a", ["1", "1", "2"], "b", ["1", "3"])
    assert (pair.shared, pair.total) == (1, 2)


def test_overlaps_returns_the_six_pairs_in_model_order():
    rankings = {"jaccard": ["a", "b"], "tfidf": ["a", "c"], "bm25": ["a", "b"], "dense": ["z", "b"]}
    result = overlaps(rankings)
    assert [(p.a, p.b) for p in result] == [
        ("jaccard", "tfidf"), ("jaccard", "bm25"), ("jaccard", "dense"),
        ("tfidf", "bm25"), ("tfidf", "dense"), ("bm25", "dense"),
    ]
    by_pair = {(p.a, p.b): (p.shared, p.total) for p in result}
    assert by_pair[("jaccard", "bm25")] == (2, 2)
    assert by_pair[("tfidf", "dense")] == (0, 2)
    assert by_pair[("bm25", "dense")] == (1, 2)


def test_overlaps_with_empty_and_short_rankings():
    rankings = {"jaccard": [], "tfidf": ["a"], "bm25": ["a", "b", "c"], "dense": []}
    by_pair = {(p.a, p.b): p for p in overlaps(rankings)}
    assert by_pair[("jaccard", "dense")].percent is None
    assert by_pair[("tfidf", "bm25")] == PairOverlap("tfidf", "bm25", 1, 1)
    assert overlaps({"solo": ["a"]}) == []


def test_mean_percent_skips_pairs_without_documents():
    pairs = [PairOverlap("a", "b", 1, 2), PairOverlap("a", "c", 0, 0), PairOverlap("b", "c", 2, 2)]
    assert mean_percent(pairs) == 75.0
    assert mean_percent([PairOverlap("a", "b", 0, 0)]) is None
    assert mean_percent([]) is None


def test_appearances_and_exclusive_documents():
    rankings = {"jaccard": ["a", "b"], "tfidf": ["a", "c"], "bm25": [], "dense": ["a", "b", "d"]}
    seen = appearances(rankings)
    assert seen["a"] == ("jaccard", "tfidf", "dense")
    assert seen["b"] == ("jaccard", "dense")
    assert seen["d"] == ("dense",)
    assert exclusive_counts(rankings) == {"jaccard": 0, "tfidf": 1, "bm25": 0, "dense": 1}


# ------------------------------------------------------------ ayudas de la vista


def test_snippet_cuts_on_whole_words_and_flattens_lines():
    assert comparison.snippet("uno\n\ndos   tres") == "uno dos tres"
    text = "palabra " * 50
    short = comparison.snippet(text, 20)
    assert short == "palabra palabra…"
    assert comparison.snippet("", 20) == ""


def test_formatting():
    assert comparison.format_ms(3.456) == "3.5 ms"
    assert comparison.format_ms(148.6) == "149 ms"


def test_highlight_html_marks_query_words_and_escapes_the_rest():
    text = "Index <b>funds</b> 🚀 y fondos de índice: cheap Funds."
    marked = highlight_html(text, ("fund",), stem=True)
    assert marked == ("Index &lt;b&gt;<mark>funds</mark>&lt;/b&gt; 🚀 y fondos de índice: "
                      "cheap <mark>Funds</mark>.")
    assert highlight_html(text, (), stem=False) == html.escape(text)


def test_highlight_html_marks_the_same_words_as_the_utf16_spans():
    text = "😀 Índice: invest, investing 🚀 and investment café"
    matched = ("invest",)
    spans = highlight_spans(text, matched, stem=True)
    units = text.encode("utf-16-le")
    expected = [units[2 * a:2 * b].decode("utf-16-le") for a, b in spans]
    marked = re.findall(r"<mark>(.*?)</mark>", highlight_html(text, matched, stem=True))
    assert marked == expected and len(marked) >= 2


# --------------------------------------------------------------- motor simulado


class FakeEngine:
    """Devuelve rankings fijos por modelo y registra cómo se le pregunta."""

    def __init__(self, rankings, errors=None, texts=None, delays=None):
        self.rankings = rankings
        self.errors = errors or {}
        self.texts = texts or {}
        self.delays = delays or {}
        self.calls = []

    def search(self, model, query, k, *, stem=False, skip_empty=False):
        self.calls.append((model, query, k, stem, skip_empty))
        time.sleep(self.delays.get(model, 0))
        if model in self.errors:
            raise self.errors[model]
        hits = tuple(
            Hit(rank, doc_id, "", self.texts.get(doc_id, f"Texto **del** documento {doc_id}"),
                10.0 - rank)
            for rank, doc_id in enumerate(self.rankings.get(model, [])[:k], start=1)
        )
        return SearchResponse(model=model, stem=stem, query=query, k=k, hits=hits, elapsed_ms=7.0)

    query_terms = staticmethod(SearchEngine.query_terms)

    def matched_terms(self, doc_id, terms, *, stem=False):
        present = set(preprocess(self.texts.get(doc_id, ""), stem=stem))
        return tuple(term for term in terms if term in present)


def make_context(engine, query="index funds", k=5):
    return AppContext(engine=engine, query=query, model="bm25", k=k, stem=False, reports_dir=Path("r"))


FULL = {
    "jaccard": ["1", "2", "3", "4", "5"],
    "tfidf": ["1", "2", "6", "7", "8"],
    "bm25": ["1", "3", "6", "9", "10"],
    "dense": ["1", "11", "12", "13", "14"],
}


def test_context_search_skip_empty_is_opt_in():
    engine = FakeEngine(FULL)
    context = make_context(engine)
    context.search("bm25")
    context.search("dense", skip_empty=True)
    assert engine.calls == [
        ("bm25", "index funds", 5, False, False), ("dense", "index funds", 5, False, True),
    ]


def test_run_models_searches_all_models_like_the_search_view():
    engine = FakeEngine(FULL)
    runs = comparison.run_models(make_context(engine, k=5))
    assert list(runs) == list(MODELS)
    assert [call[0] for call in engine.calls] == list(MODELS)
    assert all(call[1:] == ("index funds", 5, False, True) for call in engine.calls)
    assert runs["dense"].ids == ["1", "11", "12", "13", "14"]
    assert runs["bm25"].elapsed_ms == 7.0


def test_run_models_keeps_other_models_when_one_fails():
    engine = FakeEngine(FULL, errors={"dense": ModelUnavailableError("Embeddings no disponible")})
    runs = comparison.run_models(make_context(engine))
    assert runs["dense"].error == "Embeddings no disponible" and runs["dense"].ids == []
    assert runs["bm25"].error is None and runs["bm25"].ids == FULL["bm25"]


def test_run_models_propagates_a_timeout_raised_by_the_search():
    # Un TimeoutError del propio motor no es «aún no termina»: debe salir, no girar sin fin.
    engine = FakeEngine(FULL, errors={"jaccard": TimeoutError("agotado")})
    waits = []
    try:
        comparison.run_models(make_context(engine), on_wait=waits.append)
    except TimeoutError as error:
        assert str(error) == "agotado" and waits == []
    else:
        raise AssertionError("el TimeoutError de la búsqueda se perdió")


def test_run_models_reports_the_slow_model_while_waiting(monkeypatch):
    # Mientras Embeddings tarda, la vista recibe avisos periódicos (puntos en los que
    # Streamlit puede atender otra petición) y el resultado no cambia.
    monkeypatch.setattr(comparison, "WAIT_STEP_S", 0.02)
    engine = FakeEngine(FULL, delays={"dense": 0.15})
    waits = []
    runs = comparison.run_models(make_context(engine), on_wait=waits.append)
    assert waits and set(waits) == {"dense"}
    assert [call[0] for call in engine.calls] == list(MODELS)
    assert runs["dense"].ids == FULL["dense"]


# ------------------------------------------------------------------ vista completa


def _view(rankings, query="index funds", k=5, errors=None, texts=None, stem=False):
    def script(rankings, query, k, errors, texts, stem):
        from pathlib import Path

        from app.context import AppContext
        from app.views.comparison import render
        from tests.test_comparison_view import FakeEngine

        engine = FakeEngine(rankings, errors, texts)
        render(AppContext(engine=engine, query=query, model="bm25", k=k, stem=stem,
                          reports_dir=Path("r")))

    return AppTest.from_function(script, kwargs={
        "rankings": rankings, "query": query, "k": k, "errors": errors, "texts": texts,
        "stem": stem,
    }).run(timeout=30)


def _html(at):
    return "".join(element.proto.body for element in at.get("html"))


def test_empty_query_draws_nothing_and_leaves_the_empty_state_to_the_bar():
    # Sin consulta, el componente de la barra muestra la invitación y los ejemplos
    # (tests/test_app.py); la vista no pinta avisos de Streamlit ni busca.
    for query in ("", "   "):
        at = _view(FULL, query=query)
        assert not at.exception
        assert at.info == [] and at.header == [] and at.get("html") == []


def test_results_have_no_page_title_and_a_minimal_context_line():
    at = _view(FULL)
    page = _html(at)
    assert at.header == [] and at.caption == [] and at.info == []
    assert "Comparación</" not in page
    assert '<span class="cmp-pill">Top 5</span>' in page and ">Porter<" not in page
    assert '<span class="cmp-pill" data-kind="porter">Porter</span>' in _html(_view(FULL, stem=True))
    # Ayuda: un botón «?» con el tooltip de la página, sin desplegable
    assert "<details class=\"cmp-help\"" not in page and "Cómo leerla" not in page
    button = page.split('<button type="button" class="cmp-help"', 1)[1].split("</button>", 1)[0]
    assert 'aria-label="Cómo leer la comparación"' in button and ">?" in button
    assert "documento compartido" in button and "Ver en" in button
    assert "no se comparan" in button
    # Sin copy viejo: ni rayas largas ni frases de carga o de resumen por columna
    visible = page.split("</style>", 1)[1]
    for gone in ("—", "Buscando con los", "solo aquí", "Solo aquí", "resultados</p>", "media "):
        assert gone not in visible
    # Con la vista pintada, la línea «Buscando…» del componente se oculta
    assert ".jab .cmp-pending { display: none !important; }" in page


def test_full_rankings_render_a_counted_segmented_filter_and_neutral_positions():
    page = _html(_view(FULL))
    assert page.count('class="cmp-col"') == 4
    assert page.count('<li class="cmp-doc"') == 20
    visible = page.split("</style>", 1)[1]
    for removed in ('data-heat', 'cmp-matrix', 'cmp-scale', 'cmp-chip', 'cmp-consensus', 'coinciden en'):
        assert removed not in page
    assert 'data-only="false" aria-pressed="true">Todos</button>' in visible
    assert 'data-only="true" aria-pressed="false">Coincidencias (4)</button>' in visible
    assert "Texto del documento 1" in page and "**" not in visible


def test_jump_button_points_to_the_next_column_and_wraps_around():
    page = _html(_view(FULL))
    tfidf = page.split('<section class="cmp-col">')[2]
    shared = tfidf.split('data-doc-id="1"', 1)[1].split('</li>', 1)[0]
    assert 'data-jump-model="bm25"' in shared and "Ver en BM25 #1" in shared
    assert 'data-tip="También en Jaccard #1, BM25 #1, Embeddings #1"' in shared
    assert 'TF-IDF #' not in shared
    dense = page.split('<section class="cmp-col">')[4]
    wrapped = dense.split('data-doc-id="1"', 1)[1].split('</li>', 1)[0]
    assert 'data-jump-model="jaccard"' in wrapped and "Ver en Jaccard #1" in wrapped
    unique = tfidf.split('data-doc-id="8"', 1)[1].split('</li>', 1)[0]
    assert 'cmp-jump' not in unique and 'data-model="tfidf"' in unique
    assert "tabindex" not in unique and "tabindex=0" in shared


def test_short_empty_and_unavailable_rankings_use_only_shown_documents():
    rankings = {"jaccard": [], "tfidf": ["1", "2"], "bm25": ["1", "2", "3"], "dense": ["2"]}
    page = _html(_view(rankings, k=1, errors={"dense": ModelUnavailableError("Unavailable")}))
    assert "Sin resultados para esta consulta." in page and "Unavailable" in page
    assert "Coincidencias (1)</button>" in page


def test_disjoint_rankings_show_no_toggle_or_tags():
    page = _html(_view({"tfidf": ["a"], "bm25": ["b"]}))
    visible = page.split("</style>", 1)[1]
    assert 'cmp-seg' not in visible and 'cmp-jump' not in visible


def test_tags_escape_ids_and_model_names(monkeypatch):
    attack = '<img src=x onerror="alert(1)">'
    monkeypatch.setitem(comparison.MODEL_NAMES, "bm25", attack)
    page = _html(_view({"tfidf": [attack], "bm25": [attack]}))
    assert attack not in page
    assert f'Ver en {html.escape(attack)} #1' in page
    assert f'data-tip="También en {html.escape(attack)} #1"' in page


def test_many_shared_documents_are_only_counted():
    page = _html(_view({model: [str(i) for i in range(50)] for model in MODELS}, k=50))
    assert "Coincidencias (50)</button>" in page
    assert 'prefers-reduced-motion: reduce' in comparison.CSS


def test_classical_models_highlight_query_words_and_embeddings_does_not():
    texts = {"1": "Index funds 🚀 and Índice: <low> fees for funds. " + "filler " * 60,
             "11": "Index funds again"}
    page = _html(_view(FULL, texts=texts))
    jaccard, tfidf, bm25, dense = page.split('<section class="cmp-col">')[1:]
    for column in (jaccard, tfidf, bm25):
        assert "<mark>Index</mark> <mark>funds</mark> 🚀 and Índice: &lt;low&gt;" in column
        # el texto completo de «Ver más» también resalta
        assert "<summary><span class=\"cmp-more\">Ver más</span>" in column
        assert "fees for <mark>funds</mark>." in column.split("<details>", 1)[1]
    assert "<mark>" not in dense


def test_no_results_in_any_model_shows_a_notice():
    at = _view({})
    assert not at.exception
    page = _html(at)
    assert at.info == []  # aviso propio, no el cuadro azul de Streamlit
    assert '<p class="cmp-notice" role="status">Ningún modelo devolvió resultados' in page
    assert 'class="cmp-col"' not in page


def test_corpus_text_ids_and_errors_are_escaped():
    attack = '<img src=x onerror="alert(1)">'
    rankings = {"jaccard": ["<b>1</b>"], "tfidf": [], "bm25": [], "dense": []}
    at = _view(rankings, texts={"<b>1</b>": attack},
               errors={"dense": ModelUnavailableError("<script>x</script>")})
    page = _html(at)
    assert attack not in page and "<script>x" not in page and "<b>1</b>" not in page
    assert "&lt;img src=x" in page and "&lt;script&gt;x" in page and "&lt;b&gt;1" in page


def test_view_follows_the_agreed_contract():
    assert list(comparison.render.__annotations__) == ["context", "return"]
