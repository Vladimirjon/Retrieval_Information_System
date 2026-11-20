"""Solapamiento y vista de comparación lado a lado (#14), sin motor ni red."""

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
from app.views import comparison
from ris.engine import MODELS, Hit, ModelUnavailableError, SearchResponse

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


def test_coverage_badge_levels():
    names = comparison.MODEL_NAMES
    assert comparison.coverage_badge(("bm25",), "bm25", 4)[:2] == ("Solo aquí", "solo")
    text, level, help_text = comparison.coverage_badge(("bm25", "dense"), "bm25", 4)
    assert (text, level) == ("En 2 de 4", "varios")
    assert names["dense"] in help_text and names["bm25"] not in help_text
    assert comparison.coverage_badge(tuple(MODELS), "bm25", 4)[:2] == ("En 4 de 4", "todos")
    assert comparison.coverage_badge(("bm25",), "bm25", 1) is None


def test_formatting():
    assert comparison.format_ms(3.456) == "3.5 ms"
    assert comparison.format_ms(148.6) == "149 ms"
    assert comparison.format_percent(None) == "—"
    assert comparison.format_percent(66.66) == "67 %"


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


def test_results_have_no_page_title_and_one_short_context_line():
    at = _view(FULL)
    page = _html(at)
    assert at.header == [] and at.caption == [] and at.info == []
    assert "Comparación</" not in page
    assert '<p class="cmp-context"><b>Top-5</b> por modelo · Porter desactivado</p>' in page
    assert "Porter activado" in _html(_view(FULL, stem=True))
    # Las explicaciones largas quedan en una ayuda desplegable, no en párrafos visibles
    help_box = page.split('<details class="cmp-help">', 1)[1].split("</details>", 1)[0]
    assert "<summary>" in help_box and "no se comparan" in help_box
    assert "menor de los" in help_box and "5 si ambos están completos" in help_box
    assert "Cobertura" in help_box
    # Con la vista pintada, la línea «Buscando…» del componente se oculta
    assert ".jab .cmp-pending { display: none !important; }" in page


def test_full_rankings_render_four_columns_and_counts():
    at = _view(FULL)
    assert not at.exception
    page = _html(at)
    assert page.count('class="cmp-col"') == 4
    for name in comparison.MODEL_NAMES.values():
        assert name in page
    assert page.count('<li class="cmp-doc"') == 20
    # jaccard∩tfidf = {1,2}: 2 / 5, 40 %; seis pares en dos grupos de tres
    assert '<span class="cmp-count"><b>2</b> / 5</span><span class="cmp-pct">40 %</span>' in page
    assert page.count('<li class="cmp-pair"') == 6 and page.count('class="cmp-group"') == 2
    assert 'title="Jaccard ∩ TF-IDF: 2 de 5 documentos en común, 40 %"' in page
    assert "media 40 %" in page  # léxicos: (40 + 40 + 40) / 3
    # el documento 1 sale en los cuatro modelos; el 14 solo en Embeddings
    assert page.count("En 4 de 4") >= 4
    assert "Solo aquí" in page
    assert "Texto del documento 1" in page and "**" not in page


def test_short_and_empty_rankings_do_not_break_the_view():
    rankings = {"jaccard": [], "tfidf": ["1", "2"], "bm25": ["1", "2", "3", "4", "5"], "dense": ["2"]}
    at = _view(rankings)
    assert not at.exception
    page = _html(at)
    assert "Sin resultados para esta consulta." in page
    assert "2 de 5 resultados" in page and "1 de 5 resultados" in page
    # denominador efectivo: tfidf (2) frente a bm25 (5) comparte 2 de 2
    assert '<span class="cmp-count"><b>2</b> / 2</span><span class="cmp-pct">100 %</span>' in page
    # pares con Jaccard vacío: «—», sin porcentaje y con la explicación en el título
    assert page.count('<li class="cmp-pair" data-empty="true"') == 3
    assert '<span class="cmp-count">—</span><span class="cmp-pct">—</span>' in page
    assert "sin documentos que comparar" in page


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


def test_unavailable_model_is_reported_and_the_rest_still_show():
    at = _view(FULL, errors={"dense": ModelUnavailableError("Embeddings no disponible")})
    assert not at.exception
    page = _html(at)
    assert "Embeddings no disponible" in page and "No disponible" in page
    assert page.count('<li class="cmp-doc"') == 15
    assert "En 3 de 3" in page
    assert "Cobertura:" in page and "En 2 de 3" in page  # la leyenda usa los 3 disponibles


def test_view_follows_the_agreed_contract():
    assert list(comparison.render.__annotations__) == ["context", "return"]
