import re
from pathlib import Path

import pytest

from app.context import AppContext
from app.ui import (
    DEFAULT_REQUEST,
    FRONTEND,
    TOP_K,
    WORDMARK,
    Request,
    build_assets,
    clean_text,
    highlight_spans,
    parse_request,
    parse_view,
    respond,
)
from app.views import comparison, metrics
from ris.engine import ArtifactsMissingError, Hit, SearchResponse

# ---------- Petición del navegador (no confiable) ----------

def test_valid_request_is_kept_and_whitespace_normalized():
    request = parse_request({"q": "  credit   score ", "model": "tfidf", "k": 20, "stem": True, "seq": 7})
    assert request == Request(q="credit score", model="tfidf", k=20, stem=True, view="buscar", seq=7)


@pytest.mark.parametrize("raw", [None, "x", 3, [], {"q": 5}, {"model": "lsi"}, {}])
def test_invalid_requests_fall_back_to_defaults(raw):
    request = parse_request(raw)
    assert request.model in {"jaccard", "tfidf", "bm25", "dense"}
    assert request.k in TOP_K
    assert isinstance(request.q, str)


@pytest.mark.parametrize("k", [0, 3, 51, -5, "10", 10.0, True])
def test_top_k_outside_the_offered_options_falls_back_to_ten(k):
    assert parse_request({"q": "a", "k": k}).k == 10


@pytest.mark.parametrize("seq", [-1, "3", 1.5, True, None])
def test_bad_seq_becomes_zero(seq):
    assert parse_request({"q": "a", "seq": seq}).seq == 0


def test_stem_is_ignored_for_dense_and_requires_true():
    assert parse_request({"q": "a", "model": "dense", "stem": True}).stem is False
    assert parse_request({"q": "a", "model": "bm25", "stem": "yes"}).stem is False


def test_long_query_is_truncated():
    assert len(parse_request({"q": "a" * 1000}).q) == 300


def test_view_is_whitelisted_and_travels_with_the_request():
    assert parse_view("comparar") == "comparar"
    assert parse_view("<script>") == "buscar"
    assert parse_view(None) == "buscar"
    assert parse_request({"q": "a", "view": "metricas"}).view == "metricas"
    assert parse_request({"q": "a", "view": "admin"}).view == "buscar"


def test_search_key_ignores_view_and_seq_but_not_the_search_parameters():
    a = parse_request({"q": "a", "model": "bm25", "k": 10, "view": "buscar", "seq": 1})
    b = parse_request({"q": "a", "model": "bm25", "k": 10, "view": "comparar", "seq": 9})
    c = parse_request({"q": "a", "model": "bm25", "k": 20, "view": "buscar", "seq": 1})
    assert a.key == b.key
    assert a.key != c.key


# ---------- Respuesta ----------

class FakeEngine:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def search(self, model, query, k, *, stem, skip_empty=False):
        self.calls.append((model, query, k, stem, skip_empty))
        if self.error:
            raise self.error
        hits = (Hit(1, "501384", "", "Texto <b>uno</b>.", 10.5), Hit(2, "315741", "", "", 9.0))
        return SearchResponse(model=model, stem=stem, query=query, k=k, hits=hits, elapsed_ms=12.345)

    @staticmethod
    def query_terms(query, *, stem=False):
        return tuple(dict.fromkeys(query.lower().split()))

    def matched_terms(self, doc_id, terms, *, stem=False):
        self.calls.append(("matched", doc_id, terms, stem))
        return terms[:1] if doc_id == "501384" else ()


def test_response_carries_seq_hits_and_rounded_time():
    engine = FakeEngine()
    request = Request(q="savings", model="bm25", k=5, stem=True, view="buscar", seq=4)
    response = respond(engine, request)
    assert engine.calls[0] == ("bm25", "savings", 5, True, True)
    assert response["seq"] == 4
    assert response["key"] == request.key
    assert response["elapsed_ms"] == 12.3
    assert response["terms"] == ["savings"]
    assert response["hits"][0] == {
        "rank": 1, "doc_id": "501384", "text": "Texto <b>uno</b>.", "score": 10.5,
        "matched": ["savings"], "spans": [],
    }
    assert response["hits"][1]["matched"] == []
    assert response["hits"][1]["spans"] == []
    assert len(response["hits"]) == 2


def test_dense_response_has_no_terms_and_does_not_touch_the_classical_index():
    engine = FakeEngine()
    request = Request(q="savings", model="dense", k=5, stem=False, view="buscar", seq=1)
    response = respond(engine, request)
    assert response["terms"] == []
    assert all(hit["matched"] == [] for hit in response["hits"])
    assert not any(call[0] == "matched" for call in engine.calls)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**Here's a sneak peek** of [top posts](https://np.reddit.com/r/x/top/?sort=top&amp;t=year)",
         "Here's a sneak peek of top posts"),
        (r"\#1: [Hi](http://a.b/c) ^^I'm ^^a ^^bot", "#1: Hi I'm a bot"),
        ("# Heading\n> quoted &amp; more", "Heading\nquoted & more"),
        ("2^3 = 8 and snake_case_name", "2^3 = 8 and snake_case_name"),
        ("<b>no es markdown</b>", "<b>no es markdown</b>"),
        ("I'm a bot, *bleep*, *bloop*. 2*3 and a * b", "I'm a bot, bleep, bloop. 2*3 and a * b"),
    ],
)
def test_clean_text_strips_reddit_markdown_but_keeps_plain_text(raw, expected):
    assert clean_text(raw) == expected


def test_highlight_spans_follow_the_same_preprocessing_and_visible_offsets(monkeypatch):
    def fake_preprocess(word, *, stem=False):
        if "-" in word:
            return []
        return [word.lower().removesuffix("s") if stem else word.lower()]

    monkeypatch.setattr("app.ui.preprocess", fake_preprocess)
    from app.ui import _word_terms

    _word_terms.cache_clear()
    text = "Invests & savings, but GOOG-stock and 1.000 stay."
    spans = highlight_spans(text, ("invest", "saving", "goog", "1"), stem=True)
    assert [text[a:b] for a, b in spans] == ["Invests", "savings"]
    assert spans == [[0, 7], [10, 17]]


def test_highlight_spans_use_utf16_offsets_and_cache_repeated_words(monkeypatch):
    from app.ui import _word_terms

    calls = []
    def fake_preprocess(word, *, stem=False):
        calls.append((word, stem))
        return [word]

    monkeypatch.setattr("app.ui.preprocess", fake_preprocess)
    _word_terms.cache_clear()
    assert highlight_spans("😀 Cat cat", ("cat",), stem=False) == [[3, 6], [7, 10]]
    assert calls.count(("cat", False)) == 1
    _word_terms.cache_clear()


def test_empty_query_does_not_search():
    engine = FakeEngine()
    assert respond(engine, DEFAULT_REQUEST) is None
    assert engine.calls == []


def test_engine_errors_become_a_readable_message_with_the_same_seq():
    engine = FakeEngine(ArtifactsMissingError("Faltan los archivos de BM25"))
    request = Request(q="x", model="bm25", k=10, stem=False, view="buscar", seq=9)
    response = respond(engine, request)
    assert response == {"seq": 9, "key": request.key, "error": "Faltan los archivos de BM25"}


# ---------- Interfaz (archivos del frontend) ----------

def test_assets_inline_the_logo_and_the_suggestions_safely():
    html, css, js = build_assets(('How should I invest?', 'quote " and </script>'))
    assert "{{LOGO}}" not in html and '<circle class="dot"' in html
    assert "__SUGGESTIONS__" not in js
    assert '"quote \\" and </script>"' in js  # JSON válido dentro del módulo JS
    assert css.strip()


def test_css_is_scoped_to_the_component():
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    # Ninguna regla global salvo tokens, animaciones y transiciones de vista
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    selectors = re.findall(r"(?m)^\s*([^@{}\n][^{}]*)\{", body)
    unscoped = [
        s for s in selectors
        if not any(part.strip().startswith((".jab", ":root", "::view-transition", "from", "to"))
                   for part in s.split(","))
    ]
    assert unscoped == []


def test_corpus_text_is_escaped_in_the_frontend():
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    # El texto, el id y los mensajes de error del servidor pasan por esc()
    assert "${marked(lead, h.spans, leadAt)}" in js
    assert "${marked(rest, h.spans, restAt)}" in js
    assert "<mark>${esc(text.slice(a - offset, b - offset))}</mark>" in js
    assert "${esc(h.doc_id)}" in js and "${esc(r.error)}" in js
    assert "${esc(scoreTip(r.model, h, r.terms || []))}" in js


def test_palette_is_ciruela_and_old_colors_are_gone():
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    assert "--accent: #7A2E5E" in css
    for old in ("#FF6A13", "#1D2433", "#B9440A", "data-palette"):
        assert old not in css


def test_wordmark_uses_current_color_and_one_dot():
    svg = WORDMARK.read_text(encoding="utf-8")
    assert 'stroke="currentColor"' in svg
    assert svg.count('class="dot"') == 1


# ---------- Contrato con #14 y #15 ----------

class RecordingEngine:
    def __init__(self):
        self.calls = []

    def search(self, model, query, k, *, stem):
        self.calls.append((model, query, k, stem))
        return "respuesta"


def test_context_search_uses_its_parameters_and_allows_another_model():
    engine = RecordingEngine()
    context = AppContext(engine=engine, query="gato", model="bm25", k=10, stem=True, reports_dir=Path("r"))
    assert context.search() == "respuesta"
    assert context.search("dense") == "respuesta"
    assert engine.calls == [("bm25", "gato", 10, True), ("dense", "gato", 10, True)]


def test_context_is_immutable():
    context = AppContext(engine=RecordingEngine(), query="", model="bm25", k=10, stem=False, reports_dir=Path("r"))
    with pytest.raises(AttributeError):
        context.k = 3


@pytest.mark.parametrize("view", [comparison, metrics])
def test_reserved_views_expose_render_with_the_agreed_contract(view):
    assert callable(view.render)
    assert list(view.render.__annotations__) == ["context", "return"]
