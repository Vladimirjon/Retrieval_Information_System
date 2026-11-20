import re
import shutil
import subprocess
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
    page_for,
    parse_request,
    parse_view,
    request_from_url,
    resolve_state,
    respond,
    same_state,
    url_params,
)
from app.views import comparison, metrics
from ris.engine import ArtifactsMissingError, Hit, ModelUnavailableError, SearchResponse
from ris.graph import GraphNode, QueryGraph

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


def test_graph_is_off_unless_the_browser_sends_true_and_changes_the_search_key():
    assert DEFAULT_REQUEST.graph is False
    assert parse_request({"q": "a"}).graph is False
    for value in (1, "true", "on", None, [True]):
        assert parse_request({"q": "a", "graph": value}).graph is False
    with_graph = parse_request({"q": "a", "graph": True})
    assert with_graph.graph is True
    assert with_graph.key != parse_request({"q": "a"}).key  # otra respuesta: lleva el grafo


# ---------- Respuesta ----------

class FakeEngine:
    def __init__(self, error=None, graph_error=None, no_hits=False, suggestion=None, extra=0):
        self.error = error
        self.graph_error = graph_error
        self.no_hits = no_hits
        self.suggestion = suggestion
        self.extra = extra  # resultados de más que devuelve si se piden más de 2
        self.calls = []

    def correct_query(self, query):
        if isinstance(self.suggestion, Exception):
            raise self.suggestion
        return self.suggestion

    def query_graph(self, query, doc_ids, linked=None):
        self.calls.append(("graph", query, list(doc_ids), linked))
        if self.graph_error:
            raise self.graph_error
        nodes = (GraphNode("501384", (0.123456, -0.5, 0.0), 0.987654),
                 GraphNode("315741", (1.0, 0.0, 0.0), 0.25))
        return QueryGraph(nodes=nodes, edges=((0, 1, 0.333333),), explained_variance=0.61234567)

    def search(self, model, query, k, *, stem, skip_empty=False):
        self.calls.append((model, query, k, stem, skip_empty))
        if self.error:
            raise self.error
        hits = (Hit(1, "501384", "", "Texto <b>uno</b>.", 10.5), Hit(2, "315741", "", "", 9.0))
        if self.no_hits:
            hits = ()
        elif k > 2:
            hits += tuple(Hit(3 + i, f"ctx{i}", "", f"**Contexto** {i}", 1.0) for i in range(self.extra))
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


def graph_request(**changes):
    values = {"q": "savings", "model": "bm25", "k": 5, "stem": False, "view": "buscar",
              "seq": 2, "graph": True, **changes}
    return Request(**values)


def test_graph_is_computed_only_when_requested():
    engine = FakeEngine()
    response = respond(engine, graph_request(graph=False))
    assert "graph" not in response
    assert not any(call[0] == "graph" for call in engine.calls)


def test_graph_travels_with_the_hits_in_the_same_order_and_rounded():
    engine = FakeEngine()
    response = respond(engine, graph_request())
    assert ("graph", "savings", ["501384", "315741"], 2) in engine.calls
    graph = response["graph"]
    assert graph["nodes"] == [[0.1235, -0.5, 0.0, 0.9877], [1.0, 0.0, 0.0, 0.25]]
    assert graph["edges"] == [[0, 1, 0.3333]]
    assert graph["variance"] == 0.6123
    assert graph["elapsed_ms"] >= 0
    assert len(graph["nodes"]) == len(response["hits"])


def test_graph_failure_keeps_the_results():
    engine = FakeEngine(graph_error=ModelUnavailableError("No se pudo calcular el grafo: x"))
    response = respond(engine, graph_request())
    assert response["graph"] == {"error": "No se pudo calcular el grafo: x"}
    assert len(response["hits"]) == 2 and "error" not in response


def test_no_hits_means_no_graph():
    engine = FakeEngine(no_hits=True)
    response = respond(engine, graph_request())
    assert response["hits"] == [] and "graph" not in response
    assert not any(call[0] == "graph" for call in engine.calls)


def test_empty_query_does_not_search():
    engine = FakeEngine()
    assert respond(engine, DEFAULT_REQUEST) is None
    assert engine.calls == []


def test_engine_errors_become_a_readable_message_with_the_same_seq():
    engine = FakeEngine(ArtifactsMissingError("Faltan los archivos de BM25"))
    request = Request(q="x", model="bm25", k=10, stem=False, view="buscar", seq=9)
    response = respond(engine, request)
    assert response == {"seq": 9, "key": request.key, "error": "Faltan los archivos de BM25",
                        "suggestion": None}


# ---------- Interfaz (archivos del frontend) ----------

def test_assets_inline_the_logo_and_the_suggestions_safely():
    html, css, js = build_assets(('How should I invest?', 'quote " and </script>'))
    assert "{{LOGO}}" not in html and '<circle class="dot"' in html
    assert "__SUGGESTIONS__" not in js
    assert '"quote \\" and </script>"' in js  # JSON válido dentro del módulo JS
    assert css.strip()


def test_css_is_scoped_to_the_component():
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    # Ninguna regla global salvo tokens, pasos de animación y transiciones de vista
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    selectors = re.findall(r"(?m)^\s*([^@{}\n][^{}]*)\{", body)
    unscoped = [
        s for s in selectors
        if not any(part.strip().startswith((".jab", ":root", "::view-transition", "from", "to"))
                   or re.fullmatch(r"\d+%", part.strip()) for part in s.split(","))
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
    # El grafo escribe textos del corpus solo con textContent o en el canvas
    assert "pointTip(target.edge ? edgeTip(target.edge) : nodeTip(target.node), point)" in js
    assert "tip.textContent = text" in js and "graphError.textContent" in js
    assert "${esc(part)}" in js  # «Quizás quisiste decir» también escapa


def test_graph_viewer_is_optional_and_draws_without_external_libraries():
    html, _, js = build_assets(())
    assert "data-graph-switch" in html and 'class="graph"' in html and "<canvas" in html
    assert 'aria-label="Grafo 3D de la consulta" hidden' in html  # oculto hasta que se pide
    assert "graph: state.graph" in js
    assert not re.search(r"^\s*import\s|https?://", js, flags=re.MULTILINE)
    assert "Grafo 3D de la consulta" in html and "Stemming (Porter)" in html
    assert "MiniLM" not in html + js  # la interfaz explica sin nombres internos
    assert "Raíz de palabras" not in html


def test_status_counts_the_visible_page_without_a_rank_range():
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    status = js.split("function renderStatus(", 1)[1].split("function renderPage(", 1)[0]
    assert 'Mostrando <b>${count}</b>' in status
    assert 'count === 1 ? "resultado" : "resultados"' in status
    assert 'de <b>${n}</b>' in status
    assert "first" not in status and "–" not in status
    assert "renderStatus(r, Math.min(PAGE_SIZE, n - first))" in js


def test_compare_without_query_reuses_the_home_examples_and_shows_progress():
    html, css, js = build_assets(())
    # Un solo juego de ejemplos: se mueve entre el inicio de Buscar y Comparar vacío
    assert html.count('class="example"') == 3 and "data-cmp-start" in html
    assert '$(state.view === "comparar" ? "[data-cmp-start]" : ".home").append(examples)' in js
    assert 'root.dataset.query = state.q.trim() ? "on" : "off"' in js
    assert '.jab[data-view="comparar"][data-query="off"] .cmp-start' in css
    # Mientras Python no pinta la comparación, la barra dice que está buscando
    assert 'class="cmp-pending" role="status"' in html
    assert '.jab[data-view="comparar"][data-query="on"] .cmp-pending' in css
    # Comparar y su cabecera comparten ancho; la barra conserva el de Buscar
    assert '.jab[data-view="comparar"] .jab-main { width: min(1120px' in css


def test_compare_waits_longer_and_search_never_shows_an_old_list_as_current():
    _, _, js = build_assets(())
    assert 'state.view === "comparar" ? COMPARE_DEBOUNCE : DEBOUNCE' in js
    compare = int(re.search(r"const COMPARE_DEBOUNCE = (\d+);", js).group(1))
    assert compare > int(re.search(r"const DEBOUNCE = (\d+);", js).group(1))
    # Volver a Buscar con una consulta editada en Comparar relanza la búsqueda (lista
    # atenuada y barra ocupada) en vez de mostrar la anterior como vigente.
    set_view = js.split("function setView(", 1)[1].split("root.querySelectorAll", 1)[0]
    assert 'view === "buscar" && state.searchedKey !== searchKey()) search();' in set_view
    assert "state.searchedKey = searchKey();" in js.split("function search(", 1)[1]


def test_model_menu_is_compact_but_keeps_touch_targets():
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    opt = css.split(".jab .opt {", 1)[1].split("}", 1)[0]
    setting = css.split(".jab .setting {", 1)[1].split("}", 1)[0]
    assert "min-height: 40px" in opt and "min-height: 40px" in setting
    assert "@media (pointer: coarse) { .jab .opt, .jab .setting, .jab .example { min-height: 44px; } }" in css


def test_query_is_a_bounded_textarea_with_the_combobox_contract():
    html, css, js = build_assets(())
    field = re.search(r'<textarea\b[^>]*id="jab-q"[^>]*></textarea>', html)
    assert field is not None and '<input id="jab-q"' not in html
    for attribute in ('rows="1"', 'maxlength="300"', 'role="combobox"',
                      'aria-autocomplete="both"', 'aria-expanded="false"',
                      'aria-controls="jab-suggest"', 'aria-describedby="jab-ghost-hint"'):
        assert attribute in field.group()
    assert 'input.setAttribute("aria-activedescendant", `jab-sg-${active}`)' in js
    assert 'input.removeAttribute("aria-activedescendant")' in js
    assert "input.setRangeText" in js and "input.maxLength" in js
    assert "ghost.scrollTop = input.scrollTop" in js
    assert "${esc(v)}</span>${esc(ghostRest)}" in js
    assert ".jab .bar textarea, .jab .ghost" in css
    assert "white-space: pre-wrap; overflow-wrap: anywhere" in css


def test_graph_legend_uses_actual_ranks_and_hides_absent_context():
    html, _, js = build_assets(())
    legend = html.split('class="graph-legend"', 1)[1].split("</ul>", 1)[0]
    assert "Consulta" in legend
    assert "Mostrados" not in legend and "Alrededor" not in legend
    assert "`Tu lista (1–${shown})`" in js
    assert "`Puestos ${shown + 1}–${d.nodes.length}`" in js
    assert '$("[data-legend-context]").hidden = around === 0' in js


def test_query_tooltip_does_not_quote_the_pca_variance_of_a_redrawn_map():
    # El mapa dibujado no es la proyección PCA tal cual (radio por puesto de parecido,
    # direcciones reequilibradas): citar su varianza conservada sería engañoso.
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    node_tip = js.split("function nodeTip(", 1)[1].split("function edgeTip(", 1)[0]
    assert "variance" not in node_tip
    assert "orden de parecido" in node_tip


def test_graph_has_no_planes_axes_or_grid_and_frames_the_cloud():
    _, css, js = build_assets(())
    for removed in ("drawPlanes", "planePoint", "--plane-", "Eje ", "quadrants"):
        assert removed not in css + js
    assert "g.layout.map((p) => Math.hypot(...p))" in js
    assert "frameGraph();" in js
    # Nodos planos: sin brillo, degradado metálico ni sombra (pedido del usuario).
    assert "createRadialGradient" not in js and "shadowBlur" not in js


def run_frontend_js(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node no disponible para ejecutar las funciones del frontend")
    subprocess.run([node, "--input-type=module", "-"], input=script, text=True,
                   encoding="utf-8", capture_output=True, check=True)


def frontend_function(name, next_name):
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    return f"function {name}(" + js.split(f"function {name}(", 1)[1].split(
        f"function {next_name}(", 1)[0]



def test_graph_layout_uses_real_dissimilarity_and_preserves_radii():
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "nodes.map((n) => 1 - n[3])" in js
    run_frontend_js("""
import assert from 'node:assert/strict';
const g = {yaw: 0.65, pitch: 0.45}, CAMERA = 4;
""" + frontend_function("projected", "project")
        + frontend_function("layoutGraph", "frameGraph") + """
const nodes = [[1, 0, 0, .9], [0, 1, 0, .5], [0, 0, 1, -.1]];
const original = JSON.stringify(nodes), positions = layoutGraph(nodes);
const radii = positions.map(p => Math.hypot(...p));
// Radio por puesto de disimilitud: el más parecido al centro, el menos al borde.
assert.ok(Math.abs(radii[0] - .22) < 1e-12);
assert.ok(Math.abs(radii[1] - (.22 + .78 * .5)) < 1e-12);
assert.ok(Math.abs(radii[2] - 1) < 1e-12);
// Muy parecidos entre sí no se amontonan: el orden manda, no la diferencia mínima.
const close = layoutGraph([[1, 0, 0, .91], [0, 1, 0, .9], [0, 0, 1, .89], [1, 1, 0, .2]]).map(p => Math.hypot(...p));
assert.deepEqual(close.map(r => +r.toFixed(6)), [.22, .48, .74, 1]);
// Menos sesgo de dirección: todos apuntaban a +x (componente media 0,958); tras quitar
// el 80 % de la dirección común queda en torno a 0,56 y la nube rodea más la consulta.
const skewed = layoutGraph(Array.from({length: 12}, (_, i) => [1, .3 * Math.cos(i), .3 * Math.sin(i), .9 - i * .05]));
const meanX = skewed.reduce((s, p) => s + p[0] / Math.hypot(...p), 0) / skewed.length;
assert.ok(meanX < .65);
assert.equal(JSON.stringify(nodes), original);
assert.deepEqual(layoutGraph(nodes), positions);
const coincident = Array.from({length: 50}, () => [0, 0, 0, .6]);
const fallback = layoutGraph(coincident);
assert.deepEqual(layoutGraph(coincident), fallback);
assert.equal(new Set(fallback.map(JSON.stringify)).size, 50);
assert.ok(fallback.every(p => p.every(Number.isFinite) && Math.abs(Math.hypot(...p) - .61) < 1e-12));
const crowded = layoutGraph(Array.from({length: 10}, () => [1, 0, 0, .6]));
assert.ok(crowded.every(p => Math.abs(Math.hypot(...p) - .61) < 1e-12));
assert.ok(crowded.some(p => Math.abs(p[1]) > 1e-6));
""")


def test_spin_pauses_without_changing_preference_and_focus_animates():
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    assert "setSpin(false)" not in js
    assert "g.spin && !reduced()" in js
    assert "now >= g.resumeAt" in js
    assert "!g.focusFrame" in js
    assert 'scrollIntoView({ block: "nearest"' in js
    assert "cancelFrame(g.focusFrame)" in js
    run_frontend_js("""
import assert from 'node:assert/strict';
let now = 100, motion = false, pending, draws = 0;
const performance = {now: () => now};
const reduced = () => motion;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const cancelFrame = () => { pending = null; };
const requestFrame = fn => { pending = fn; return 1; };
const drawGraph = () => { draws++; };
const projected = p => ({x:p[0], y:p[1]});
const g = {spin:true, layout:[[.5, .2, 0], [-.1, .4, 0]], zoom:1,
  panX:0, panY:0, centerX:0, centerY:0, scale:100, focusFrame:0};
""" + frontend_function("pauseSpin", "stopFocus")
        + frontend_function("stopFocus", "tick")
        + frontend_function("focusTarget", "zoomAt") + """
pauseSpin(); assert.equal(g.resumeAt, 2100); assert.equal(g.spin, true);
focusTarget({node:1}); assert.equal(g.zoom, 1); assert.equal(g.focusFrame, 1);
now = 550; pending(now);
assert.equal(g.zoom, 2); assert.equal(g.panX, -100); assert.equal(g.panY, -40);
assert.equal(g.focusFrame, 0); assert.equal(g.resumeAt, 2550);
assert.deepEqual(g.pin, {node:1});
motion = true;
focusTarget({edge:{a:0,b:1}});
assert.equal(g.panX, -40); assert.ok(Math.abs(g.panY + 60) < 1e-10);
assert.equal(g.focusFrame, 0); assert.equal(g.spin, true);
g.spin = false; pauseSpin(); assert.equal(g.spin, false);
""")


def test_spin_resumes_after_two_seconds_but_not_when_disabled_or_reduced():
    js = (FRONTEND / "app.js").read_text(encoding="utf-8")
    spin_source = 'const spinning = ' + js.split('const spinning = ', 1)[1].split(
        '  function setPanMode(', 1)[0]
    run_frontend_js("""
import assert from 'node:assert/strict';
let now = 100, motion = false, gesture = null, draws = 0, pending;
const performance = {now: () => now}, reduced = () => motion;
const graphBox = {hidden:false}, spinBtn = {setAttribute:() => {}};
const requestFrame = fn => {pending = fn; return 1;}, cancelFrame = () => {};
const drawGraph = () => {draws++;};
const g = {spin:true, data:{}, frame:0, last:100, yaw:0, hover:null,
  pin:{node:1}, focusFrame:0, visible:true};
""" + spin_source + """
pauseSpin(); tick(2099); assert.equal(g.yaw,0);
tick(2101); assert.ok(g.yaw > 0);  // el pin no bloquea la preferencia
const angle = g.yaw;
g.hover = {node:1}; tick(2200); assert.equal(g.yaw,angle);
g.hover = null; now = 2200; pauseSpin();
tick(4199); assert.equal(g.yaw,angle);
tick(4201); assert.ok(g.yaw > angle);
const resumed = g.yaw;
g.focusFrame = 3; tick(4300); assert.equal(g.yaw,resumed);
g.focusFrame = 0; gesture = {}; tick(4400); assert.equal(g.yaw,resumed);
gesture = null; setSpin(false); tick(10000); assert.equal(g.yaw,resumed);
setSpin(true); motion = true; tick(11000); assert.equal(g.yaw,resumed);
""")


def test_completion_keeps_lowercase_and_sizes_the_full_ghost():
    _, _, js = build_assets(())
    assert 'document.activeElement === input && atEnd' in js
    assert 'ghost.style.height = "auto"' in js
    assert 'Math.min(138, Math.max(input.scrollHeight, ghost.scrollHeight))' in js
    assert 'editInput(ghostRest, input.value.length, input.value.length)' in js
    run_frontend_js("""
import assert from 'node:assert/strict';
const SUGG = ['HELOC vs. Mortgage'], LOWER = SUGG.map(s => s.toLowerCase());
const VOCAB = ['investing'];
""" + frontend_function("completion", "sizeInput") + """
assert.equal('he' + completion('he'), 'heloc vs. mortgage');
assert.equal('HE' + completion('HE'), 'HELOC vs. Mortgage');
assert.equal('He' + completion('He'), 'HeLOC vs. Mortgage');
assert.equal('how to inv' + completion('how to inv'), 'how to investing');
""")


def test_programmatic_edits_use_native_undo_and_fallback_dispatches_input():
    _, _, js = build_assets(())
    assert 'editInput(dymBtn.dataset.q || "")' in js
    assert 'editInput(b.textContent.trim())' in js
    assert 'editInput("")' in js
    assert 'function pick(n) { const value = items[n]; editInput(value);' in js
    assert 'e.key.toLowerCase() === "z"' not in js
    preview = js.split('function markSuggest()', 1)[1].split('function pick(', 1)[0]
    assert 'input.value =' not in preview  # navegar no borra el historial nativo
    helper = js.split('  function editInput(', 1)[1].split('  // ---------- Entrada', 1)[0]
    run_frontend_js("""
import assert from 'node:assert/strict';
let native = true, events = 0, command;
const input = {value:'old query', focus:() => {},
  setSelectionRange(a,b) {this.start=a; this.end=b;},
  setRangeText(text,a,b) {this.value=this.value.slice(0,a)+text+this.value.slice(b);},
  dispatchEvent(e) {assert.equal(e.type,'input'); events++;}};
const document = {execCommand(cmd, ui, text) {
  command = cmd;
  if (native) {input.setRangeText(text,input.start,input.end); input.dispatchEvent(new Event('input'));}
  return native;
}};
""" + 'function editInput(' + helper + """
editInput(''); assert.equal(command,'delete'); assert.equal(input.value,'');
editInput('he'); editInput('loc',2,2); assert.equal(input.value,'heloc');
assert.equal(command,'insertText'); assert.equal(events,3);
native = false; editInput('other'); assert.equal(input.value,'other'); assert.equal(events,4);
editInput('',2,2); assert.equal(input.value,'other'); assert.equal(events,4);
""")


def test_graph_cursor_is_centralized_and_feedback_is_not_interactive():
    html, css, js = build_assets(())
    assert js.count("canvas.style.cursor =") == 1
    assert 'canvas.style.cursor = ""' not in js
    assert '"move" : "grabbing"' in js and '"pointer" : "grab"' in js
    assert 'on(canvas, "lostpointercapture"' in js
    assert "canvas.releasePointerCapture" in js
    assert 'class="graph-feedback" aria-live="polite"' in html
    feedback = css.split(".jab .graph-feedback {", 1)[1].split("}", 1)[0]
    assert "pointer-events: none" in feedback
    assert 'feedback.classList.remove("visible"), 600)' in js


def test_assets_inline_the_vocabulary_for_grey_completion():
    _, _, js = build_assets(("a",), ("investment", "</script>"))
    assert "__VOCABULARY__" not in js
    assert '["investment", "</script>"]' in js


def test_numbers_do_not_show_a_help_cursor():
    css = (FRONTEND / "styles.css").read_text(encoding="utf-8")
    assert "cursor: help" not in css


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



# ---------- URL: rutas y parámetros ----------

@pytest.mark.parametrize(("page", "view"), [
    ("home", "buscar"), ("search", "buscar"), ("comparison", "comparar"),
    ("metrics", "metricas"), ("otra", "buscar"),
])
def test_each_route_opens_its_view(page, view):
    assert request_from_url(page, {}).view == view


def test_url_params_are_read_and_validated():
    request = request_from_url("search", {
        "q": "  how   to invest ", "model": "Embeddings", "results": "20",
        "stemming": "true", "graph": "TRUE",
    }, seq=4)
    assert request == Request(q="how to invest", model="dense", k=20, stem=False,
                              view="buscar", seq=4, graph=True,
                              route="search")  # Porter no aplica a Embeddings
    bad = request_from_url("search", {"model": "lsi", "results": "7", "stemming": "yes",
                                      "graph": "1", "q": ["x"]})
    assert (bad.model, bad.k, bad.stem, bad.graph, bad.q) == ("bm25", 10, False, False, "")


@pytest.mark.parametrize("results", ["²", "٣", "9" * 5000, " 20", "20.0", "-5", "", None, 20])
def test_odd_results_values_fall_back_to_ten_without_crashing(results):
    # "²".isdigit() es True pero int("²") falla: un enlace así no debe romper la app.
    assert request_from_url("search", {"results": results}).k == 10


def test_route_follows_the_view_and_what_the_interface_chose():
    assert page_for(Request("", "bm25", 10, False, "buscar", 0)) == "home"
    # Escribir en el inicio no cambia de página (se perdería el foco al volver a montar).
    assert page_for(Request("x", "bm25", 10, False, "buscar", 0)) == "home"
    assert page_for(Request("x", "bm25", 10, False, "buscar", 0, route="search")) == "search"
    # Borrar en /search tampoco: se queda en /search hasta pulsar el logo.
    assert page_for(Request("", "bm25", 10, False, "buscar", 0, route="search")) == "search"
    assert page_for(Request("", "bm25", 10, False, "comparar", 0)) == "comparison"
    assert page_for(Request("x", "bm25", 10, False, "metricas", 0, route="search")) == "metrics"


def test_route_from_the_browser_is_whitelisted():
    assert parse_request({"q": "x", "route": "search"}).route == "search"
    for bad in ("metrics", "/admin", None, 1, True):
        assert parse_request({"q": "x", "route": bad}).route == "home"
    assert request_from_url("search", {}).route == "search"
    assert request_from_url("home", {"q": "x"}).route == "home"


def test_url_shows_the_chosen_options():
    request = Request("index funds", "tfidf", 20, True, "buscar", 3, graph=True)
    assert url_params(request) == {"q": "index funds", "model": "tfidf", "results": "20",
                                   "stemming": "true", "graph": "true"}
    dense = url_params(Request("x", "dense", 10, False, "comparar", 0))
    assert dense == {"q": "x", "model": "embeddings", "results": "10", "graph": "false"}


def test_home_url_only_keeps_what_differs_from_the_defaults():
    assert url_params(DEFAULT_REQUEST) == {}
    assert url_params(Request("", "dense", 10, False, "buscar", 0, graph=True)) == {
        "model": "embeddings", "graph": "true"}


@pytest.mark.parametrize("request_", [
    Request("index funds", "tfidf", 20, True, "buscar", 0, graph=True),
    Request("x", "dense", 5, False, "comparar", 0),
    Request("", "jaccard", 50, True, "metricas", 0),
    DEFAULT_REQUEST,
])
def test_url_round_trip_keeps_the_state(request_):
    back = request_from_url(page_for(request_), url_params(request_), seq=request_.seq)
    assert back == request_ and same_state(back, request_)


def test_same_state_ignores_seq_but_not_view_or_options():
    a = Request("x", "bm25", 10, False, "buscar", 1)
    assert same_state(a, Request("x", "bm25", 10, False, "buscar", 9))
    assert not same_state(a, Request("x", "bm25", 10, False, "comparar", 1))
    assert not same_state(a, Request("x", "bm25", 10, False, "buscar", 1, graph=True))


# ---------- Quizás quisiste decir y contexto del grafo ----------

def test_suggestion_travels_with_results_errors_and_never_breaks_search():
    response = respond(FakeEngine(suggestion="what is a mortgage"), graph_request(graph=False))
    assert response["suggestion"] == "what is a mortgage" and len(response["hits"]) == 2
    error = respond(FakeEngine(error=ArtifactsMissingError("x"), suggestion="y"), graph_request())
    assert error["suggestion"] == "y" and error["error"] == "x"
    broken = respond(FakeEngine(suggestion=RuntimeError("boom")), graph_request(graph=False))
    assert broken["suggestion"] is None and len(broken["hits"]) == 2


def test_graph_places_up_to_fifty_results_but_only_lists_the_chosen_k():
    engine = FakeEngine(extra=3)
    response = respond(engine, graph_request(k=5))
    search = next(call for call in engine.calls if call[0] == "bm25")
    assert search[2] == 50  # pide los 50 para situar el contexto
    assert len(response["hits"]) == 5
    graph_call = next(call for call in engine.calls if call[0] == "graph")
    assert graph_call[3] == 5 and len(graph_call[2]) == 5
    without_graph = FakeEngine(extra=3)
    respond(without_graph, graph_request(k=5, graph=False))
    assert next(call for call in without_graph.calls if call[0] == "bm25")[2] == 5


def test_graph_context_is_cleaned_and_ranked():
    engine = FakeEngine(extra=3)
    engine.query_graph = lambda query, doc_ids, linked=None: _graph_for(doc_ids)
    response = respond(engine, graph_request(k=2))
    graph = response["graph"]
    assert graph["shown"] == 2
    assert graph["context"] == [{"rank": 3 + i, "doc_id": f"ctx{i}", "text": f"Contexto {i}"}
                                for i in range(3)]
    assert len(graph["nodes"]) == 5


def _graph_for(doc_ids):
    nodes = tuple(GraphNode(doc_id, (0.0, 0.0, 0.0), 0.5) for doc_id in doc_ids)
    return QueryGraph(nodes=nodes, edges=(), explained_variance=1.0)


# ---------- Estado vigente: navegador frente a URL ----------

SEARCH_A = {"q": "a", "model": "bm25", "results": "10", "stemming": "false", "graph": "false"}


def test_first_load_takes_the_url_and_announces_it():
    state, nav, seen = resolve_state(None, None, "search", SEARCH_A, None, 0)
    assert (state.q, state.view, state.seq, nav) == ("a", "buscar", 0, 1)
    assert seen == ("search", state.key, "buscar", "search")


def test_an_interface_action_wins_over_the_url():
    current = request_from_url("search", SEARCH_A, seq=3)
    sent = Request("b", "bm25", 10, False, "metricas", 4)
    state, nav, _ = resolve_state(current, sent, "search", SEARCH_A, None, 2)
    assert state == sent and nav == 2  # la URL vieja no lo deshace


def test_a_stale_url_from_an_in_flight_run_is_not_adopted():
    # La app ya pasó a q="" y pidió ir a "/"; llega una ejecución con la URL anterior.
    _, _, seen = resolve_state(None, None, "search", SEARCH_A, None, 0)
    current = Request("", "bm25", 10, False, "buscar", 5)
    state, nav, _ = resolve_state(current, current, "search", SEARCH_A, seen, 1)
    assert (state.q, state.seq, nav) == ("", 5, 1)


def test_back_button_adopts_the_previous_url_with_a_new_seq():
    _, _, seen_metrics = resolve_state(None, None, "metrics", SEARCH_A, None, 0)
    current = request_from_url("metrics", SEARCH_A, seq=7)
    state, nav, seen = resolve_state(current, None, "search", SEARCH_A, seen_metrics, 3)
    assert (state.view, state.q, state.seq, nav) == ("buscar", "a", 8, 4)
    assert seen != seen_metrics
