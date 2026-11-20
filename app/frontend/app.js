// jab — interfaz de búsqueda (componente de Streamlit, issue #13).
// El navegador gestiona escritura, menús y animación; Python ejecuta la búsqueda.
// Contrato con Python (app/ui.py):
//   estado "request" → {q, model, k, stem, graph, view, seq}; cada cambio sube seq (ninguno se pierde)
//   data.initial     ← estado vigente en Python (incluye su seq). `nav` sube cuando ese estado llegó
//                      desde la URL (recargar, atrás, adelante, un enlace): entonces la interfaz lo adopta.
//   data.response    ← {seq, key, model, stem, elapsed_ms, terms, suggestion, hits:[{rank, doc_id, text, score, matched, spans}]}
//                      o {seq, key, error, suggestion}
//   response.graph   ← solo con graph activo: {nodes:[[x, y, z, coseno]], edges:[[i, j, coseno]], shown,
//                      context:[{rank, doc_id, text}], variance, elapsed_ms} o {error}

const SUGG = __SUGGESTIONS__;
const LOWER = SUGG.map((s) => s.toLowerCase());
const VOCAB = __VOCABULARY__;
const MODELS = [
  { id: "jaccard", name: "Jaccard", hint: "Palabras en común", tip: "Compara los conjuntos de palabras distintas; 1 significa que coinciden por completo." },
  { id: "tfidf", name: "TF-IDF", hint: "Palabras ponderadas por rareza", tip: "Compara palabras y da más peso a las menos frecuentes en FiQA." },
  { id: "bm25", name: "BM25", hint: "Ponderadas y ajustadas por longitud", tip: "Suma aportes de palabras y ajusta repeticiones y largo del documento." },
  { id: "dense", name: "Embeddings", hint: "Por significado, no por palabras", tip: "Convierte la consulta y cada documento en un vector que resume su significado: encuentra textos parecidos aunque usen otras palabras." },
];
const KS = [5, 10, 20, 50];
const DEBOUNCE = 220;
const COMPARE_DEBOUNCE = 450;
const PAGE_SIZE = 5;
const CORPUS_SIZE = "57 638";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const decimal = (v) => v.toFixed(2).replace(".", ",");
const listEs = (items) => items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} y ${items[items.length - 1]}`;
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const CLOCK = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="6"/><path d="M8 4.8V8l2.2 1.4"/></svg>';

export default function ({ parentElement, data, setStateValue }) {
  const root = parentElement.querySelector(".jab");
  root._send = setStateValue;
  if (!root._jab) root._jab = mount(root, data || {});
  root._jab.receive(data || {});
}

function mount(root, data) {
  const $ = (s) => root.querySelector(s);
  const init = data.initial || {};
  const state = {
    q: init.q || "", model: init.model || "bm25", k: init.k || 10, stem: !!init.stem, graph: !!init.graph,
    view: init.view || "buscar", route: init.route || "home", seq: init.seq || 0, nav: init.nav || 0, timer: null, hasHits: false,
    renderedKey: null, page: 1, response: null, pageTimer: null,
  };
  const bar = $(".bar"), input = $("#jab-q"), clear = $(".clear"), chip = $(".chip"), menu = $("#jab-menu");
  const suggest = $("#jab-suggest"), list = $(".list"), status = $(".status"), pager = $(".pager"), tip = $("#jab-tip");
  const sw = $("[data-stem-row] .switch"), gsw = $("[data-graph-switch]"), ghost = $(".ghost"), dym = $(".dym"), dymBtn = $("[data-dym]");
  const send = (name, value) => root._send(name, value);

  // Una sola señal de cancelación para los oyentes globales si el componente se vuelve a montar.
  window.__jabTipCleanup?.();
  window.__jabListeners?.abort();
  const listeners = (window.__jabListeners = new AbortController());
  const on = (target, type, fn, options = {}) => target.addEventListener(type, fn, { signal: listeners.signal, ...options });
  const frames = new Set();
  function requestFrame(callback) {
    const id = requestAnimationFrame((now) => {
      frames.delete(id);
      if (!listeners.signal.aborted) callback(now);
    });
    frames.add(id);
    return id;
  }
  function cancelFrame(id) { cancelAnimationFrame(id); frames.delete(id); }
  listeners.signal.addEventListener("abort", () => {
    frames.forEach(cancelAnimationFrame); frames.clear();
  });


  const logoTpl = root.querySelector("[data-logo-svg]");
  root.querySelectorAll("[data-logo]").forEach((el) => el.append(logoTpl.content.cloneNode(true)));
  try {  // la entrada del punto del logo ocurre una vez por sesión del navegador
    if (sessionStorage.getItem("jab-dot")) root.querySelector(".logo.first")?.classList.remove("first");
    else sessionStorage.setItem("jab-dot", "1");
  } catch { /* sin almacenamiento: animación normal */ }

  // El punto del logo salta al buscar, al cambiar de modelo o de vista y al pulsarlo.
  function hop() {
    if (reduced()) return;
    root.querySelectorAll(".logo .dot").forEach((dot) => {
      dot.classList.remove("hop"); dot.getBoundingClientRect(); dot.classList.add("hop");
    });
  }
  on(root, "animationend", (e) => { if (e.animationName === "jab-hop") e.target.classList.remove("hop"); });

  // ---------- Modo: inicio ↔ resultados ----------
  let moving = false;
  const placeBar = (mode) => root.querySelector(`[data-slot="${mode}"]`).append(bar);
  // Los mismos tres ejemplos sirven al inicio de Buscar y al estado vacío de Comparar.
  const examples = $(".examples");
  const placeExamples = () => $(state.view === "comparar" ? "[data-cmp-start]" : ".home").append(examples);
  function setMode(mode) {
    if (root.dataset.mode === mode) return;
    const focused = document.activeElement === input, reopen = !suggest.hidden;
    const apply = () => {
      moving = true;
      // Mover la barra en el DOM le quita el foco: se mira otra vez justo antes, porque
      // el campo pudo recibirlo después de pedir el cambio (p. ej. tras pulsar ×).
      const keep = focused || document.activeElement === input;
      root.dataset.mode = mode; placeBar(mode);
      if (keep) input.focus({ preventScroll: true });
      if (reopen) showSuggest(input.value);
      moving = false;
      updateGhost();
    };
    if (document.startViewTransition && !reduced()) document.startViewTransition(apply);
    else apply();
  }

  // ---------- Búsqueda: Python responde con el mismo seq ----------
  // Búsqueda que refleja la lista de Buscar; cambia al editar la consulta o las opciones.
  const searchKey = () => JSON.stringify([state.q.trim(), state.model, state.k, state.stem, state.graph]);
  // Cada petición actualiza si hay consulta: Comparar sin consulta muestra los ejemplos.
  const currentRequest = () => {
    root.dataset.query = state.q.trim() ? "on" : "off";
    return { q: state.q.trim(), model: state.model, k: state.k, stem: state.stem, graph: state.graph, view: state.view, route: state.route, seq: state.seq };
  };
  function search() {
    clearTimeout(state.timer);
    clearTimeout(state.pageTimer);
    list.classList.remove("leaving");
    state.searchedKey = searchKey();
    const q = state.q.trim();
    state.seq += 1;
    state.page = 1; state.renderedKey = null;
    send("request", currentRequest());
    if (!q) { state.hasHits = false; state.renderedKey = null; state.response = null; list.innerHTML = ""; list.dataset.stale = "false"; status.innerHTML = ""; pager.hidden = true; dym.hidden = true; bar.dataset.busy = "false"; showGraph(null); if (isHome()) setMode("home"); return; }
    setMode("results");
    bar.dataset.busy = "true";
    graphBox.dataset.stale = "true";
    if (state.hasHits) list.dataset.stale = "true"; else renderSkeleton();
  }
  // Comparar ejecuta los cuatro modelos: espera a una pausa más larga antes de pedirlos.
  const schedule = () => { clearTimeout(state.timer); state.timer = setTimeout(search, state.view === "comparar" ? COMPARE_DEBOUNCE : DEBOUNCE); };
  const searchNow = () => { hop(); search(); };
  // Diseño de inicio (logo grande): Buscar sin consulta y fuera de /search.
  const isHome = () => state.view === "buscar" && !state.q.trim() && state.route !== "search";
  // Acción explícita (Enter, sugerencia, ejemplo, «Quizás quisiste decir»): pasa a /search.
  // Ese cambio de página vuelve a montar el componente; el aviso devuelve el foco al campo.
  const commitSearch = () => {
    if (state.route !== "search" && state.q.trim()) { state.route = "search"; window.__jabRefocus = true; }
    searchNow();
  };

  function split(text) {
    const start = text.search(/\S/);
    if (start < 0) return { lead: "Documento sin texto", rest: "", leadAt: 0, restAt: 0 };
    const t = text.slice(start).trimEnd();
    const m = t.match(/^([\s\S]{20,140}?[.?!])\s/);
    if (m) {
      const restAt = start + m[1].length + (t.slice(m[1].length).match(/^\s*/)?.[0].length || 0);
      return { lead: m[1], rest: text.slice(restAt).trimEnd(), leadAt: start, restAt };
    }
    if (t.length <= 110) return { lead: t, rest: "", leadAt: start, restAt: 0 };
    const boundary = t.slice(0, 101).lastIndexOf(" ");
    const cut = boundary > 20 ? boundary : 100;
    const restAt = start + cut + (t.slice(cut).match(/^\s*/)?.[0].length || 0);
    return { lead: t.slice(0, cut).trimEnd() + "…", rest: text.slice(restAt).trimEnd(), leadAt: start, restAt };
  }
  function marked(text, spans, offset) {
    const end = offset + text.length, parts = [];
    let cursor = offset;
    for (const [a, b] of spans || []) {
      if (a < cursor || b > end || a >= b) continue;
      parts.push(esc(text.slice(cursor - offset, a - offset)), `<mark>${esc(text.slice(a - offset, b - offset))}</mark>`);
      cursor = b;
    }
    parts.push(esc(text.slice(cursor - offset)));
    return parts.join("");
  }
  function scoreTip(model, h, terms) {
    const value = decimal(h.score);
    const sought = terms.length ? ` Se buscó: ${terms.join(", ")}.` : " No quedaron términos tras el preprocesamiento.";
    if (model === "bm25") return `${value}: suma de lo que aportan ${listEs(terms) || "los términos"} en este documento. No tiene tope; solo sirve para comparar resultados de esta misma búsqueda.${sought}`;
    if (model === "tfidf") return `${value}: coseno entre consulta y documento, de 0 a 1. Las palabras raras en la colección pesan más.${sought}`;
    if (model === "jaccard") return `${value}: el ${(h.score * 100).toLocaleString("es", { maximumFractionDigits: 0 })} % de las palabras distintas de consulta y documento son comunes.${sought}`;
    return `${value}: cuánto se parece el significado de tu consulta al de este documento, de −1 a 1 (similitud coseno entre sus vectores de significado). Más alto, más parecido; no es un porcentaje de relevancia.`;
  }
  function formatMs(ms) {
    if (ms < 1) return "menos de 1 ms";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
  }
  function renderStatus(r, count) {
    const n = r.hits.length, name = MODELS.find((m) => m.id === r.model)?.name || r.model;
    const shown = `Mostrando <b>${count}</b> ${count === 1 ? "resultado" : "resultados"} de <b>${n}</b>`;
    status.innerHTML = `<span>${shown}</span><span class="time tip-target" tabindex="0" data-tip="${esc(`Lo que tardó ${name} en ordenar los ${CORPUS_SIZE} documentos de FiQA.`)}">${CLOCK}${formatMs(r.elapsed_ms)}</span>`;
  }
  function renderPage(r) {
    closeTip();
    const n = r.hits.length, pages = Math.ceil(n / PAGE_SIZE);
    state.page = Math.min(state.page, pages);
    const first = (state.page - 1) * PAGE_SIZE;
    renderStatus(r, Math.min(PAGE_SIZE, n - first));
    const top = r.hits[0].score;
    list.innerHTML = r.hits.slice(first, first + PAGE_SIZE).map((h, i) => {
      const { lead, rest, leadAt, restAt } = split(h.text);
      const pct = top > 0 ? Math.max(0, Math.min(100, (h.score / top) * 100)) : 0;
      return `<li class="row" style="--i:${i}" data-rank="${h.rank}" ${h.rank === 1 ? 'data-top="true"' : ""}>
        <span class="rank num">${h.rank}</span>
        <div>
          <p class="lead" lang="en">${marked(lead, h.spans, leadAt)}</p>
          ${rest ? `<p class="snippet" lang="en">${marked(rest, h.spans, restAt)}</p>` : ""}
          <div class="meta">
            <span class="score tip-target" tabindex="0" data-tip="${esc(scoreTip(r.model, h, r.terms || []))}">${h.score.toFixed(2)}<span class="score-bar" aria-hidden="true"><i style="width:${pct.toFixed(0)}%"></i></span></span>
            <span class="tip-target" tabindex="0" data-tip="Identificador del documento en FiQA.">Doc. ${esc(h.doc_id)}</span>
            ${rest.length > 160 ? `<button class="more" type="button" aria-expanded="false">Leer todo</button>` : ""}
          </div>
        </div>
      </li>`;
    }).join("");
    pager.hidden = pages <= 1;
    pager.innerHTML = pages <= 1 ? "" : `<button type="button" data-action="previous" data-page="${state.page - 1}" aria-label="Página anterior" ${state.page === 1 ? "disabled" : ""}>Anterior</button>${Array.from({ length: pages }, (_, i) => `<button type="button" data-page="${i + 1}" ${state.page === i + 1 ? 'aria-current="page"' : ""} aria-label="Página ${i + 1}">${i + 1}</button>`).join("")}<button type="button" data-action="next" data-page="${state.page + 1}" aria-label="Página siguiente" ${state.page === pages ? "disabled" : ""}>Siguiente</button>`;
    drawGraph();  // el grafo marca los resultados de la página visible
  }
  // «Quizás quisiste decir»: las palabras corregidas van en negrita.
  function renderSuggestion(r) {
    const fixed = r.suggestion, typedQ = state.q.trim();
    dym.hidden = !fixed || fixed.toLowerCase() === typedQ.toLowerCase();
    if (dym.hidden) return;
    const before = typedQ.split(/([A-Za-z]+)/), after = fixed.split(/([A-Za-z]+)/);
    dymBtn.innerHTML = after.map((part, i) => part !== before[i] ? `<b>${esc(part)}</b>` : esc(part)).join("");
    dymBtn.dataset.q = fixed;
  }
  on(dymBtn, "click", () => {
    editInput(dymBtn.dataset.q || ""); syncClear(); hideSuggest(); commitSearch(); input.focus();
  });
  function render(r) {
    clearTimeout(state.pageTimer);
    list.classList.remove("leaving"); closeTip();
    bar.dataset.busy = "false"; list.dataset.stale = "false";
    state.response = r;
    renderSuggestion(r);
    showGraph(r);
    if (r.error) { state.hasHits = false; status.innerHTML = ""; pager.hidden = true; list.innerHTML = `<li class="empty"><p>${esc(r.error)}</p></li>`; return; }
    const model = MODELS.find((m) => m.id === r.model), name = model?.name || r.model;
    if (!r.hits.length) {
      state.hasHits = false; status.innerHTML = ""; pager.hidden = true;
      const alt = r.model === "dense" ? "" : `<button type="button" data-try-dense>Probar con Embeddings</button>`;
      list.innerHTML = `<li class="empty"><p>Sin resultados con ${esc(name)}.</p>${alt}</li>`;
      return;
    }
    state.hasHits = true;
    renderPage(r);
  }
  function renderSkeleton() {
    state.renderedKey = null;
    list.dataset.stale = "false";
    status.innerHTML = "";
    pager.hidden = true;
    list.innerHTML = Array.from({ length: 4 }, () =>
      `<li class="row skeleton" aria-hidden="true"><span class="rank">0</span><div><p class="lead">.</p><p class="snippet">.</p></div></li>`).join("");
  }
  on(list, "click", (e) => {
    const more = e.target.closest(".more");
    if (more) {
      const row = more.closest(".row"), open = row.classList.toggle("open");
      more.textContent = open ? "Mostrar menos" : "Leer todo"; more.setAttribute("aria-expanded", open);
    }
    if (e.target.closest("[data-try-dense]")) setModel("dense");
  });
  on(pager, "click", (e) => {
    const button = e.target.closest("[data-page]");
    if (!button || button.disabled || !state.response?.hits || bar.dataset.busy === "true" || list.dataset.stale === "true") return;
    const page = Number(button.dataset.page);
    if (page === state.page) return;
    list.classList.add("leaving");
    const keepFocus = document.activeElement === button;
    const action = button.dataset.action;
    const change = () => {
      if (bar.dataset.busy === "true" || list.dataset.stale === "true") return;
      state.page = page; renderPage(state.response); list.classList.remove("leaving");
      if (keepFocus) {
        const equivalent = action ? pager.querySelector(`[data-action="${action}"]:not(:disabled)`) : pager.querySelector(`[data-page="${page}"]:not([data-action])`);
        (equivalent || pager.querySelector('[aria-current="page"]'))?.focus();
      }
    };
    if (reduced()) change();
    else state.pageTimer = setTimeout(change, 120);
  });

  // Burbuja única: su posición fija no desplaza resultados ni menús.
  let tipDelay, tipHide, tipTarget;
  window.__jabTipCleanup = () => {
    clearTimeout(tipDelay); clearTimeout(tipHide);
    tipTarget?.removeAttribute("aria-describedby");
    tip.hidden = true;
  };
  function closeTip() {
    clearTimeout(tipDelay); clearTimeout(tipHide);
    tipTarget?.removeAttribute("aria-describedby"); tipTarget = null;
    tip.classList.remove("visible");
    tipHide = setTimeout(() => { tip.hidden = true; }, 140);
  }
  // A la derecha del control; si no cabe, a su izquierda.
  function placeTip(rect) {
    const box = tip.getBoundingClientRect();
    const right = rect.right + 10, left = right + box.width + 8 <= window.innerWidth ? right : rect.left - box.width - 10;
    tip.style.left = `${Math.max(8, Math.min(left, window.innerWidth - box.width - 8))}px`;
    tip.style.top = `${Math.max(8, Math.min(rect.top, window.innerHeight - box.height - 8))}px`;
  }
  function openTip(target) {
    if (target === tipTarget) return;
    closeTip(); tipTarget = target;
    tip.textContent = target.dataset.tip;
    tipDelay = setTimeout(() => {
      if (tipTarget !== target) return;
      clearTimeout(tipHide);
      tip.hidden = false;
      target.setAttribute("aria-describedby", "jab-tip");
      placeTip(target.getBoundingClientRect());
      requestFrame(() => tip.classList.add("visible"));
    }, 300);
  }
  // Burbuja que sigue al puntero (nodos y relaciones del grafo).
  function pointTip(text, point) {
    closeTip(); clearTimeout(tipHide);
    tip.textContent = text; tip.hidden = false;
    placeTip({ left: point.clientX - 14, right: point.clientX + 4, top: point.clientY + 14 });
    tip.classList.add("visible");
  }
  on(root, "pointerover", (e) => { const target = e.target.closest("[data-tip]"); if (target) openTip(target); });
  on(root, "pointerout", (e) => { if (tipTarget?.contains(e.target) && !tipTarget.contains(e.relatedTarget)) closeTip(); });
  let suppressMenuFocusTip = false;
  on(root, "focusin", (e) => { const target = e.target.closest("[data-tip]"); if (target && !suppressMenuFocusTip) openTip(target); });
  on(root, "focusout", (e) => { if (tipTarget?.contains(e.target)) closeTip(); });
  on(document, "keydown", (e) => { if (e.key === "Escape" && tipTarget) closeTip(); });

  // ---------- Grafo 3D de la consulta (canvas propio, sin bibliotecas) ----------
  // Python envía las posiciones (la consulta en el origen) y aquí se gira, se mueve, se acerca y se pinta.
  const graphBox = $(".graph"), stage = $(".graph-stage"), canvas = stage.querySelector("canvas"), graphError = $(".graph-error");
  const spinBtn = $("[data-graph-spin]"), panBtn = $("[data-graph-pan]"), ctx = canvas.getContext("2d");
  const YAW = 0.65, PITCH = 0.45, CAMERA = 4;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  // hover/pin: {node: índice en pts} o {edge: {a, b, sim}}; pts[0] es la consulta y pts[i + 1] el documento i.
  const g = {
    data: null, yaw: YAW, pitch: PITCH, zoom: 1, panX: 0, panY: 0, spin: !reduced(), panMode: false,
    hover: null, pin: null, frame: 0, last: 0, visible: true, w: 0, h: 0, pts: [], order: [], edges: [], layout: [], resumeAt: 0, focusFrame: 0, scale: 1, centerX: 0, centerY: 0,
  };
  const shownCount = () => g.data ? Math.min(g.data.shown ?? state.response.hits.length, g.data.nodes.length) : 0;
  function projected(p) {
    const cy = Math.cos(g.yaw), sy = Math.sin(g.yaw), cp = Math.cos(g.pitch), sp = Math.sin(g.pitch);
    const x = p[0] * cy + p[2] * sy, depth = p[2] * cy - p[0] * sy;
    const y = p[1] * cp - depth * sp, z = p[1] * sp + depth * cp;
    const s = CAMERA / (CAMERA - z);
    return { x: x * s, y: -y * s, z, s };
  }
  function project(p) {
    const v = projected(p), r = g.scale * g.zoom;
    return { ...v, x: g.w / 2 + g.panX + (v.x - g.centerX) * r, y: g.h / 2 + g.panY + (v.y - g.centerY) * r };
  }
  // Conserva la dirección PCA; el radio representa la disimilitud real con la consulta.
  function layoutGraph(nodes) {
    const dissimilarities = nodes.map((n) => 1 - n[3]);
    // Radio según el puesto de disimilitud (empates: puesto medio). Conserva el orden
    // —más lejos, menos parecido— y reparte los nodos en vez de amontonarlos junto a la
    // consulta cuando casi todos se parecen mucho.
    const order = dissimilarities.map((d, i) => [d, i]).sort((a, b) => a[0] - b[0]), rank = [];
    for (let s = 0; s < order.length;) {
      let e = s; while (e + 1 < order.length && order[e + 1][0] === order[s][0]) e++;
      for (let k = s; k <= e; k++) rank[order[k][1]] = (s + e) / 2;
      s = e + 1;
    }
    const radii = rank.map((r) => 0.22 + 0.78 * (nodes.length > 1 ? r / (nodes.length - 1) : 0.5));
    // Direcciones PCA sin su sesgo común: la nube rodea la consulta en vez de caer a un lado.
    const unit = nodes.map((n) => { const norm = Math.hypot(...n.slice(0, 3)); return norm > 1e-6 ? n.slice(0, 3).map((v) => v / norm) : null; });
    const valid = unit.filter(Boolean), mean = [0, 1, 2].map((k) => valid.reduce((sum, u) => sum + u[k], 0) / (valid.length || 1));
    const positions = nodes.map((n, i) => {
      const z = 1 - 2 * (i + 0.5) / nodes.length, angle = i * 2.399963229728653;
      const balanced = unit[i] && unit[i].map((v, k) => v - 0.8 * mean[k]), norm = balanced ? Math.hypot(...balanced) : 0;
      const direction = norm > 1e-6 ? balanced.map((v) => v / norm)
        : [Math.sqrt(1 - z * z) * Math.cos(angle), Math.sqrt(1 - z * z) * Math.sin(angle), z];
      return direction.map((v) => v * radii[i]);
    });
    // Separa solapamientos en la vista inicial con pequeños impulsos 3D tangenciales.
    // Renormalizar conserva exactamente el radio (también para posiciones PCA nulas).
    for (let iteration = 0; iteration < 5; iteration++) {
      const projectedPoints = positions.map(projected);
      const pushes = positions.map(() => [0, 0, 0]);
      for (let i = 0; i < positions.length; i++) for (let j = i + 1; j < positions.length; j++) {
        const dx = projectedPoints[i].x - projectedPoints[j].x, dy = projectedPoints[i].y - projectedPoints[j].y;
        const distance = Math.hypot(dx, dy);
        if (distance >= 0.14) continue;
        const angle = (i + j + 1) * 2.399963229728653;
        const ux = distance > 1e-6 ? dx / distance : Math.cos(angle);
        const uy = distance > 1e-6 ? dy / distance : Math.sin(angle);
        const strength = (0.14 - distance) * 0.12;
        const impulse = [Math.cos(g.yaw) * ux - Math.sin(g.yaw) * Math.sin(g.pitch) * uy,
          -Math.cos(g.pitch) * uy, Math.sin(g.yaw) * ux + Math.cos(g.yaw) * Math.sin(g.pitch) * uy];
        for (let k = 0; k < 3; k++) { pushes[i][k] += impulse[k] * strength; pushes[j][k] -= impulse[k] * strength; }
      }
      positions.forEach((p, i) => {
        const length = Math.hypot(...pushes[i]), limit = Math.min(1, 0.025 / (length || 1));
        const next = p.map((v, k) => v + pushes[i][k] * limit), norm = Math.hypot(...next);
        if (norm > 1e-6) positions[i] = next.map((v) => v * radii[i] / norm);
      });
    }
    return positions;
  }
  function frameGraph() {
    if (!g.data || !g.w) return;
    // Encuadre estable durante el giro: esfera que contiene toda la nube y sus etiquetas.
    const radius = Math.max(0.35, ...g.layout.map((p) => Math.hypot(...p)));
    const extent = radius * CAMERA / Math.sqrt(CAMERA * CAMERA - radius * radius);
    g.centerX = 0; g.centerY = 0;
    g.scale = Math.min(Math.max(1, g.w - 100), Math.max(1, g.h - 96)) / (2 * extent);
  }
  const nodeInfo = (i) => {  // i: índice del documento en el ranking
    const shown = shownCount();
    return i < shown ? { ...state.response.hits[i], context: false } : { ...g.data.context[i - shown], context: true };
  };
  function line(a, b, color, alpha, width, dash) {
    ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    ctx.setLineDash([]);
  }
  function pill(text, x, y, color) {
    ctx.font = `600 11px ${getComputedStyle(root).fontFamily}`;
    const w = ctx.measureText(text).width + 10;
    ctx.globalAlpha = 1; ctx.fillStyle = "#fff"; ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(x - w / 2, y - 9, w, 18, 9); ctx.fill(); ctx.stroke();
    ctx.fillStyle = color; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, x, y + 0.5);
  }
  const sameEdge = (e, f) => e && f && e.a === f.a && e.b === f.b;
  function edgeLit(e) {
    return [g.hover, g.pin].some((h) => h && (h.edge ? sameEdge(h.edge, e)
      : h.node === e.a + 1 || h.node === e.b + 1 || (h.node === 0 && e.a === -1)));
  }
  function drawGraph() {
    const d = g.data;
    if (!d || !g.w) return;
    const css = getComputedStyle(root), token = (name) => css.getPropertyValue(name).trim();
    const ink = token("--ink"), ink3 = token("--ink-3"), accent = token("--accent"), font = css.fontFamily;
    const shown = shownCount(), first = (state.page - 1) * PAGE_SIZE;
    g.pts = [project([0, 0, 0]), ...g.layout.map(project)];
    ctx.clearRect(0, 0, g.w, g.h);
    ctx.lineCap = "round";

    // Relaciones: consulta–resultado (color de acento) y entre resultados parecidos.
    g.edges = [
      ...d.nodes.slice(0, shown).map((n, i) => ({ a: -1, b: i, sim: n[3] })),
      ...d.edges.filter(([a, b]) => a < shown && b < shown).map(([a, b, sim]) => ({ a, b, sim })),
    ];
    const focus = g.hover || g.pin;
    const labels = [];
    for (const e of g.edges) {
      const lit = edgeLit(e), toQuery = e.a === -1, sim = clamp(e.sim, 0, 1);
      const alpha = lit ? 0.95 : focus ? 0.07 : toQuery ? 0.18 + 0.5 * sim : 0.28;
      line(g.pts[e.a + 1], g.pts[e.b + 1], toQuery ? accent : ink3, alpha, lit ? 2.5 : toQuery ? 1 + 1.5 * sim : 1);
      if (lit) labels.push(e);
    }

    // Nodos del fondo al frente; los de alrededor, pequeños y tenues.
    g.order = g.pts.map((p, i) => ({ i, ...p, radius: (i === 0 ? 8 : i - 1 < shown ? 11 : 4.5) * p.s })).sort((a, b) => a.z - b.z);
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    for (const node of g.order) {
      const hot = [g.hover, g.pin].some((h) => h?.node === node.i || h?.edge && (h.edge.a + 1 === node.i || h.edge.b + 1 === node.i));
      const depth = 0.42 + 0.58 * clamp((node.z + 1) / 2, 0, 1);
      const doc = node.i - 1, context = doc >= shown;
      const onPage = doc >= first && doc < first + PAGE_SIZE;
      const dim = focus && !hot && !labels.some((e) => e.a === doc || e.b === doc);
      const similarity = node.i === 0 ? 1 : clamp(d.nodes[doc][3], 0, 1);
      // Estilo plano: relleno sólido que va de casi blanco (poco parecido) a ciruela claro
      // (muy parecido) y un borde fino del mismo tono; sin brillo, degradado ni sombra.
      const fill = node.i === 0 ? accent : `hsl(318 ${12 + 38 * similarity}% ${96 - 16 * similarity}%)`;
      const edge = `hsl(318 ${12 + 38 * similarity}% ${62 - 24 * similarity}%)`;
      const radius = hot && context ? node.radius + 2 : node.radius;
      ctx.globalAlpha = node.i === 0 || hot ? 1 : context ? 0.35 * depth : dim ? 0.35 * depth : depth;
      ctx.fillStyle = context && !hot ? edge : fill;
      ctx.beginPath(); ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI); ctx.fill();
      if (node.i === 0 || context && !hot) continue;
      ctx.lineWidth = hot || onPage ? 2 : 1.25; ctx.strokeStyle = hot ? accent : onPage ? ink : edge; ctx.stroke();
      if (!context || hot) {
        ctx.fillStyle = ink; ctx.font = `600 ${Math.round(11 * node.s)}px ${font}`;
        ctx.fillText(String(doc + 1), node.x, node.y + 0.5);
      }
    }
    // Etiqueta de la consulta con contorno blanco, y el parecido de las relaciones resaltadas.
    const q = g.pts[0], label = ["consulta", q.x, q.y + 8 * q.s + 4];
    ctx.globalAlpha = 1; ctx.font = `600 12px ${font}`; ctx.textBaseline = "top"; ctx.lineJoin = "round";
    ctx.lineWidth = 4; ctx.strokeStyle = "#fff"; ctx.strokeText(...label);
    ctx.fillStyle = accent; ctx.fillText(...label);
    for (const e of labels) {
      const a = g.pts[e.a + 1], b = g.pts[e.b + 1];
      pill(decimal(e.sim), (a.x + b.x) / 2, (a.y + b.y) / 2, e.a === -1 ? accent : ink);
    }
  }
  function fit() {
    const rect = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!rect.width || !rect.height) return;
    g.w = rect.width; g.h = rect.height;
    canvas.width = Math.round(g.w * dpr); canvas.height = Math.round(g.h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    stopFocus(); frameGraph(); drawGraph();
  }
  const spinning = () => g.spin && !reduced() && g.data && !graphBox.hidden;
  function pauseSpin() { g.resumeAt = performance.now() + 2000; }
  function stopFocus() { cancelFrame(g.focusFrame); g.focusFrame = 0; pauseSpin(); }
  function tick(now) {
    g.frame = 0;
    if (!spinning()) return;
    if (g.last && !gesture && !g.hover && !g.focusFrame && now >= g.resumeAt && g.visible) { g.yaw += (Math.min(now - g.last, 50) / 1000) * 0.22; drawGraph(); }
    g.last = now;
    g.frame = requestFrame(tick);
  }
  function spin() { if (!g.frame && spinning()) { g.last = 0; g.frame = requestFrame(tick); } }
  function setSpin(value) { g.spin = value; spinBtn.setAttribute("aria-pressed", value); if (!value) { cancelFrame(g.frame); g.frame = 0; } spin(); }
  function setPanMode(value) { g.panMode = value; panBtn.setAttribute("aria-pressed", value); updateGraphCursor(); }
  function resetView() { stopFocus(); g.yaw = YAW; g.pitch = PITCH; g.zoom = 1; g.panX = 0; g.panY = 0; g.pin = null; g.hover = null; closeTip(); frameGraph(); updateGraphCursor(); drawGraph(); }
  function showGraph(r) {
    const d = state.graph && r && !r.error && r.hits?.length ? r.graph : null;
    stopFocus();
    cancelFrame(g.frame); g.frame = 0;
    if (g.hover) closeTip();
    for (const id of pointers.keys()) if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
    g.hover = null; g.pin = null; gesture = null; pointers.clear(); updateGraphCursor();
    clearTimeout(feedbackTimer); feedback.classList.remove("visible");
    graphBox.hidden = !d; graphBox.dataset.stale = "false";
    g.data = d && !d.error ? d : null;
    stage.hidden = !g.data; graphError.hidden = !(d && d.error);
    if (!d) return;
    $("[data-graph-model]").textContent = MODELS.find((m) => m.id === r.model)?.name || r.model;
    if (d.error) { graphError.textContent = `No se pudo dibujar el grafo. ${d.error}`; return; }
    const shown = shownCount(), around = d.nodes.length - shown;
    $("[data-legend-shown]").textContent = `Tu lista (1–${shown})`;
    $("[data-legend-range]").textContent = `Puestos ${shown + 1}–${d.nodes.length}`;
    $("[data-legend-context]").hidden = around === 0;
    g.yaw = YAW; g.pitch = PITCH; g.zoom = 1; g.panX = 0; g.panY = 0;
    g.layout = layoutGraph(d.nodes);
    canvas.setAttribute("aria-label", `Grafo 3D: la consulta en el centro, ${shown} resultados mostrados${around ? ` y ${around} más alrededor` : ""}, situados según el parecido de su significado. La lista contiene la misma información.`);
    fit(); spin();
  }
  function nodeTip(i) {
    // La distancia sigue el orden de parecido (no la proyección PCA, que solo orienta): no se
    // cita la varianza del PCA porque no describe este mapa.
    if (i === 0) return "Tu consulta. Cuanto más cerca está un resultado, más se parece su significado: la distancia sigue el orden de parecido con la consulta y la dirección es solo orientativa. El parecido exacto es el número de cada relación.";
    const h = nodeInfo(i - 1), sim = decimal(g.data.nodes[i - 1][3]);
    const where = h.context ? ` Fuera de los ${shownCount()} mostrados.` : "";
    return `Resultado ${h.rank} · Doc. ${h.doc_id} · parecido con la consulta ${sim}.${where} ${split(h.text).lead}`;
  }
  function edgeTip(e) {
    const rank = (i) => nodeInfo(i).rank;
    return e.a === -1
      ? `Consulta y resultado ${rank(e.b)}: parecido ${decimal(e.sim)}`
      : `Resultados ${rank(e.a)} y ${rank(e.b)}: parecido ${decimal(e.sim)}`;
  }
  // Qué hay bajo el puntero: primero nodos (los del frente ganan), luego relaciones.
  function targetAt(e) {
    const rect = canvas.getBoundingClientRect(), x = e.clientX - rect.left, y = e.clientY - rect.top;
    for (let n = g.order.length - 1; n >= 0; n--) {
      const node = g.order[n];
      if (Math.hypot(x - node.x, y - node.y) <= node.radius + 3) return { node: node.i };
    }
    let best = null, bestD = 5;
    for (const edge of g.edges) {
      const a = g.pts[edge.a + 1], b = g.pts[edge.b + 1], dx = b.x - a.x, dy = b.y - a.y;
      const t = clamp(((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy || 1), 0, 1);
      const dist = Math.hypot(x - (a.x + t * dx), y - (a.y + t * dy));
      if (dist < bestD) { bestD = dist; best = { edge }; }
    }
    return best;
  }
  const sameTarget = (a, b) => (a?.node ?? null) === (b?.node ?? null) && (a?.edge ? sameEdge(a.edge, b?.edge) : !b?.edge);
  function setHover(target, point) {
    if (!g.data || sameTarget(target, g.hover)) return;
    pauseSpin();
    g.hover = target;
    updateGraphCursor();
    if (target && point) pointTip(target.edge ? edgeTip(target.edge) : nodeTip(target.node), point);
    else if (!target) closeTip();
    drawGraph();
  }
  function goToHit(index) {
    const r = state.response;
    if (bar.dataset.busy === "true" || !r?.hits?.[index]) return;
    const page = Math.floor(index / PAGE_SIZE) + 1;
    if (page !== state.page) { state.page = page; renderPage(r); }
    const row = list.querySelector(`[data-rank="${r.hits[index].rank}"]`);
    if (!row) return;
    row.scrollIntoView({ block: "nearest", behavior: reduced() ? "auto" : "smooth" });
    row.classList.remove("flash"); void row.offsetWidth; row.classList.add("flash");
  }
  function focusTarget(target) {
    stopFocus(); g.pin = target;
    const position = target.edge
      ? (target.edge.a === -1 ? [0, 0, 0] : g.layout[target.edge.a]).map((v, k) => (v + g.layout[target.edge.b][k]) / 2)
      : target.node === 0 ? [0, 0, 0] : g.layout[target.node - 1];
    const p = projected(position), zoom = clamp(Math.max(2, g.zoom), 0.5, 6);
    const start = { x: g.panX, y: g.panY, zoom: g.zoom }, began = performance.now();
    const endX = -(p.x - g.centerX) * g.scale * zoom, endY = -(p.y - g.centerY) * g.scale * zoom;
    const animate = (now) => {
      const t = reduced() ? 1 : clamp((now - began) / 450, 0, 1), eased = 1 - (1 - t) ** 3;
      g.zoom = start.zoom + (zoom - start.zoom) * eased;
      g.panX = start.x + (endX - start.x) * eased; g.panY = start.y + (endY - start.y) * eased;
      drawGraph(); pauseSpin();
      g.focusFrame = t < 1 ? requestFrame(animate) : 0;
    };
    animate(began);
  }
  function zoomAt(factor, cx, cy) {
    stopFocus();
    const zoom = clamp(g.zoom * factor, 0.5, 6), ratio = zoom / g.zoom;
    const ox = cx - g.w / 2, oy = cy - g.h / 2;  // el punto bajo el cursor no se mueve
    g.panX = ox - (ox - g.panX) * ratio; g.panY = oy - (oy - g.panY) * ratio; g.zoom = zoom;
    graphFeedback(`Zoom ${g.zoom.toFixed(1).replace(".", ",")}×`);
    drawGraph();
  }
  // Gestos: arrastrar gira (o mueve en modo Mover, con Mayús o clic derecho); dos dedos acercan y mueven.
  const pointers = new Map();
  let gesture = null, feedbackTimer;
  const feedback = $(".graph-feedback");
  function updateGraphCursor() {
    canvas.style.cursor = gesture ? (gesture.mode === "pan" ? "move" : "grabbing") : g.panMode ? "move" : g.hover ? "pointer" : "grab";
  }
  function graphFeedback(text, held = false) {
    clearTimeout(feedbackTimer); feedback.textContent = text; feedback.classList.add("visible");
    if (!held) feedbackTimer = setTimeout(() => feedback.classList.remove("visible"), 600);
  }
  const local = (e) => { const rect = canvas.getBoundingClientRect(); return { x: e.clientX - rect.left, y: e.clientY - rect.top }; };
  on(canvas, "contextmenu", (e) => e.preventDefault());
  on(canvas, "pointerdown", (e) => {
    if (e.button > 2) return;
    stopFocus();
    e.preventDefault(); canvas.focus({ preventScroll: true, focusVisible: false });  // teclas tras pulsar, sin anillo de foco
    setHover(null); closeTip();
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, local(e));
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      gesture = { mode: "pinch", dist: Math.hypot(a.x - b.x, a.y - b.y), mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, moved: true };
    } else if (pointers.size === 1) {
      gesture = { mode: g.panMode || e.shiftKey || e.button === 1 || e.button === 2 ? "pan" : "rotate", ...local(e), moved: false };
    }
    updateGraphCursor();
  });
  on(canvas, "pointermove", (e) => {
    if (!pointers.has(e.pointerId)) { if (!gesture) setHover(targetAt(e), e); return; }
    pointers.set(e.pointerId, local(e));
    pauseSpin();
    if (!gesture) return;
    if (gesture.mode === "pinch" && pointers.size === 2) {
      const [a, b] = [...pointers.values()], dist = Math.hypot(a.x - b.x, a.y - b.y), mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      g.panX += mid.x - gesture.mid.x; g.panY += mid.y - gesture.mid.y;
      zoomAt(dist / (gesture.dist || dist), mid.x, mid.y);
      graphFeedback(`Zoom ${g.zoom.toFixed(1).replace(".", ",")}×`, true);
      gesture.dist = dist; gesture.mid = mid;
      return;
    }
    const p = local(e), dx = p.x - gesture.x, dy = p.y - gesture.y;
    if (!gesture.moved && Math.hypot(dx, dy) < 4) return;
    if (!gesture.moved) { gesture.moved = true; setHover(null); closeTip(); updateGraphCursor(); }
    if (gesture.mode === "pan") { g.panX += dx; g.panY += dy; }
    else { g.yaw += dx * 0.01; g.pitch = clamp(g.pitch + dy * 0.01, -1.45, 1.45); }
    graphFeedback(gesture.mode === "pan" ? "Moviendo" : "Girando", true);
    gesture.x = p.x; gesture.y = p.y;
    drawGraph();
  });
  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    const done = gesture;
    pauseSpin();
    if (pointers.size === 0) gesture = null;
    else if (done?.mode === "pinch") gesture = null;  // al soltar un dedo no se salta a girar
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    if (!gesture && done?.moved) graphFeedback(feedback.textContent);
    setHover(e.type === "pointerup" && !pointers.size ? targetAt(e) : null, e);
    updateGraphCursor();
    if (!done || done.moved || e.type !== "pointerup") return;
    const target = targetAt(e);
    if (target) {
      focusTarget(target);
      if (target.node > 0 && target.node - 1 < shownCount()) goToHit(target.node - 1);
    } else { stopFocus(); g.pin = null; }
    drawGraph();
  };
  on(canvas, "pointerup", endPointer);
  on(canvas, "pointercancel", endPointer);
  on(canvas, "lostpointercapture", (e) => { if (pointers.has(e.pointerId)) endPointer(e); });
  on(canvas, "pointerleave", () => { if (!gesture) setHover(null); });
  on(canvas, "wheel", (e) => {
    e.preventDefault();
    const p = local(e);
    zoomAt(Math.exp(-e.deltaY * 0.0015), p.x, p.y);
  }, { passive: false });
  on(canvas, "keydown", (e) => {
    const step = { ArrowLeft: [-0.2, 0], ArrowRight: [0.2, 0], ArrowUp: [0, -0.2], ArrowDown: [0, 0.2] }[e.key];
    if (step) { e.preventDefault(); stopFocus(); g.yaw += step[0]; g.pitch = clamp(g.pitch + step[1], -1.45, 1.45); graphFeedback("Girando"); drawGraph(); }
    else if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomAt(1.25, g.w / 2, g.h / 2); }
    else if (e.key === "-") { e.preventDefault(); zoomAt(0.8, g.w / 2, g.h / 2); }
    else if (e.key === "0") { e.preventDefault(); resetView(); }
  });
  root.querySelectorAll("[data-graph-zoom]").forEach((b) => on(b, "click", () => zoomAt(b.dataset.graphZoom === "in" ? 1.3 : 1 / 1.3, g.w / 2, g.h / 2)));
  on(spinBtn, "click", () => setSpin(!g.spin));
  on(panBtn, "click", () => setPanMode(!g.panMode));
  on($("[data-graph-reset]"), "click", resetView);
  // Pasar por un resultado de la lista resalta su punto y sus relaciones en el grafo.
  on(list, "pointerover", (e) => { const row = e.target.closest("[data-rank]"); if (row && g.data) { setHover({ node: +row.dataset.rank }); } });
  on(list, "pointerleave", () => { if (g.hover) setHover(null); });
  on(matchMedia("(prefers-reduced-motion: reduce)"), "change", () => { if (reduced()) stopFocus(); spin(); });
  spinBtn.setAttribute("aria-pressed", g.spin);
  const barResized = new ResizeObserver(updateGhost); barResized.observe(bar);
  const resized = new ResizeObserver(fit), seen = new IntersectionObserver((entries) => { g.visible = entries[0].isIntersecting; });
  resized.observe(stage); seen.observe(stage);
  listeners.signal.addEventListener("abort", () => { clearTimeout(tipDelay); clearTimeout(tipHide); clearTimeout(feedbackTimer); clearTimeout(state.timer); clearTimeout(state.pageTimer); clearTimeout(blurTimer); cancelFrame(g.frame); cancelFrame(g.focusFrame); barResized.disconnect(); resized.disconnect(); seen.disconnect(); });

  // ---------- Sugerencias: consultas de entrenamiento de FiQA ----------
  let active = -1, items = [];
  function showSuggest(text) {
    const q = text.trim().toLowerCase();
    items = [];
    if (q.length >= 2) {
      const starts = [], has = [];
      for (let i = 0; i < LOWER.length && starts.length < 6; i++) {
        const at = LOWER[i].indexOf(q);
        if (at === 0 && LOWER[i] !== q) starts.push(i); else if (at > 0 && has.length < 6) has.push(i);
      }
      items = starts.concat(has).slice(0, 6).map((i) => SUGG[i]);
    }
    active = -1;
    if (!items.length) return hideSuggest();
    suggest.innerHTML = items.map((s, n) => {
      const at = s.toLowerCase().indexOf(q);
      return `<li id="jab-sg-${n}" role="option" aria-selected="false"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/></svg><span>${esc(s.slice(0, at))}<b>${esc(s.slice(at, at + q.length))}</b>${esc(s.slice(at + q.length))}</span><kbd class="kbd" aria-hidden="true">↵</kbd></li>`;
    }).join("");
    suggest.hidden = false; input.setAttribute("aria-expanded", "true");
  }
  function hideSuggest() { suggest.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; }
  function markSuggest() {
    [...suggest.children].forEach((li, n) => li.setAttribute("aria-selected", n === active));
    if (active >= 0) input.setAttribute("aria-activedescendant", `jab-sg-${active}`);
    else input.removeAttribute("aria-activedescendant");
    updateGhost();
  }
  function pick(n) { const value = items[n]; editInput(value); hideSuggest(); syncClear(); commitSearch(); }
  on(suggest, "pointerdown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); pick([...suggest.children].indexOf(li)); } });

  // ---------- Autocompletado en gris (como en los buscadores web) ----------
  // Primero una consulta de FiQA que empiece igual; si no hay, la palabra frecuente que completa la última.
  let ghostRest = "";
  function completion(value) {
    const low = value.toLowerCase();
    if (low.trim().length >= 2) {
      const i = LOWER.findIndex((s) => s.length > low.length && s.startsWith(low));
      if (i >= 0) {
        const rest = SUGG[i].slice(value.length);
        return value === low ? rest.toLowerCase() : rest;
      }
    }
    const m = value.match(/([A-Za-z]{2,})$/);
    if (!m) return "";
    const part = m[1].toLowerCase(), word = VOCAB.find((w) => w.length > part.length && w.startsWith(part));
    return word ? word.slice(part.length) : "";
  }
  function sizeInput() {
    const focused = document.activeElement === input;
    const scroll = input.scrollTop;
    input.style.overflowY = "hidden";
    input.style.height = "42px";
    input.style.height = `${focused ? Math.min(138, input.scrollHeight) : 42}px`;
    input.style.overflowY = focused && input.scrollHeight > input.clientHeight ? "auto" : "hidden";
    input.scrollTop = focused ? scroll : 0;
  }
  function updateGhost() {
    const v = input.value, atEnd = input.selectionStart === v.length && input.selectionEnd === v.length;
    ghostRest = document.activeElement === input && atEnd && active < 0 && v.trim() ? completion(v) : "";
    ghostRest = ghostRest.slice(0, Math.max(0, input.maxLength - v.length));
    ghost.innerHTML = ghostRest ? `<span class="typed">${esc(v)}</span>${esc(ghostRest)}` : "";
    sizeInput();
    const field = input.getBoundingClientRect(), box = bar.getBoundingClientRect();
    const scrollbar = input.offsetWidth - input.clientWidth;
    // Conserva los subpíxeles del flex: redondear clientWidth cambia dónde envuelve el espejo.
    ghost.style.left = `${field.left - box.left - bar.clientLeft}px`;
    ghost.style.top = `${field.top - box.top - bar.clientTop}px`;
    ghost.style.width = `${field.width - scrollbar}px`; ghost.style.height = `${input.clientHeight}px`;
    // El espejo mide el texto completo, incluido el gris, con el mismo ancho y límite.
    if (ghostRest) {
      ghost.style.height = "auto";
      input.style.height = `${Math.min(138, Math.max(input.scrollHeight, ghost.scrollHeight))}px`;
      ghost.style.height = `${input.clientHeight}px`;
    }
    ghost.scrollTop = input.scrollTop;
  }
  function acceptGhost() {
    editInput(ghostRest, input.value.length, input.value.length);
    syncClear(); showSuggest(input.value); updateGhost(); schedule();
  }

  // Las ediciones programáticas usan la pila nativa de deshacer del textarea.
  // El respaldo conserva la edición y el evento input si el navegador rechaza execCommand.
  function editInput(text, start = 0, end = input.value.length) {
    if (!text && start === end) return;
    input.focus({ preventScroll: true });
    input.setSelectionRange(start, end);
    const before = input.value;
    let handled = false;
    try { handled = document.execCommand(text ? "insertText" : "delete", false, text); } catch { /* respaldo */ }
    if (!handled && input.value === before) {
      input.setRangeText(text, start, end, "end");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  }

  // ---------- Entrada ----------
  const syncClear = () => { clear.hidden = !input.value; sizeInput(); };
  on(input, "input", () => {
    const start = input.selectionStart, end = input.selectionEnd;
    const value = input.value.replace(/[\r\n]/g, " ");
    if (value !== input.value) { input.value = value; input.setSelectionRange(start, end); }
    state.q = input.value; syncClear(); showSuggest(input.value); updateGhost(); schedule();
  });
  on(input, "keydown", (e) => {
    const atEnd = input.selectionStart === input.value.length;
    if ((e.key === "Tab" && !e.shiftKey || e.key === "ArrowRight" && atEnd) && ghostRest) { e.preventDefault(); acceptGhost(); }
    else if (e.key === "ArrowDown" && items.length) { e.preventDefault(); active = (active + 1) % items.length; markSuggest(); }
    else if (e.key === "ArrowUp" && items.length) { e.preventDefault(); active = active <= 0 ? -1 : active - 1; markSuggest(); }
    else if (e.key === "Enter") { e.preventDefault(); if (active >= 0) pick(active); else { hideSuggest(); state.q = input.value; commitSearch(); } }
    else if (e.key === "Escape" && !suggest.hidden) { hideSuggest(); updateGhost(); }
  });
  on(input, "paste", (e) => {
    const text = e.clipboardData?.getData("text/plain");
    if (text === undefined) return;
    e.preventDefault();
    const room = input.maxLength - (input.value.length - (input.selectionEnd - input.selectionStart));
    editInput(text.replace(/\r\n|[\r\n]/g, " ").slice(0, Math.max(0, room)), input.selectionStart, input.selectionEnd);
  });
  on(input, "scroll", () => { ghost.scrollTop = input.scrollTop; });
  on(input, "select", updateGhost);
  on(input, "keyup", updateGhost);
  on(input, "click", updateGhost);
  on(input, "focus", updateGhost);
  let blurTimer;
  on(input, "blur", () => { clearTimeout(blurTimer); blurTimer = setTimeout(() => { updateGhost(); if (!moving && !bar.contains(document.activeElement)) hideSuggest(); }, 0); });
  on(clear, "click", () => { editInput(""); syncClear(); hideSuggest(); input.focus(); search(); updateGhost(); });
  // Ejemplos de inicio: tres preguntas reales de FiQA (train), distintas en cada visita.
  const pool = SUGG.filter((s) => s.length >= 18 && s.length <= 52 && s.endsWith("?"));
  if (pool.length >= 3) {
    const picks = new Set();
    while (picks.size < 3) picks.add(pool[Math.floor(Math.random() * pool.length)]);
    const chosen = [...picks];
    root.querySelectorAll(".example span").forEach((span, i) => { span.textContent = chosen[i]; });
  }
  root.querySelectorAll(".example").forEach((b) => on(b, "click", () => { editInput(b.textContent.trim()); syncClear(); commitSearch(); input.focus(); }));
  root.querySelectorAll("[data-logo]").forEach((a) => on(a, "click", (e) => { e.preventDefault(); hop(); state.route = "home"; setView("buscar"); clear.click(); }));

  // ---------- Menú del modelo (chip o Ctrl/⌘ K) ----------
  root.querySelector("[data-models]").innerHTML = MODELS.map((m, i) => `
    <button class="opt" type="button" role="menuitemradio" aria-checked="false" data-model="${m.id}" data-tip="${esc(m.tip)}">
      <span><strong>${m.name}</strong><small>${m.hint}</small></span><kbd class="kbd">${i + 1}</kbd>
    </button>`).join("");
  root.querySelector("[data-k]").innerHTML = KS.map((k) => `<button type="button" aria-pressed="false" data-kval="${k}" class="num">${k}</button>`).join("");
  function openMenu(open) {
    menu.hidden = !open; chip.setAttribute("aria-expanded", open);
    if (open) {
      hideSuggest();
      suppressMenuFocusTip = true;
      (menu.querySelector('[aria-checked="true"]') || menu.querySelector(".opt")).focus({ preventScroll: true });
      suppressMenuFocusTip = false;
      // Si el menú no cabe bajo la barra (p. ej. en el inicio con la ventana baja), se
      // desplaza la página lo justo para verlo entero en vez de dejarlo cortado.
      requestFrame(() => {
        if (menu.getBoundingClientRect().bottom > window.innerHeight - 8) menu.scrollIntoView({ block: "end", behavior: reduced() ? "auto" : "smooth" });
      });
    }
  }
  function syncControls() {
    root.querySelector("[data-model-name]").textContent = MODELS.find((m) => m.id === state.model).name;
    menu.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-checked", b.dataset.model === state.model));
    menu.querySelectorAll("[data-kval]").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.kval === state.k));
    const dense = state.model === "dense";
    sw.disabled = dense; sw.setAttribute("aria-checked", state.stem && !dense);
    root.querySelector("[data-stem-row]").setAttribute("aria-disabled", dense);
    gsw.setAttribute("aria-checked", state.graph); root.dataset.graph = state.graph ? "on" : "off";
  }
  const rerun = () => { if (state.q.trim()) search(); else { state.seq += 1; send("request", currentRequest()); } };
  function setModel(id) { state.model = id; syncControls(); hop(); rerun(); }
  on(chip, "click", () => openMenu(menu.hidden));
  on(menu, "click", (e) => {
    const m = e.target.closest("[data-model]"); if (m) { setModel(m.dataset.model); openMenu(false); input.focus(); }
    const k = e.target.closest("[data-kval]"); if (k) { state.k = +k.dataset.kval; syncControls(); rerun(); }
    const toggle = e.target.closest(".switch");
    if (toggle === sw && !sw.disabled) { state.stem = !state.stem; syncControls(); rerun(); }
    if (toggle === gsw) { state.graph = !state.graph; syncControls(); if (!state.graph) showGraph(null); rerun(); }
  });
  on(menu, "keydown", (e) => {
    const opts = [...menu.querySelectorAll(".opt")], i = opts.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); opts[(i + 1) % opts.length].focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); opts[(i - 1 + opts.length) % opts.length].focus(); }
    if (/^[1-4]$/.test(e.key)) { e.preventDefault(); setModel(MODELS[+e.key - 1].id); openMenu(false); input.focus(); }
    if (e.key === "Escape") { openMenu(false); chip.focus(); }
  });
  on(document, "pointerdown", (e) => { if (!menu.hidden && !menu.contains(e.target) && !chip.contains(e.target)) openMenu(false); });
  on(document, "keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openMenu(menu.hidden); }
    else if (e.key === "/" && !typing) { e.preventDefault(); input.focus(); }
  });

  // ---------- Vistas: Buscar / Comparar / Métricas (cada una con su ruta) ----------
  function markView() {
    root.dataset.view = state.view;
    placeExamples();
    root.querySelectorAll("[data-view-link]").forEach((a) => {
      if (a.dataset.viewLink === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }
  function setView(view) {
    if (state.view === view) return;
    state.view = view;
    markView(); hop();
    setMode(isHome() ? "home" : "results");
    // Si la consulta cambió en otra vista y aún no se buscó, Buscar no muestra la lista
    // anterior como vigente: la marca como desactualizada mientras llega la respuesta.
    if (view === "buscar" && state.searchedKey !== searchKey()) search();
    else { state.seq += 1; send("request", currentRequest()); }
  }
  root.querySelectorAll("[data-view-link]").forEach((a) => on(a, "click", (e) => { e.preventDefault(); setView(a.dataset.viewLink); }));

  // Estado que llega desde la URL (recargar, atrás, adelante, un enlace compartido).
  function adopt(init) {
    Object.assign(state, {
      q: init.q || "", model: init.model || "bm25", k: init.k || 10, stem: !!init.stem, graph: !!init.graph,
      view: init.view || "buscar", route: init.route || "home", seq: init.seq || 0, nav: init.nav || 0, renderedKey: null, page: 1, hasHits: false,
    });
    input.value = state.q; syncClear(); hideSuggest(); syncControls(); markView(); updateGhost();
    // Atrás, adelante y recargar no pasan por currentRequest(): sin esto Comparar vacío conservaría
    // data-query="on" (línea «Buscando…» eterna y sin ejemplos).
    root.dataset.query = state.q.trim() ? "on" : "off"; state.searchedKey = searchKey();
    setMode(isHome() ? "home" : "results");
    if (state.q.trim()) { bar.dataset.busy = "true"; renderSkeleton(); }
    else { list.innerHTML = ""; status.innerHTML = ""; pager.hidden = true; dym.hidden = true; showGraph(null); }
  }

  // ---------- Estado inicial (también tras volver a montar) ----------
  input.value = state.q; syncClear(); syncControls(); placeBar("home"); markView();
  if (window.__jabRefocus) {  // se volvió a montar tras pasar a /search desde el campo
    window.__jabRefocus = false;
    requestFrame(() => { input.focus({ preventScroll: true }); input.setSelectionRange(input.value.length, input.value.length); });
  }
  root.dataset.query = state.q.trim() ? "on" : "off";
  state.searchedKey = searchKey();  // Python ya responde a este estado inicial
  if (!isHome()) { root.dataset.mode = "results"; placeBar("results"); }

  let resent = -1;         // si Python atendió un estado viejo, se reenvía una vez con seq nuevo
  return {
    receive(d) {
      if ((d.initial?.nav ?? 0) > state.nav) adopt(d.initial);
      const r = d.response, pySeq = d.initial?.seq ?? 0;
      if (r && r.seq === state.seq) {
        bar.dataset.busy = "false"; list.dataset.stale = "false";
        if (r.key !== state.renderedKey || r.error) { render(r); state.renderedKey = r.error ? null : r.key; }
      }
      if (pySeq < state.seq && resent !== state.seq) {
        state.seq += 1; resent = state.seq; send("request", currentRequest());
      }
    },
  };
}
