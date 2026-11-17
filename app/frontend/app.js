// jab — interfaz de búsqueda (componente de Streamlit, issue #13).
// El navegador gestiona escritura, menús y animación; Python ejecuta la búsqueda.
// Contrato con Python (app/ui.py):
//   estado "request" → {q, model, k, stem, view, seq}; cada cambio sube seq (ninguno se pierde)
//   data.initial     ← último estado atendido por Python (incluye su seq)
//   data.response    ← {seq, key, model, stem, elapsed_ms, terms, hits:[{rank, doc_id, text, score, matched, spans}]} o {seq, key, error}

const SUGG = __SUGGESTIONS__;
const LOWER = SUGG.map((s) => s.toLowerCase());
const MODELS = [
  { id: "jaccard", name: "Jaccard", hint: "Palabras en común", tip: "Compara los conjuntos de palabras distintas; 1 significa que coinciden por completo." },
  { id: "tfidf", name: "TF-IDF", hint: "Palabras ponderadas por rareza", tip: "Compara palabras y da más peso a las menos frecuentes en FiQA." },
  { id: "bm25", name: "BM25", hint: "Ponderadas y ajustadas por longitud", tip: "Suma aportes de palabras y ajusta repeticiones y largo del documento." },
  { id: "dense", name: "Embeddings", hint: "Por significado, no por palabras", tip: "MiniLM compara el significado aunque no se repitan las mismas palabras." },
];
const KS = [5, 10, 20, 50];
const DEBOUNCE = 220;
const PAGE_SIZE = 5;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function clearSectionAnchor() {
  if (window.location.hash) {
    // Quitar anclas antiguas sin recargar, perder parámetros ni añadir historial.
    window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search);
  }
}

export default function ({ parentElement, data, setStateValue }) {
  clearSectionAnchor();
  const root = parentElement.querySelector(".jab");
  root._send = setStateValue;
  if (!root._jab) root._jab = mount(root, data || {});
  root._jab.receive(data || {});
}

function mount(root, data) {
  const $ = (s) => root.querySelector(s);
  const init = data.initial || {};
  const state = {
    q: init.q || "", model: init.model || "bm25", k: init.k || 10, stem: !!init.stem,
    view: init.view || "buscar", seq: init.seq || 0, timer: null, hasHits: false,
    renderedKey: null, page: 1, response: null, pageTimer: null,
  };
  const bar = $(".bar"), input = $("#jab-q"), clear = $(".clear"), chip = $(".chip"), menu = $("#jab-menu");
  const suggest = $("#jab-suggest"), list = $(".list"), status = $(".status"), sw = $(".switch"), pager = $(".pager"), tip = $("#jab-tip");
  const send = (name, value) => root._send(name, value);

  // Una sola señal de cancelación para los oyentes globales si el componente se vuelve a montar.
  window.__jabTipCleanup?.();
  window.__jabListeners?.abort();
  const listeners = (window.__jabListeners = new AbortController());
  const on = (target, type, fn) => target.addEventListener(type, fn, { signal: listeners.signal });

  const logoTpl = root.querySelector("[data-logo-svg]");
  root.querySelectorAll("[data-logo]").forEach((el) => el.append(logoTpl.content.cloneNode(true)));
  try {  // la entrada del punto del logo ocurre una vez por sesión del navegador
    if (sessionStorage.getItem("jab-dot")) root.querySelector(".logo.first")?.classList.remove("first");
    else sessionStorage.setItem("jab-dot", "1");
  } catch { /* sin almacenamiento: animación normal */ }

  // ---------- Modo: inicio ↔ resultados ----------
  let moving = false;
  const placeBar = (mode) => root.querySelector(`[data-slot="${mode}"]`).append(bar);
  function setMode(mode) {
    if (root.dataset.mode === mode) return;
    const focused = document.activeElement === input, reopen = !suggest.hidden;
    const apply = () => {
      moving = true;
      root.dataset.mode = mode; placeBar(mode);
      if (focused) input.focus({ preventScroll: true });
      if (reopen) showSuggest(input.value);
      moving = false;
    };
    if (document.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) document.startViewTransition(apply);
    else apply();
  }

  // ---------- Búsqueda: Python responde con el mismo seq ----------
  const currentRequest = () => ({ q: state.q.trim(), model: state.model, k: state.k, stem: state.stem, view: state.view, seq: state.seq });
  function search() {
    clearTimeout(state.timer);
    clearTimeout(state.pageTimer);
    list.classList.remove("leaving");
    const q = state.q.trim();
    state.seq += 1;
    state.page = 1; state.renderedKey = null;
    send("request", currentRequest());
    if (!q) { state.hasHits = false; state.renderedKey = null; state.response = null; list.innerHTML = ""; list.dataset.stale = "false"; status.textContent = ""; pager.hidden = true; bar.dataset.busy = "false"; if (state.view === "buscar") setMode("home"); return; }
    setMode("results");
    bar.dataset.busy = "true";
    if (state.hasHits) list.dataset.stale = "true"; else renderSkeleton();
  }
  const schedule = () => { clearTimeout(state.timer); state.timer = setTimeout(search, DEBOUNCE); };

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
    const value = h.score.toFixed(2).replace(".", ",");
    const sought = terms.length ? ` Se buscó: ${terms.join(", ")}.` : " No quedaron términos tras el preprocesamiento.";
    if (model === "bm25") return `${value}: suma de lo que aportan ${terms.join(" y ") || "los términos"} en este documento. No tiene tope; solo sirve para comparar resultados de esta misma búsqueda.${sought}`;
    if (model === "tfidf") return `${value}: coseno entre consulta y documento, de 0 a 1. Las palabras raras en la colección pesan más.${sought}`;
    if (model === "jaccard") return `${value}: el ${(Number(value.replace(",", ".")) * 100).toLocaleString("es", { maximumFractionDigits: 0 })} % de las palabras distintas de consulta y documento son comunes.${sought}`;
    return `${value}: coseno entre significados (MiniLM). No es un porcentaje de relevancia.`;
  }
  function renderPage(r) {
    closeTip();
    const n = r.hits.length, pages = Math.ceil(n / PAGE_SIZE);
    state.page = Math.min(state.page, pages);
    const first = (state.page - 1) * PAGE_SIZE;
    const ms = r.elapsed_ms < 1 ? "menos de 1" : Math.round(r.elapsed_ms);
    status.textContent = `Resultados ${first + 1}–${Math.min(first + PAGE_SIZE, n)} de ${n} · ${ms} ms`;
    const top = r.hits[0].score;
    list.innerHTML = r.hits.slice(first, first + PAGE_SIZE).map((h, i) => {
      const { lead, rest, leadAt, restAt } = split(h.text);
      const pct = top > 0 ? Math.max(0, Math.min(100, (h.score / top) * 100)) : 0;
      return `<li class="row" style="--i:${i}" ${h.rank === 1 ? 'data-top="true"' : ""}>
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
  }
  function render(r) {
    clearTimeout(state.pageTimer);
    list.classList.remove("leaving"); closeTip();
    bar.dataset.busy = "false"; list.dataset.stale = "false";
    state.response = r;
    if (r.error) { state.hasHits = false; status.textContent = ""; pager.hidden = true; list.innerHTML = `<li class="empty"><p>${esc(r.error)}</p></li>`; return; }
    const model = MODELS.find((m) => m.id === r.model), name = model?.name || r.model;
    if (!r.hits.length) {
      state.hasHits = false; status.textContent = ""; pager.hidden = true;
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
    status.textContent = "";
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
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) change();
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
  function openTip(target) {
    if (target === tipTarget) return;
    closeTip(); tipTarget = target;
    tip.textContent = target.dataset.tip;
    tipDelay = setTimeout(() => {
      if (tipTarget !== target) return;
      clearTimeout(tipHide);
      tip.hidden = false;
      target.setAttribute("aria-describedby", "jab-tip");
      const rect = target.getBoundingClientRect();
      const width = tip.getBoundingClientRect().width;
      const left = Math.max(8, Math.min(rect.right + 10, window.innerWidth - width - 8));
      tip.style.left = `${left}px`;
      tip.style.top = `${Math.max(8, Math.min(rect.top, window.innerHeight - tip.getBoundingClientRect().height - 8))}px`;
      requestAnimationFrame(() => tip.classList.add("visible"));
    }, 300);
  }
  on(root, "pointerover", (e) => { const target = e.target.closest("[data-tip]"); if (target) openTip(target); });
  on(root, "pointerout", (e) => { if (tipTarget?.contains(e.target) && !tipTarget.contains(e.relatedTarget)) closeTip(); });
  let suppressMenuFocusTip = false;
  on(root, "focusin", (e) => { const target = e.target.closest("[data-tip]"); if (target && !suppressMenuFocusTip) openTip(target); });
  on(root, "focusout", (e) => { if (tipTarget?.contains(e.target)) closeTip(); });
  on(document, "keydown", (e) => { if (e.key === "Escape" && tipTarget) closeTip(); });

  // ---------- Sugerencias: consultas de entrenamiento de FiQA ----------
  let active = -1, items = [], typed = state.q;
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
      return `<li id="jab-sg-${n}" role="option" aria-selected="false"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/></svg><span>${esc(s.slice(0, at))}<b>${esc(s.slice(at, at + q.length))}</b>${esc(s.slice(at + q.length))}</span></li>`;
    }).join("");
    suggest.hidden = false; input.setAttribute("aria-expanded", "true");
  }
  function hideSuggest() { suggest.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; }
  function markSuggest() {
    [...suggest.children].forEach((li, n) => li.setAttribute("aria-selected", n === active));
    if (active >= 0) { input.value = items[active]; input.setAttribute("aria-activedescendant", `jab-sg-${active}`); }
    else input.value = typed;
  }
  function pick(n) { input.value = state.q = typed = items[n]; hideSuggest(); syncClear(); search(); }
  on(suggest, "pointerdown", (e) => { const li = e.target.closest("li"); if (li) { e.preventDefault(); pick([...suggest.children].indexOf(li)); } });

  // ---------- Entrada ----------
  const syncClear = () => { clear.hidden = !input.value; };
  on(input, "input", () => { typed = state.q = input.value; syncClear(); showSuggest(input.value); schedule(); });
  on(input, "keydown", (e) => {
    if (e.key === "ArrowDown" && items.length) { e.preventDefault(); active = (active + 1) % items.length; markSuggest(); }
    else if (e.key === "ArrowUp" && items.length) { e.preventDefault(); active = active <= 0 ? -1 : active - 1; markSuggest(); }
    else if (e.key === "Enter") { e.preventDefault(); if (active >= 0) pick(active); else { hideSuggest(); state.q = input.value; search(); } }
    else if (e.key === "Escape" && !suggest.hidden) { hideSuggest(); input.value = typed; }
  });
  on(input, "blur", () => setTimeout(() => { if (!moving && !bar.contains(document.activeElement)) hideSuggest(); }, 0));
  on(clear, "click", () => { input.value = state.q = typed = ""; syncClear(); hideSuggest(); search(); input.focus(); });
  // Ejemplos de inicio: tres preguntas reales de FiQA (train), distintas en cada visita.
  const pool = SUGG.filter((s) => s.length >= 18 && s.length <= 52 && s.endsWith("?"));
  if (pool.length >= 3) {
    const picks = new Set();
    while (picks.size < 3) picks.add(pool[Math.floor(Math.random() * pool.length)]);
    const chosen = [...picks];
    root.querySelectorAll(".example").forEach((b, i) => { b.textContent = chosen[i]; });
  }
  root.querySelectorAll(".example").forEach((b) => on(b, "click", () => { input.value = state.q = typed = b.textContent; syncClear(); search(); input.focus(); }));
  root.querySelectorAll("[data-logo]").forEach((a) => on(a, "click", (e) => { e.preventDefault(); setView("buscar"); clear.click(); }));

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
      (menu.querySelector('[aria-checked="true"]') || menu.querySelector(".opt")).focus();
      suppressMenuFocusTip = false;
    }
  }
  function syncControls() {
    root.querySelector("[data-model-name]").textContent = MODELS.find((m) => m.id === state.model).name;
    menu.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-checked", b.dataset.model === state.model));
    menu.querySelectorAll("[data-kval]").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.kval === state.k));
    const dense = state.model === "dense";
    sw.disabled = dense; sw.setAttribute("aria-checked", state.stem && !dense);
    root.querySelector("[data-stem-row]").setAttribute("aria-disabled", dense);
  }
  const rerun = () => { if (state.q.trim()) search(); else { state.seq += 1; send("request", currentRequest()); } };
  function setModel(id) { state.model = id; syncControls(); rerun(); }
  on(chip, "click", () => openMenu(menu.hidden));
  on(menu, "click", (e) => {
    const m = e.target.closest("[data-model]"); if (m) { setModel(m.dataset.model); openMenu(false); input.focus(); }
    const k = e.target.closest("[data-kval]"); if (k) { state.k = +k.dataset.kval; syncControls(); rerun(); }
    if (e.target.closest(".switch") && !sw.disabled) { state.stem = !state.stem; syncControls(); rerun(); }
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

  // ---------- Vistas: Buscar / Comparar / Métricas ----------
  function markView() {
    root.querySelectorAll("[data-view-link]").forEach((a) => {
      if (a.dataset.viewLink === state.view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
  }
  function setView(view) {
    clearSectionAnchor();
    if (state.view === view) return;
    state.view = view; root.dataset.view = view;
    markView();
    if (view !== "buscar") setMode("results"); else if (!state.q.trim()) setMode("home");
    state.seq += 1; send("request", currentRequest());
  }
  root.querySelectorAll("[data-view-link]").forEach((a) => on(a, "click", (e) => { e.preventDefault(); setView(a.dataset.viewLink); }));

  // ---------- Estado inicial (también tras volver a montar) ----------
  input.value = state.q; syncClear(); syncControls(); placeBar("home");
  root.dataset.view = state.view; markView();
  if (state.q.trim() || state.view !== "buscar") { root.dataset.mode = "results"; placeBar("results"); }

  let resent = -1;         // si Python atendió un estado viejo, se reenvía una vez con seq nuevo
  return {
    receive(d) {
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
