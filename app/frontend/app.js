// jab — interfaz de búsqueda (componente de Streamlit, issue #13).
// El navegador gestiona escritura, menús y animación; Python ejecuta la búsqueda.
// Contrato con Python (app/ui.py):
//   estado "request" → {q, model, k, stem, view, seq}; cada cambio sube seq (ninguno se pierde)
//   data.initial     ← último estado atendido por Python (incluye su seq)
//   data.response    ← {seq, key, model, stem, elapsed_ms, hits:[{rank, doc_id, text, score}]} o {seq, key, error}

const SUGG = __SUGGESTIONS__;
const LOWER = SUGG.map((s) => s.toLowerCase());
const MODELS = [
  { id: "jaccard", name: "Jaccard", hint: "Palabras en común" },
  { id: "tfidf", name: "TF-IDF", hint: "Palabras ponderadas por rareza" },
  { id: "bm25", name: "BM25", hint: "Ponderadas y ajustadas por longitud" },
  { id: "dense", name: "Embeddings", hint: "Por significado, no por palabras" },
];
const KS = [5, 10, 20, 50];
const DEBOUNCE = 220;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

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
    q: init.q || "", model: init.model || "bm25", k: init.k || 10, stem: !!init.stem,
    view: init.view || "buscar", seq: init.seq || 0, timer: null, hasHits: false,
    renderedKey: null,  // búsqueda pintada en la lista; se anula cuando la lista se sustituye
  };
  const bar = $(".bar"), input = $("#jab-q"), clear = $(".clear"), chip = $(".chip"), menu = $("#jab-menu");
  const suggest = $("#jab-suggest"), list = $(".list"), status = $(".status"), sw = $(".switch");
  const send = (name, value) => root._send(name, value);

  // Una sola señal de cancelación para los oyentes globales si el componente se vuelve a montar.
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
    const q = state.q.trim();
    state.seq += 1;
    send("request", currentRequest());
    if (!q) { state.hasHits = false; state.renderedKey = null; list.innerHTML = ""; status.textContent = ""; bar.dataset.busy = "false"; if (state.view === "buscar") setMode("home"); return; }
    setMode("results");
    bar.dataset.busy = "true";
    if (state.hasHits) list.dataset.stale = "true"; else renderSkeleton();
  }
  const schedule = () => { clearTimeout(state.timer); state.timer = setTimeout(search, DEBOUNCE); };

  function split(text) {
    const t = text.trim().replace(/\s+/g, " ");
    const m = t.match(/^(.{20,140}?[.?!])\s/);
    if (m) return [m[1], t.slice(m[0].length)];
    if (t.length <= 110) return [t, ""];
    const cut = t.slice(0, 100).replace(/\s+\S*$/, "");
    return [cut + "…", t.slice(cut.length).trim()];
  }
  function render(r) {
    bar.dataset.busy = "false"; list.dataset.stale = "false";
    if (r.error) { state.hasHits = false; status.textContent = ""; list.innerHTML = `<li class="empty"><p>${esc(r.error)}</p></li>`; return; }
    const name = MODELS.find((m) => m.id === r.model)?.name || r.model;
    if (!r.hits.length) {
      state.hasHits = false; status.textContent = "";
      const alt = r.model === "dense" ? "" : `<button type="button" data-try-dense>Probar con Embeddings</button>`;
      list.innerHTML = `<li class="empty"><p>Sin resultados con ${esc(name)}.</p>${alt}</li>`;
      return;
    }
    state.hasHits = true;
    const n = r.hits.length, ms = r.elapsed_ms < 1 ? "menos de 1" : Math.round(r.elapsed_ms);
    status.textContent = `${n} ${n === 1 ? "resultado" : "resultados"} en ${ms} ms`;
    const top = r.hits[0].score;
    list.innerHTML = r.hits.map((h, i) => {
      const [lead, rest] = h.text.trim() ? split(h.text) : ["Documento sin texto", ""];
      const pct = top > 0 ? Math.max(0, Math.min(100, (h.score / top) * 100)) : 0;
      return `<li class="row" style="--i:${i}">
        <span class="rank num">${h.rank}</span>
        <div>
          <p class="lead" lang="en">${esc(lead)}</p>
          ${rest ? `<p class="snippet" lang="en">${esc(rest)}</p>` : ""}
          <div class="meta">
            <span class="score" title="Puntuación de ${esc(name)}; la barra es relativa al primer resultado">${h.score.toFixed(2)}<span class="score-bar"><i style="width:${pct.toFixed(0)}%"></i></span></span>
            <span title="Identificador del documento en FiQA">Doc. ${esc(h.doc_id)}</span>
            ${rest.length > 160 ? `<button class="more" type="button" aria-expanded="false">Leer todo</button>` : ""}
          </div>
        </div>
      </li>`;
    }).join("");
  }
  function renderSkeleton() {
    state.renderedKey = null;
    status.textContent = "";
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
  root.querySelectorAll(".example").forEach((b) => on(b, "click", () => { input.value = state.q = typed = b.textContent; syncClear(); search(); input.focus(); }));
  root.querySelectorAll("[data-logo]").forEach((a) => on(a, "click", (e) => { e.preventDefault(); setView("buscar"); clear.click(); }));

  // ---------- Menú del modelo (chip o Ctrl/⌘ K) ----------
  root.querySelector("[data-models]").innerHTML = MODELS.map((m, i) => `
    <button class="opt" type="button" role="menuitemradio" aria-checked="false" data-model="${m.id}">
      <span><strong>${m.name}</strong><small>${m.hint}</small></span><kbd class="kbd">${i + 1}</kbd>
    </button>`).join("");
  root.querySelector("[data-k]").innerHTML = KS.map((k) => `<button type="button" aria-pressed="false" data-kval="${k}" class="num">${k}</button>`).join("");
  function openMenu(open) {
    menu.hidden = !open; chip.setAttribute("aria-expanded", open);
    if (open) { hideSuggest(); (menu.querySelector('[aria-checked="true"]') || menu.querySelector(".opt")).focus(); }
  }
  function syncControls() {
    root.querySelector("[data-model-name]").textContent = MODELS.find((m) => m.id === state.model).name;
    menu.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-checked", b.dataset.model === state.model));
    menu.querySelectorAll("[data-kval]").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.kval === state.k));
    const dense = state.model === "dense";
    sw.disabled = dense; sw.setAttribute("aria-checked", state.stem && !dense);
    root.querySelector("[data-stem-row]").setAttribute("aria-disabled", dense);
    sw.title = dense ? "No aplica: los embeddings leen el texto original" : "";
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
