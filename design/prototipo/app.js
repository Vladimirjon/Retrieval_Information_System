// jab — prototipo de interacción. Busca con el motor real a través de server.py.

const MODELS = [
  { id: "jaccard", name: "Jaccard", hint: "Palabras en común" },
  { id: "tfidf", name: "TF-IDF", hint: "Palabras ponderadas por rareza" },
  { id: "bm25", name: "BM25", hint: "Ponderadas y ajustadas por longitud" },
  { id: "dense", name: "Embeddings", hint: "Por significado, no por palabras" },
];
const KS = [5, 10, 20, 50];
const DEBOUNCE = 220;

const $ = (s, r = document) => r.querySelector(s);
const state = { q: "", model: "bm25", k: 10, stem: false, hits: [], ctrl: null, timer: null, seq: 0 };

const bar = $("#bar"), input = $("#q"), clear = $(".clear"), chip = $(".chip"), menu = $("#menu");
const suggest = $("#suggest"), list = $(".list"), status = $(".status");
const sw = $(".switch");

// Logo en ambas posiciones a partir de una sola plantilla
document.querySelectorAll("[data-logo]").forEach((el) => el.append($("#logo-svg").content.cloneNode(true)));

// ---------- Modo: inicio ↔ resultados (con View Transitions) ----------
function placeBar(mode) {
  $(`[data-slot="${mode}"]`).append(bar);
}
let moving = false;
function setMode(mode) {
  if (document.body.dataset.mode === mode) return;
  const focused = document.activeElement === input;
  const reopen = !suggest.hidden;
  // Mover la barra en el DOM quita el foco: se restaura junto con las sugerencias.
  const apply = () => {
    moving = true;
    document.body.dataset.mode = mode; placeBar(mode);
    if (focused) input.focus({ preventScroll: true });
    if (reopen) showSuggest(input.value);
    moving = false;
  };
  if (document.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    document.startViewTransition(apply);
  } else apply();
}
placeBar("home");

// ---------- Búsqueda (stale-while-revalidate, cancelable) ----------
async function search() {
  const q = state.q.trim();
  if (!q) { setMode("home"); list.innerHTML = ""; status.textContent = ""; return; }
  setMode("results");
  state.ctrl?.abort();
  const ctrl = state.ctrl = new AbortController();
  const seq = ++state.seq;
  bar.dataset.busy = "true";
  if (state.hits.length) list.dataset.stale = "true"; else renderSkeleton();
  try {
    const params = new URLSearchParams({ q, model: state.model, k: state.k, stem: state.stem ? "1" : "0" });
    const res = await fetch(`/api/search?${params}`, { signal: ctrl.signal });
    const data = await res.json();
    if (seq !== state.seq) return;
    if (!res.ok) return renderError(data.error);
    state.hits = data.hits;
    render(data);
  } catch (e) {
    if (e.name !== "AbortError") renderError("No se pudo conectar con el motor.");
  } finally {
    if (seq === state.seq) { bar.dataset.busy = "false"; list.dataset.stale = "false"; }
  }
}
const schedule = () => { clearTimeout(state.timer); state.timer = setTimeout(search, DEBOUNCE); };

// ---------- Render ----------
const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function split(text) {
  const t = text.trim().replace(/\s+/g, " ");
  const m = t.match(/^(.{20,140}?[.?!])\s/);
  if (m) return [m[1], t.slice(m[0].length)];
  if (t.length <= 110) return [t, ""];
  const cut = t.slice(0, 100).replace(/\s+\S*$/, "");
  return [cut + "…", t.slice(cut.length).trim()];
}
function render(data) {
  const name = MODELS.find((m) => m.id === data.model).name;
  if (!data.hits.length) {
    status.textContent = "";
    const alt = data.model === "dense" ? "" : `<button type="button" data-try-dense>Probar con Embeddings</button>`;
    list.innerHTML = `<li class="empty"><p>Sin resultados con ${name}.</p>${alt}</li>`;
    return;
  }
  const n = data.hits.length, ms = data.elapsed_ms < 1 ? "menos de 1" : Math.round(data.elapsed_ms);
  status.textContent = `${n} ${n === 1 ? "resultado" : "resultados"} en ${ms} ms`;
  const top = data.hits[0].score;
  list.innerHTML = data.hits.map((h, i) => {
    const [lead, rest] = h.text.trim() ? split(h.text) : ["Documento sin texto", ""];
    const pct = top > 0 ? Math.max(0, Math.min(100, (h.score / top) * 100)) : 0;
    return `<li class="row" style="--i:${i}">
      <span class="rank num">${h.rank}</span>
      <div>
        <p class="lead" lang="en">${esc(lead)}</p>
        ${rest ? `<p class="snippet" lang="en">${esc(rest)}</p>` : ""}
        <div class="meta">
          <span class="score" title="Puntuación de ${name}; la barra es relativa al primer resultado">
            ${h.score.toFixed(2)}<span class="score-bar"><i style="width:${pct.toFixed(0)}%"></i></span>
          </span>
          <span title="Identificador del documento en FiQA">Doc. ${esc(h.doc_id)}</span>
          ${rest.length > 160 ? `<button class="more" type="button" aria-expanded="false">Leer todo</button>` : ""}
        </div>
      </div>
    </li>`;
  }).join("");
}
function renderSkeleton() {
  status.textContent = "";
  list.innerHTML = Array.from({ length: 4 }, () =>
    `<li class="row skeleton" aria-hidden="true"><span class="rank">0</span><div><p class="lead">.</p><p class="snippet">.</p></div></li>`).join("");
}
function renderError(msg) {
  status.textContent = "";
  list.innerHTML = `<li class="empty"><p>${esc(msg || "Algo falló.")}</p></li>`;
}
list.addEventListener("click", (e) => {
  const more = e.target.closest(".more");
  if (more) {
    const row = more.closest(".row"); const open = row.classList.toggle("open");
    more.textContent = open ? "Mostrar menos" : "Leer todo"; more.setAttribute("aria-expanded", open);
  }
  if (e.target.closest("[data-try-dense]")) setModel("dense");
});

// ---------- Sugerencias (consultas de entrenamiento de FiQA) ----------
let SUGG = [], LOWER = [], active = -1, items = [], typed = "";
fetch("/api/suggestions").then((r) => r.json()).then((d) => { SUGG = d.items; LOWER = SUGG.map((s) => s.toLowerCase()); });
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
    return `<li id="sg-${n}" role="option" aria-selected="false"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/></svg><span>${esc(s.slice(0, at))}<b>${esc(s.slice(at, at + q.length))}</b>${esc(s.slice(at + q.length))}</span></li>`;
  }).join("");
  suggest.hidden = false; input.setAttribute("aria-expanded", "true");
}
function hideSuggest() { suggest.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; }
function markSuggest() {
  [...suggest.children].forEach((li, n) => li.setAttribute("aria-selected", n === active));
  if (active >= 0) { input.value = items[active]; input.setAttribute("aria-activedescendant", `sg-${active}`); }
  else input.value = typed;
}
suggest.addEventListener("pointerdown", (e) => {
  const li = e.target.closest("li"); if (!li) return;
  e.preventDefault(); pick([...suggest.children].indexOf(li));
});
function pick(n) { input.value = state.q = items[n]; hideSuggest(); syncClear(); clearTimeout(state.timer); search(); }

// ---------- Entrada ----------
function syncClear() { clear.hidden = !input.value; }
input.addEventListener("input", () => { typed = state.q = input.value; syncClear(); showSuggest(input.value); schedule(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" && items.length) { e.preventDefault(); active = (active + 1) % items.length; markSuggest(); }
  else if (e.key === "ArrowUp" && items.length) { e.preventDefault(); active = active <= 0 ? -1 : active - 1; markSuggest(); }
  else if (e.key === "Enter") { e.preventDefault(); if (active >= 0) pick(active); else { hideSuggest(); clearTimeout(state.timer); state.q = input.value; search(); } }
  else if (e.key === "Escape" && !suggest.hidden) { hideSuggest(); input.value = typed; }
});
input.addEventListener("blur", () => setTimeout(() => { if (!moving && !bar.contains(document.activeElement)) hideSuggest(); }, 0));
clear.addEventListener("click", () => { input.value = state.q = typed = ""; syncClear(); hideSuggest(); search(); input.focus(); });
document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => {
  input.value = state.q = typed = b.textContent; syncClear(); search(); input.focus();
}));
document.querySelector("[data-home]").addEventListener("click", (e) => { e.preventDefault(); clear.click(); });

// ---------- Menú del modelo ----------
$("[data-models]").innerHTML = MODELS.map((m, i) => `
  <button class="opt" type="button" role="menuitemradio" aria-checked="${m.id === state.model}" data-model="${m.id}">
    <span><strong>${m.name}</strong><small>${m.hint}</small></span><kbd class="kbd">${i + 1}</kbd>
  </button>`).join("");
$("[data-k]").innerHTML = KS.map((k) => `<button type="button" aria-pressed="${k === state.k}" data-kval="${k}" class="num">${k}</button>`).join("");
function openMenu(open) {
  menu.hidden = !open; chip.setAttribute("aria-expanded", open);
  if (open) { hideSuggest(); (menu.querySelector('[aria-checked="true"]') || menu.querySelector(".opt")).focus(); }
}
function setModel(id) {
  state.model = id;
  $("[data-model-name]").textContent = MODELS.find((m) => m.id === id).name;
  menu.querySelectorAll("[data-model]").forEach((b) => b.setAttribute("aria-checked", b.dataset.model === id));
  const dense = id === "dense";
  sw.disabled = dense; $("[data-stem-row]").setAttribute("aria-disabled", dense);
  sw.title = dense ? "No aplica: los embeddings leen el texto original" : "";
  if (state.q.trim()) search();
}
chip.addEventListener("click", () => openMenu(menu.hidden));
menu.addEventListener("click", (e) => {
  const m = e.target.closest("[data-model]"); if (m) { setModel(m.dataset.model); openMenu(false); input.focus(); }
  const k = e.target.closest("[data-kval]");
  if (k) { state.k = +k.dataset.kval; menu.querySelectorAll("[data-kval]").forEach((b) => b.setAttribute("aria-pressed", b === k)); if (state.q.trim()) search(); }
  if (e.target.closest(".switch") && !sw.disabled) { state.stem = !state.stem; sw.setAttribute("aria-checked", state.stem); if (state.q.trim()) search(); }
});
menu.addEventListener("keydown", (e) => {
  const opts = [...menu.querySelectorAll(".opt")];
  const i = opts.indexOf(document.activeElement);
  if (e.key === "ArrowDown") { e.preventDefault(); opts[(i + 1) % opts.length].focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); opts[(i - 1 + opts.length) % opts.length].focus(); }
  if (/^[1-4]$/.test(e.key)) { e.preventDefault(); setModel(MODELS[+e.key - 1].id); openMenu(false); input.focus(); }
  if (e.key === "Escape") { openMenu(false); chip.focus(); }
});
document.addEventListener("pointerdown", (e) => { if (!menu.hidden && !menu.contains(e.target) && !chip.contains(e.target)) openMenu(false); });

// ---------- Atajos globales ----------
document.addEventListener("keydown", (e) => {
  const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openMenu(menu.hidden); }
  else if (e.key === "/" && !typing) { e.preventDefault(); input.focus(); }
});

// ---------- Paleta (solo prototipo) ----------
document.querySelectorAll("[data-palette-btn]").forEach((b) => b.addEventListener("click", () => {
  document.documentElement.dataset.palette = b.dataset.paletteBtn;
  document.querySelectorAll("[data-palette-btn]").forEach((x) => x.setAttribute("aria-pressed", x === b));
}));
