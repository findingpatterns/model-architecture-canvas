// Compare view (?compare=a,b): a spec table aligned by standard key (with bars and the
// left ÷ right ratio) above the two models' canvases side by side, driven by shared
// detail-level tabs. Rows come from each catalog entry's `summary` (web/model-summary.js).
import { JSONCanvasViewer, parser, fetchCanvas, Controls } from "https://unpkg.com/json-canvas-viewer@4.3.2";
import { el, isHardware } from "./gallery.js";
import { formatRatio, isHttpUrl, joinModelSummary } from "./model-summary.js";

const $ = (id) => document.getElementById(id);
const viewers = [null, null];
let theme = "dark";
let pair = null; // [entryA, entryB]
let levelLabel = null; // shared tab label; each side falls back to its first level
let loadSeq = 0;

const levelsOf = (e) => (Array.isArray(e.levels) && e.levels.length ? e.levels : [{ label: "Diagram", file: e.file }]);
const compareUrl = (a, b, label) =>
  `?compare=${encodeURIComponent(a)},${encodeURIComponent(b)}${label ? `&level=${encodeURIComponent(label)}` : ""}`;

// Union of both models' level labels, in A's order then B's extras.
function sharedLabels([a, b]) {
  const labels = levelsOf(a).map((l) => l.label);
  for (const l of levelsOf(b)) if (!labels.includes(l.label)) labels.push(l.label);
  return labels;
}

function cell(row, max, modelName) {
  const td = el("td");
  if (!row) {
    td.className = "muted";
    td.textContent = "not listed";
    td.title = `No value in ${modelName}'s meta.json summary for this spec.`;
    return td;
  }
  const line = td.appendChild(el("div", "cmp-line"));
  line.appendChild(el("span", "cmp-value", row.value));
  if (typeof row.number === "number" && max > 0) {
    const bar = el("div", "cmp-bar");
    bar.setAttribute("aria-hidden", "true");
    const fill = el("span");
    fill.style.width = `${row.number > 0 ? Math.max(2, (row.number / max) * 100) : 0}%`;
    bar.appendChild(fill);
    td.appendChild(bar);
  }
  const meta = el("div", "cmp-meta mono");
  if (row.estimate) meta.appendChild(el("span", "cmp-estimate", "estimate"));
  if (isHttpUrl(row.source)) {
    const a = el("a", "cmp-src", "source ↗");
    a.href = row.source;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    meta.appendChild(a);
  }
  if (meta.childNodes.length) line.appendChild(meta);
  return td;
}

function renderTable([a, b]) {
  const rows = joinModelSummary(a.summary, b.summary);
  const wrap = $("compare-specs");
  wrap.replaceChildren();
  if (!rows.length) { wrap.appendChild(el("p", "muted", "Neither model has a spec summary yet.")); return; }
  const table = el("table", "compare-table");
  const head = el("tr");
  for (const t of ["Spec", a.name, b.name, `${a.name} ÷ ${b.name}`]) head.appendChild(Object.assign(el("th", null, t), { scope: "col" }));
  table.appendChild(el("thead")).appendChild(head);
  const body = table.appendChild(el("tbody"));
  for (const r of rows) {
    const tr = el("tr");
    const th = el("th", null, r.label);
    th.scope = "row";
    const note = r.a?.note ?? r.b?.note;
    if (note) { th.title = note; th.classList.add("has-note"); }
    tr.appendChild(th);
    const max = Math.max(r.a?.number ?? 0, r.b?.number ?? 0);
    tr.appendChild(cell(r.a, max, a.name));
    tr.appendChild(cell(r.b, max, b.name));
    tr.appendChild(el("td", "cmp-ratio mono", r.ratio !== null ? formatRatio(r.ratio) : "—"));
    body.appendChild(tr);
  }
  wrap.appendChild(table);
}

function renderPickers(catalog, [a, b]) {
  const models = catalog.filter((e) => !isHardware(e));
  [["compare-pick-a", a, b], ["compare-pick-b", b, a]].forEach(([id, self, other], side) => {
    const sel = $(id);
    sel.replaceChildren(...models.map((m) => Object.assign(el("option", null, m.name), { value: m.id, disabled: m.id === other.id })));
    sel.value = self.id;
    sel.onchange = () => {
      const ids = side === 0 ? [sel.value, other.id] : [other.id, sel.value];
      location.search = compareUrl(ids[0], ids[1], levelLabel);
    };
  });
  $("compare-swap").onclick = () => { location.search = compareUrl(b.id, a.id, levelLabel); };
  $("compare-title").textContent = `${a.name} vs ${b.name}`;
  [a, b].forEach((e, i) => {
    const link = $(`compare-open-${i}`);
    link.textContent = `${e.name} ↗`;
    link.href = `?model=${encodeURIComponent(e.id)}`;
  });
}

function renderTabs(labels) {
  const tabs = $("compare-level-tabs");
  tabs.replaceChildren();
  tabs.hidden = labels.length < 2;
  for (const label of labels) {
    const t = el("button", "level-tab", label);
    t.type = "button";
    if (label === levelLabel) { t.classList.add("is-active"); t.setAttribute("aria-current", "true"); }
    t.addEventListener("click", () => loadLevel(label));
    tabs.appendChild(t);
  }
}

async function loadLevel(label) {
  const labels = sharedLabels(pair);
  levelLabel = labels.includes(label) ? label : labels[0];
  renderTabs(labels);
  history.replaceState(null, "", compareUrl(pair[0].id, pair[1].id, levelLabel === labels[0] ? null : levelLabel));
  const req = ++loadSeq;
  await Promise.all(pair.map(async (entry, i) => {
    const levels = levelsOf(entry);
    const level = levels.find((l) => l.label === levelLabel) ?? levels[0];
    const host = $(`compare-viewer-${i}`);
    $(`compare-level-${i}`).textContent =
      level.label === levelLabel || levels.length < 2 ? "" : `no "${levelLabel}" level — showing ${level.label}`;
    try {
      const canvas = await fetchCanvas(level.file);
      if (req !== loadSeq) return;
      viewers[i]?.dispose();
      host.replaceChildren();
      viewers[i] = new JSONCanvasViewer({ container: host, canvas, parser, theme }, [Controls]);
      requestAnimationFrame(() => viewers[i]?.resetView?.());
    } catch (err) {
      if (req !== loadSeq) return;
      viewers[i]?.dispose();
      viewers[i] = null;
      host.replaceChildren(el("p", "viewer-message", `Failed to load ${level.file}: ${err?.message ?? err}`));
    }
  }));
}

export function changeCompareTheme(next) {
  theme = next;
  for (const v of viewers) v?.changeTheme(next);
}

// Returns false when the ids don't name two models (caller falls back to the gallery).
export function showCompare(catalog, ids, requestedLevel, initialTheme) {
  const find = (id) => catalog.find((e) => e.id === id && !isHardware(e));
  const a = find(ids[0]);
  const b = find(ids[1]);
  if (!a || !b) return false;
  pair = [a, b];
  theme = initialTheme;
  document.title = `${a.name} vs ${b.name} — ModelCanvas`;
  $("gallery-view").hidden = true;
  $("compare-view").hidden = false;
  renderPickers(catalog, pair);
  renderTable(pair);
  // Refit each pane once it gets a real size (it may be laid out after the first fit).
  pair.forEach((_, i) => new ResizeObserver(([e]) => {
    if (e.contentRect.width && e.contentRect.height) viewers[i]?.resetView?.();
  }).observe($(`compare-viewer-${i}`)));
  loadLevel(requestedLevel);
  return true;
}

// Default pair for "Compare" entry points: `id` against the first other model with a summary.
export function defaultPartner(catalog, id) {
  const models = catalog.filter((e) => !isHardware(e) && e.id !== id);
  return (models.find((e) => Array.isArray(e.summary)) ?? models[0])?.id ?? null;
}
export { compareUrl };
