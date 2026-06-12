// ModelCanvas web app — a gallery of models; opening one shows the full canvas.
//
// Two views in one page, routed by the ?model= query param:
//   - no/invalid ?model  → gallery grid of model cards (landing)
//   - ?model=<id>        → full-screen json-canvas-viewer for that model
// Cards are plain <a href="?model=id"> links, so navigation + the browser Back
// button "just work" with no SPA router. The /editor "← Preview" link targets
// ?model=<id>, landing straight on the canvas view.
//
// Verified json-canvas-viewer@4.3.2 API:
//   new JSONCanvasViewer({ container, canvas, parser, theme }, [Minimap, Controls])
//   viewer.load({ canvas }) / viewer.dispose() / viewer.changeTheme(theme)
//   fetchCanvas(url) → parsed JSONCanvas
import {
  JSONCanvasViewer,
  parser,
  fetchCanvas,
  Minimap,
  Controls,
} from "https://unpkg.com/json-canvas-viewer@4.3.2";

// Generated from models/ by scripts/build-catalog.mjs → ./catalog.json.
// Entry: { id, file, name, description, author, tags, source }
let CATALOG = [];

const els = {
  galleryView: document.getElementById("gallery-view"),
  galleryGrid: document.getElementById("gallery-grid"),
  galleryLead: document.getElementById("gallery-lead"),
  canvasView: document.getElementById("canvas-view"),
  viewer: document.getElementById("viewer"),
  download: document.getElementById("download"),
  activeTitle: document.getElementById("active-title"),
  source: document.getElementById("source"),
  edit: document.getElementById("edit"),
  themeToggle: document.getElementById("theme-toggle"),
  themeToggleGallery: document.getElementById("theme-toggle-gallery"),
};

let viewer = null;       // single reused JSONCanvasViewer instance
let currentTheme = "dark";

// Build an element with a class + text. textContent only (no innerHTML interpolation).
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function showViewerMessage(text) {
  if (viewer) { viewer.dispose(); viewer = null; }
  els.viewer.replaceChildren();
  const p = el("p", "viewer-message", text);
  els.viewer.appendChild(p);
}

// ---- Gallery ----
// One-line intro to the model lines currently in the catalog (auto-updates as models are added).
function renderLead() {
  const names = CATALOG.map((m) => m.name);
  if (names.length === 0) { els.galleryLead.textContent = ""; return; }
  els.galleryLead.textContent =
    names.length === 1
      ? `Currently featuring the ${names[0]} architecture.`
      : `Currently featuring ${names.length} architectures — ${names.join(", ")}.`;
}

function renderGallery() {
  renderLead();
  els.galleryGrid.replaceChildren();
  if (CATALOG.length === 0) {
    els.galleryGrid.appendChild(
      el("p", "gallery-empty", "No models yet — open a PR adding a folder under models/ to contribute one."),
    );
    return;
  }
  for (const entry of CATALOG) {
    const card = el("a", "model-card");
    card.href = `?model=${encodeURIComponent(entry.id)}`;

    card.appendChild(el("div", "card-name", entry.name));
    card.appendChild(el("span", "card-id mono", entry.id));
    card.appendChild(el("p", "card-note", entry.description));

    if (Array.isArray(entry.tags) && entry.tags.length) {
      const tags = el("div", "card-tags");
      for (const t of entry.tags) tags.appendChild(el("span", "tag", t));
      card.appendChild(tags);
    }

    const foot = el("div", "card-foot");
    if (entry.author?.name) foot.appendChild(el("span", "card-author mono", `by ${entry.author.name}`));
    foot.appendChild(el("span", "card-open mono", "Open →"));
    card.appendChild(foot);

    els.galleryGrid.appendChild(card);
  }
}

// ---- Canvas (full viewer for one model) ----
async function loadModel(entry) {
  els.activeTitle.textContent = entry.name;

  // Download serves original file bytes (no re-serialization that could drop fields).
  els.download.href = entry.file;
  els.download.setAttribute("download", `${entry.id}.canvas`);
  els.download.hidden = false;

  els.edit.href = `editor/?model=${encodeURIComponent(entry.id)}`;
  els.edit.hidden = false;

  if (entry.source) {
    els.source.href = entry.source;
    els.source.hidden = false;
  } else {
    els.source.hidden = true;
    els.source.removeAttribute("href");
  }

  try {
    const canvas = await fetchCanvas(entry.file);
    viewer = new JSONCanvasViewer(
      { container: els.viewer, canvas, parser, theme: currentTheme },
      [Minimap, Controls],
    );
  } catch (err) {
    els.download.hidden = true;
    els.source.hidden = true;
    els.edit.hidden = true;
    showViewerMessage(`Failed to load ${entry.file}: ${err?.message ?? err}`);
  }
}

function showGallery() {
  els.canvasView.hidden = true;
  els.galleryView.hidden = false;
  renderGallery();
}

function showCanvas(entry) {
  els.galleryView.hidden = true;
  els.canvasView.hidden = false;
  loadModel(entry);
}

// ---- Theme (dark default; shared localStorage key with the editor) ----
const THEME_KEY = "modelcanvas-theme";

function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  const isLight = theme === "light";
  const icon = isLight ? "🌙" : "☀"; // icon = the theme you'd switch TO
  for (const btn of [els.themeToggle, els.themeToggleGallery]) {
    btn.textContent = icon;
    btn.setAttribute("aria-pressed", String(isLight));
  }
  viewer?.changeTheme(theme);
}

function toggleTheme() {
  const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
}
els.themeToggle.addEventListener("click", toggleTheme);
els.themeToggleGallery.addEventListener("click", toggleTheme);

const initialTheme = (() => {
  const saved = localStorage.getItem(THEME_KEY);
  return saved === "light" || saved === "dark" ? saved : "dark"; // dark default
})();
applyTheme(initialTheme);

// ---- Bootstrap: load catalog, then route to gallery or canvas ----
async function bootstrap() {
  try {
    const res = await fetch("./catalog.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`catalog.json → HTTP ${res.status}`);
    CATALOG = await res.json();
  } catch (err) {
    showGallery();
    els.galleryGrid.replaceChildren(
      el("p", "gallery-empty", `Could not load the catalog: ${err?.message ?? err}. Run \`npm run build\` to generate it.`),
    );
    return;
  }

  const requested = new URLSearchParams(location.search).get("model");
  const entry = requested && CATALOG.find((m) => m.id === requested);
  if (entry) showCanvas(entry);
  else showGallery();
}

bootstrap();
