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
import { el, renderGallery as renderGalleryView, isHardware } from "./gallery.js";

// Generated from models/ by scripts/build-catalog.mjs → ./catalog.json.
// Entry: { id, file, name, description, author, tags, source, kind?, chip? }
// kind: "hardware" entries also have a 3D explorer at hardware/?chip=<id>.
let CATALOG = [];

const els = {
  galleryView: document.getElementById("gallery-view"),
  galleryGrid: document.getElementById("gallery-grid"),
  galleryLead: document.getElementById("gallery-lead"),
  hardwareGrid: document.getElementById("hardware-grid"),
  hardwareSection: document.getElementById("hardware-section"),
  modelsHeading: document.getElementById("models-heading"),
  open3d: document.getElementById("open-3d"),
  compareLink: document.getElementById("compare-link"),
  canvasView: document.getElementById("canvas-view"),
  viewer: document.getElementById("viewer"),
  download: document.getElementById("download"),
  activeTitle: document.getElementById("active-title"),
  source: document.getElementById("source"),
  edit: document.getElementById("edit"),
  levelTabs: document.getElementById("level-tabs"),
  themeToggle: document.getElementById("theme-toggle"),
  themeToggleGallery: document.getElementById("theme-toggle-gallery"),
};

let viewer = null;       // single reused JSONCanvasViewer instance
let currentTheme = "dark";
let loadSeq = 0;         // guards against out-of-order level fetches

function showViewerMessage(text) {
  if (viewer) { viewer.dispose(); viewer = null; }
  els.viewer.replaceChildren();
  const p = el("p", "viewer-message", text);
  els.viewer.appendChild(p);
}

// ---- Canvas (full viewer for one model, with detail-level tabs) ----
let activeEntry = null;

// A model has one or more `levels` (detail variants). levels[0] is the default.
// Tabs show only when there are 2+ levels.
function levelsOf(entry) {
  return Array.isArray(entry.levels) && entry.levels.length
    ? entry.levels
    : [{ label: "Diagram", file: entry.file }];
}

function renderTabs(entry, activeIdx) {
  const levels = levelsOf(entry);
  els.levelTabs.replaceChildren();
  if (levels.length < 2) { els.levelTabs.hidden = true; return; }
  els.levelTabs.hidden = false;
  levels.forEach((lv, i) => {
    const tab = el("button", "level-tab", lv.label);
    tab.type = "button";
    if (i === activeIdx) { tab.classList.add("is-active"); tab.setAttribute("aria-current", "true"); }
    tab.addEventListener("click", () => loadLevel(i));
    els.levelTabs.appendChild(tab);
  });
}

// Load one detail level: swap the canvas, retarget download/edit, sync the URL.
async function loadLevel(idx) {
  const entry = activeEntry;
  const levels = levelsOf(entry);
  idx = Math.max(0, Math.min(idx, levels.length - 1));
  const level = levels[idx];
  renderTabs(entry, idx);

  // Download serves the original file bytes (no re-serialization that could drop fields).
  const suffix = levels.length > 1 ? `-${slugify(level.label)}` : "";
  els.download.href = level.file;
  els.download.setAttribute("download", `${entry.id}${suffix}.canvas`);
  els.download.hidden = false;

  els.edit.href = `editor/?model=${encodeURIComponent(entry.id)}${idx ? `&level=${idx}` : ""}`;
  els.edit.hidden = false;

  // Shareable URL: model + level (level omitted when default).
  history.replaceState(null, "", `?model=${encodeURIComponent(entry.id)}${idx ? `&level=${idx}` : ""}`);

  const reqId = ++loadSeq;
  try {
    const canvas = await fetchCanvas(level.file);
    if (reqId !== loadSeq) return;
    if (!viewer) {
      viewer = new JSONCanvasViewer({ container: els.viewer, canvas, parser, theme: currentTheme }, [Minimap, Controls]);
    } else {
      viewer.load({ canvas }); // reuse instance when switching tabs
    }
  } catch (err) {
    if (reqId !== loadSeq) return;
    showViewerMessage(`Failed to load ${level.file}: ${err?.message ?? err}`);
  }
}

function slugify(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function showGallery() {
  els.canvasView.hidden = true;
  els.galleryView.hidden = false;
  renderGalleryView(CATALOG, els);
}

function showCanvas(entry, levelIdx) {
  activeEntry = entry;
  els.galleryView.hidden = true;
  els.canvasView.hidden = false;
  els.activeTitle.textContent = entry.name;
  if (isHardware(entry)) { els.open3d.href = `hardware/?chip=${encodeURIComponent(entry.id)}`; els.open3d.hidden = false; }
  else els.open3d.hidden = true;
  if (entry.source) { els.source.href = entry.source; els.source.hidden = false; }
  else { els.source.hidden = true; els.source.removeAttribute("href"); }
  loadLevel(levelIdx || 0);
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

  const params = new URLSearchParams(location.search);
  const requested = params.get("model");
  const entry = requested && CATALOG.find((m) => m.id === requested);
  if (entry) showCanvas(entry, parseInt(params.get("level"), 10) || 0);
  else showGallery();
}

bootstrap();
