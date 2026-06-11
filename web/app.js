// ModelCanvas web viewer — preview + download for pre-generated .canvas files.
//
// Verified json-canvas-viewer@4.3.2 API (from its dist type defs):
//   - new JSONCanvasViewer({ container, parser }, [Minimap, Controls])
//       option key is `parser` (NOT `markdownParser`); modules are the 2nd arg.
//   - viewer.load({ canvas }) swaps the displayed canvas on the SAME instance (no recreate).
//   - viewer.dispose() tears the instance down (unused here — one instance, reused).
//   - fetchCanvas(url) loads a .canvas by URL and returns parsed JSONCanvas data.
import {
  JSONCanvasViewer,
  parser,
  fetchCanvas,
  Minimap,
  Controls,
} from "https://unpkg.com/json-canvas-viewer@4.3.2";

// The catalog is generated from the models/ folder by scripts/build-catalog.mjs
// and served as ./catalog.json. Each entry:
//   { id, file, name, description, author, tags, source }
// Contributors never edit this file — they add a folder under models/ and open a PR.
let CATALOG = [];

const els = {
  list: document.getElementById("catalog-list"),
  viewer: document.getElementById("viewer"),
  download: document.getElementById("download"),
  activeTitle: document.getElementById("active-title"),
  source: document.getElementById("source"),
  app: document.querySelector(".app"),
  sidebar: document.getElementById("sidebar"),
  toggle: document.getElementById("sidebar-toggle"),
  themeToggle: document.getElementById("theme-toggle"),
};

let viewer = null;       // single reused JSONCanvasViewer instance
let activeId = null;
let loadSeq = 0;         // guards against out-of-order fetches from rapid switching

// Show a plain text message in the viewer area (e.g. error / unknown model).
// Uses textContent — never interpolate untrusted strings into innerHTML.
// If a viewer instance exists its DOM is being replaced, so tear it down and
// force a fresh instance on the next successful load.
function showViewerMessage(text) {
  if (viewer) {
    viewer.dispose();
    viewer = null;
  }
  els.viewer.replaceChildren();
  const p = document.createElement("p");
  p.className = "catalog-empty";
  p.style.padding = "24px";
  p.textContent = text;
  els.viewer.appendChild(p);
}

// Build one DOM element with a class + text. textContent only — never interpolate
// catalog strings into innerHTML (defense-in-depth even though the catalog is repo-controlled).
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

// Render the sidebar list from CATALOG. Empty catalog → graceful placeholder.
function renderCatalog() {
  els.list.replaceChildren();
  if (CATALOG.length === 0) {
    els.list.appendChild(
      el("p", "catalog-empty", "No models yet — open a PR adding a folder under models/ to contribute one."),
    );
    return;
  }
  for (const entry of CATALOG) {
    const item = el("button", "catalog-item");
    item.type = "button";
    item.dataset.id = entry.id;

    item.appendChild(el("div", "item-name", entry.name));
    item.appendChild(el("span", "item-id mono", entry.id));
    item.appendChild(el("div", "item-note", entry.description));

    // Author credit (optional).
    if (entry.author?.name) {
      const by = el("div", "item-meta", `by ${entry.author.name}`);
      item.appendChild(by);
    }

    // Tag pills (optional).
    if (Array.isArray(entry.tags) && entry.tags.length) {
      const tags = el("div", "item-tags");
      for (const t of entry.tags) tags.appendChild(el("span", "tag", t));
      item.appendChild(tags);
    }

    item.addEventListener("click", () => {
      loadModel(entry.id);
      closeSidebarOnMobile();
    });
    els.list.appendChild(item);
  }
}

function setActiveHighlight(id) {
  for (const item of els.list.querySelectorAll(".catalog-item")) {
    const isActive = item.dataset.id === id;
    item.classList.toggle("is-active", isActive);
    if (isActive) item.setAttribute("aria-current", "true");
    else item.removeAttribute("aria-current");
  }
}

// Centralized model load/swap — single owner of viewer lifecycle + URL sync + download wiring.
async function loadModel(id) {
  const entry = CATALOG.find((m) => m.id === id);
  if (!entry) {
    showViewerMessage(`Unknown model: ${id}`);
    return;
  }
  activeId = id;
  setActiveHighlight(id);
  els.activeTitle.textContent = entry.name;

  // Download serves the original file bytes (no re-serialization that could drop fields).
  els.download.href = entry.file;
  els.download.setAttribute("download", `${entry.id}.canvas`);
  els.download.hidden = false;

  // Optional source link (paper / repo the diagram is based on).
  if (entry.source) {
    els.source.href = entry.source;
    els.source.hidden = false;
  } else {
    els.source.hidden = true;
    els.source.removeAttribute("href");
  }

  // Keep the URL shareable. replaceState (not pushState) avoids one history entry per click.
  history.replaceState(null, "", `?model=${encodeURIComponent(id)}`);

  const reqId = ++loadSeq;
  try {
    const canvas = await fetchCanvas(entry.file);
    if (reqId !== loadSeq) return; // a newer switch superseded this one
    if (!viewer) {
      viewer = new JSONCanvasViewer(
        { container: els.viewer, canvas, parser },
        [Minimap, Controls],
      );
    } else {
      viewer.load({ canvas }); // swap on the existing instance
    }
  } catch (err) {
    if (reqId !== loadSeq) return;
    // Failed load: clear the stale download target/title/source so they don't mislead.
    els.download.hidden = true;
    els.source.hidden = true;
    els.activeTitle.textContent = "";
    showViewerMessage(`Failed to load ${entry.file}: ${err?.message ?? err}`);
  }
}

// Collapse the drawer after picking a model on mobile (no effect on desktop).
function closeSidebarOnMobile() {
  if (mobileQuery.matches) setSidebarVisible(false);
}

// ---- Sidebar visibility ----
// Desktop default: shown (toggle collapses it, viewer reclaims the space).
// Mobile default: hidden drawer (toggle slides it in over the viewer).
// One boolean drives two classes so each breakpoint flips from its own default
// with no first-paint flash (no class on <html> = each breakpoint's default).
const mobileQuery = window.matchMedia("(max-width: 768px)");

function setSidebarVisible(visible) {
  els.app.classList.toggle("sidebar-open", visible);       // mobile: reveal drawer
  els.app.classList.toggle("sidebar-collapsed", !visible); // desktop: collapse panel
  els.toggle.setAttribute("aria-expanded", String(visible));
}

els.toggle.addEventListener("click", () => {
  setSidebarVisible(!els.app.classList.contains("sidebar-open"));
});

// Keep state sane across the breakpoint boundary.
mobileQuery.addEventListener("change", (e) => setSidebarVisible(!e.matches));

// ---- Theme: dark (default brand) / light, persisted in localStorage. ----
const THEME_KEY = "modelcanvas-theme";

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const isLight = theme === "light";
  els.themeToggle.textContent = isLight ? "🌙" : "☀";       // icon = the theme you'd switch TO
  els.themeToggle.setAttribute("aria-pressed", String(isLight));
}

function initialTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") return saved;
  // First visit: respect the OS preference, defaulting to the dark brand look.
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

els.themeToggle.addEventListener("click", () => {
  const next =
    document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  applyTheme(next);
  localStorage.setItem(THEME_KEY, next);
});

applyTheme(initialTheme());

// Resolve the initial model: ?model=<id> if valid, else the first catalog entry.
function initialModelId() {
  const requested = new URLSearchParams(location.search).get("model");
  if (requested && CATALOG.some((m) => m.id === requested)) return requested;
  return CATALOG[0]?.id ?? null;
}

setSidebarVisible(!mobileQuery.matches); // shown on desktop, collapsed drawer on mobile

// Load the generated catalog, then render the sidebar + open the initial model.
async function bootstrap() {
  try {
    const res = await fetch("./catalog.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`catalog.json → HTTP ${res.status}`);
    CATALOG = await res.json();
  } catch (err) {
    renderCatalog(); // shows the empty-state message
    showViewerMessage(`Could not load the catalog: ${err?.message ?? err}. Run \`npm run build\` to generate it.`);
    return;
  }
  renderCatalog();
  const startId = initialModelId();
  if (startId) loadModel(startId);
}

bootstrap();
