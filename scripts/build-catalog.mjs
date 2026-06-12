#!/usr/bin/env node
// Build the web catalog from the contributor-facing models/ folder.
//
// Each model is a directory under models/ containing:
//   - exactly one *.canvas file  (the JSON Canvas 1.0 diagram)
//   - meta.json                  ({ name, description, author?, tags?, source? })
//
// Default run: validate every model, then emit web/catalog.json and copy each
//   canvas to web/canvases/<id>.canvas. The id is the folder name.
// --check: validate only (used by the PR CI gate); writes nothing, exits 1 on error.
//
// No third-party deps — Node built-ins only, so CI needs no `npm install`.
import { readdirSync, readFileSync, statSync, existsSync, mkdirSync, writeFileSync, rmSync, copyFileSync } from "node:fs";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODELS_DIR = join(ROOT, "models");
const WEB_DIR = join(ROOT, "web");
const OUT_CANVASES = join(WEB_DIR, "canvases");
const OUT_LOGOS = join(WEB_DIR, "logos");
const OUT_CATALOG = join(WEB_DIR, "catalog.json");

const checkOnly = process.argv.includes("--check");
const errors = [];
const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/; // kebab-case

function fail(id, msg) {
  errors.push(`  ✗ ${id}: ${msg}`);
}

function isHttpUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

function slugify(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// Validate a .canvas file is JSON with a nodes array. Records a failure if not.
function validCanvas(id, path) {
  try {
    const c = JSON.parse(readFileSync(path, "utf8"));
    if (!Array.isArray(c.nodes)) { fail(id, `${path} has no \`nodes\` array (not valid JSON Canvas)`); return false; }
    return true;
  } catch (e) {
    fail(id, `${path} is not valid JSON (${e.message})`);
    return false;
  }
}

// Validate one model directory; returns a catalog entry, or null if invalid.
function processModel(id) {
  const dir = join(MODELS_DIR, id);
  if (!statSync(dir).isDirectory()) return null;

  if (!idPattern.test(id)) {
    fail(id, "folder name must be kebab-case (lowercase letters, digits, hyphens)");
    return null;
  }

  // meta.json
  let meta;
  try {
    meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
  } catch (e) {
    fail(id, `meta.json missing or invalid JSON (${e.message})`);
    return null;
  }
  if (typeof meta.name !== "string" || !meta.name.trim()) fail(id, "meta.name is required (non-empty string)");
  if (typeof meta.description !== "string" || !meta.description.trim()) fail(id, "meta.description is required (non-empty string)");
  if (meta.tags !== undefined && !(Array.isArray(meta.tags) && meta.tags.every((t) => typeof t === "string")))
    fail(id, "meta.tags must be an array of strings");
  if (meta.source !== undefined && meta.source !== "" && !isHttpUrl(meta.source))
    fail(id, "meta.source must be an http(s) URL");
  if (meta.author !== undefined && (typeof meta.author !== "object" || typeof meta.author.name !== "string"))
    fail(id, "meta.author must be an object with a name string");
  if (meta.logo !== undefined && typeof meta.logo !== "string")
    fail(id, "meta.logo must be a string (a filename in this folder, an http(s) URL, or a short glyph/emoji)");

  // Resolve logo: a local file in the model folder → copy it; otherwise pass the
  // string through (emoji / short text / http URL). Trademark note: only commit
  // logo image files you have the right to use.
  let logoFile = null; // abs path of a local image to copy
  let logoValue = null; // final catalog value (path / url / glyph)
  if (typeof meta.logo === "string" && meta.logo.trim()) {
    const candidate = join(dir, meta.logo);
    if (existsSync(candidate) && statSync(candidate).isFile()) logoFile = candidate;
    else logoValue = meta.logo.trim(); // emoji / short text / external URL
  }

  // Resolve diagram levels (tabs). Either explicit, ordered, labeled meta.levels
  // (any number) — or, for back-compat, the single .canvas in the folder as one
  // implicit level. The viewer shows tabs only when there are 2+ levels.
  const canvasFiles = readdirSync(dir).filter((f) => f.endsWith(".canvas"));
  let levelSpecs = []; // [{ label, src(absPath) }]
  if (meta.levels !== undefined) {
    if (!Array.isArray(meta.levels) || meta.levels.length === 0) {
      fail(id, "meta.levels must be a non-empty array of { label, file }");
      return null;
    }
    const seenLabels = new Set();
    for (const [i, lv] of meta.levels.entries()) {
      if (!lv || typeof lv.label !== "string" || !lv.label.trim() || typeof lv.file !== "string" || !lv.file.trim()) {
        fail(id, `meta.levels[${i}] must be { label: string, file: string }`);
        continue;
      }
      if (seenLabels.has(lv.label)) fail(id, `meta.levels has a duplicate label "${lv.label}"`);
      seenLabels.add(lv.label);
      const src = join(dir, lv.file);
      if (!existsSync(src) || !statSync(src).isFile()) { fail(id, `meta.levels[${i}].file "${lv.file}" not found in folder`); continue; }
      if (!validCanvas(id, src)) continue;
      levelSpecs.push({ label: lv.label.trim(), src });
    }
    if (levelSpecs.length === 0) return null;
  } else {
    if (canvasFiles.length !== 1) {
      fail(id, `expected exactly one .canvas file (or declare meta.levels for multiple), found ${canvasFiles.length}`);
      return null;
    }
    const src = join(dir, canvasFiles[0]);
    if (!validCanvas(id, src)) return null;
    levelSpecs = [{ label: "Diagram", src }];
  }

  // Assign output filenames: single level keeps `<id>.canvas`; multiple use `<id>--<slug>.canvas`.
  const multi = levelSpecs.length > 1;
  const usedNames = new Set();
  const levels = []; // catalog: [{ label, file }]
  const _levels = []; // internal copy list: [{ src, dest }]
  levelSpecs.forEach((lv, i) => {
    const base = multi ? `${id}--${slugify(lv.label) || `level-${i + 1}`}` : id;
    let dest = `${base}.canvas`;
    let n = 2;
    while (usedNames.has(dest)) dest = `${base}-${n++}.canvas`;
    usedNames.add(dest);
    levels.push({ label: lv.label, file: `canvases/${dest}` });
    _levels.push({ src: lv.src, dest });
  });

  return {
    id,
    file: levels[0].file, // default level (back-compat for editor + download)
    levels,
    name: meta.name,
    description: meta.description,
    author: meta.author ?? null,
    tags: Array.isArray(meta.tags) ? meta.tags : [],
    source: typeof meta.source === "string" && meta.source ? meta.source : null,
    logo: logoValue, // glyph / url, or replaced with a copied path below
    _levels, // internal; copied below
    _logoFile: logoFile, // internal; copied below if set
  };
}

// --- main ---
let modelIds = [];
try {
  modelIds = readdirSync(MODELS_DIR).filter((name) => {
    if (name.startsWith("_") || name.startsWith(".")) return false; // skip _template, dotfiles
    try { return statSync(join(MODELS_DIR, name)).isDirectory(); } catch { return false; }
  });
} catch {
  console.error(`No models/ directory at ${MODELS_DIR}`);
  process.exit(1);
}

const entries = modelIds.map(processModel).filter(Boolean);
entries.sort((a, b) => a.name.localeCompare(b.name));

if (errors.length) {
  console.error(`\nCatalog validation failed (${errors.length} issue${errors.length > 1 ? "s" : ""}):`);
  console.error(errors.join("\n"));
  process.exit(1);
}

console.log(`Validated ${entries.length} model${entries.length === 1 ? "" : "s"}: ${entries.map((e) => e.id).join(", ") || "(none)"}`);

if (checkOnly) {
  console.log("--check passed (no files written).");
  process.exit(0);
}

// Emit: fresh canvases/ + logos/ + catalog.json
rmSync(OUT_CANVASES, { recursive: true, force: true });
rmSync(OUT_LOGOS, { recursive: true, force: true });
mkdirSync(OUT_CANVASES, { recursive: true });
mkdirSync(OUT_LOGOS, { recursive: true });
const catalog = entries.map(({ _levels, _logoFile, ...entry }) => {
  for (const lv of _levels) copyFileSync(lv.src, join(OUT_CANVASES, lv.dest));
  if (_logoFile) {
    const dest = `${entry.id}${extname(_logoFile)}`;
    copyFileSync(_logoFile, join(OUT_LOGOS, dest));
    entry.logo = `logos/${dest}`;
  }
  return entry;
});
writeFileSync(OUT_CATALOG, JSON.stringify(catalog, null, 2) + "\n");
console.log(`Wrote ${OUT_CATALOG}, ${catalog.length} canvas file(s), and logos to ${WEB_DIR}`);
