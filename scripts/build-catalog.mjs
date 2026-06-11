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
import { readdirSync, readFileSync, statSync, mkdirSync, writeFileSync, rmSync, copyFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODELS_DIR = join(ROOT, "models");
const WEB_DIR = join(ROOT, "web");
const OUT_CANVASES = join(WEB_DIR, "canvases");
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

  // exactly one .canvas
  const canvases = readdirSync(dir).filter((f) => f.endsWith(".canvas"));
  if (canvases.length !== 1) {
    fail(id, `expected exactly one .canvas file, found ${canvases.length}`);
    return null;
  }
  const canvasPath = join(dir, canvases[0]);
  let canvasRaw;
  try {
    canvasRaw = readFileSync(canvasPath, "utf8");
    const canvas = JSON.parse(canvasRaw);
    if (!Array.isArray(canvas.nodes)) fail(id, "canvas has no `nodes` array (not a valid JSON Canvas)");
  } catch (e) {
    fail(id, `canvas is not valid JSON (${e.message})`);
    return null;
  }

  return {
    id,
    file: `canvases/${id}.canvas`,
    name: meta.name,
    description: meta.description,
    author: meta.author ?? null,
    tags: Array.isArray(meta.tags) ? meta.tags : [],
    source: typeof meta.source === "string" && meta.source ? meta.source : null,
    _canvasPath: canvasPath, // internal; stripped before write
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

// Emit: fresh canvases/ + catalog.json
rmSync(OUT_CANVASES, { recursive: true, force: true });
mkdirSync(OUT_CANVASES, { recursive: true });
const catalog = entries.map(({ _canvasPath, ...entry }) => {
  copyFileSync(_canvasPath, join(OUT_CANVASES, `${entry.id}.canvas`));
  return entry;
});
writeFileSync(OUT_CATALOG, JSON.stringify(catalog, null, 2) + "\n");
console.log(`Wrote ${OUT_CATALOG} and ${catalog.length} canvas file(s) to ${OUT_CANVASES}`);
