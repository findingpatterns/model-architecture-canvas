#!/usr/bin/env node
// Post-build safety check (red-team #11): confirm the Vite editor build did NOT
// wipe the catalog/viewer. Guards against an outDir misconfiguration (e.g. ../web
// with emptyOutDir:true) deleting the whole static site. Exits non-zero on loss.
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const required = [
  "web/index.html",
  "web/app.js",
  "web/styles.css",
  "web/canvases",
  "web/editor/index.html",
];

const missing = required.filter((p) => !existsSync(join(ROOT, p)));
if (missing.length) {
  console.error(`Build integrity check FAILED — missing after build:\n${missing.map((m) => "  - " + m).join("\n")}`);
  process.exit(1);
}
console.log("Build integrity OK — viewer, catalog, and editor all present.");
