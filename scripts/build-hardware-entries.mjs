// Hardware entries for the catalog: each hardware/<id>/ folder holds chip.json
// (validated by chip-schema.mjs) + meta.json. The catalog script calls
// processHardware() per folder; emitHardware() writes the chip JSON for the 3D
// view and a generated 2D .canvas for the existing viewer/editor.
import { readFileSync, readdirSync, writeFileSync, mkdirSync, rmSync, statSync, existsSync } from "node:fs";
import { join } from "node:path";
import { validateChip } from "./chip-schema.mjs";
import { chipToCanvas } from "./chip-to-canvas.mjs";

const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

export function processHardware(hardwareDir, id, fail) {
  const dir = join(hardwareDir, id);
  if (!statSync(dir).isDirectory()) return null;
  if (!idPattern.test(id)) { fail(id, "folder name must be kebab-case"); return null; }

  let meta, chip;
  try { meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8")); }
  catch (e) { fail(id, `meta.json missing or invalid JSON (${e.message})`); return null; }
  try { chip = JSON.parse(readFileSync(join(dir, "chip.json"), "utf8")); }
  catch (e) { fail(id, `chip.json missing or invalid JSON (${e.message})`); return null; }

  if (meta.kind !== "hardware") fail(id, 'meta.kind must be "hardware"');
  if (typeof meta.name !== "string" || !meta.name.trim()) fail(id, "meta.name is required");
  if (typeof meta.description !== "string" || !meta.description.trim()) fail(id, "meta.description is required");
  if (meta.tags !== undefined && !(Array.isArray(meta.tags) && meta.tags.every((t) => typeof t === "string")))
    fail(id, "meta.tags must be an array of strings");
  const errs = validateChip(chip);
  for (const e of errs) fail(id, `chip.json: ${e}`);
  if (!errs.length && chip.id !== id) fail(id, `chip.json id "${chip.id}" must match the folder name`);
  if (errs.length) return null;

  const file = `canvases/${id}.canvas`;
  return {
    id,
    kind: "hardware",
    file,
    levels: [{ label: "Diagram", file }],
    chip: `hardware-data/${id}.json`,
    name: meta.name,
    description: meta.description,
    arch: chip.arch,
    author: meta.author ?? null,
    tags: Array.isArray(meta.tags) ? meta.tags : [],
    source: typeof meta.source === "string" && meta.source ? meta.source : null,
    logo: typeof meta.logo === "string" && meta.logo.trim() ? meta.logo.trim() : null,
    _chip: chip,
  };
}

export function listHardwareIds(hardwareDir) {
  if (!existsSync(hardwareDir)) return [];
  return readdirSync(hardwareDir).filter((n) => {
    if (n.startsWith("_") || n.startsWith(".")) return false; // skip _template, dotfiles
    try { return statSync(join(hardwareDir, n)).isDirectory(); } catch { return false; }
  });
}

// Write chip JSON + generated canvas; returns the catalog entry without internals.
export function emitHardware(entry, webDir) {
  const { _chip, ...out } = entry;
  const dataDir = join(webDir, "hardware-data");
  mkdirSync(dataDir, { recursive: true });
  writeFileSync(join(dataDir, `${entry.id}.json`), JSON.stringify(_chip, null, 2) + "\n");
  writeFileSync(join(webDir, entry.file), JSON.stringify(chipToCanvas(_chip), null, 2) + "\n");
  return out;
}

export function resetHardwareData(webDir) {
  rmSync(join(webDir, "hardware-data"), { recursive: true, force: true });
}
