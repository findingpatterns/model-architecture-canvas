// chip.json schema validator — the single source of truth for a hardware chip
// (component tree + illustrative 3D geometry + specs with sources + flows + steps).
//
// Pure ESM, no dependencies: used by the catalog script (Node, CI without npm
// install) AND imported by the 3D app in the browser.
//
// validateChip(obj) → string[]  (empty array = valid)

export const SCHEMA_VERSION = 1;
export const GEOM_TYPES = ["box", "cylinder"];
const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const colorPattern = /^#[0-9a-fA-F]{6}$/;

const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const isStr = (v) => typeof v === "string" && v.trim().length > 0;
const isNum = (v) => typeof v === "number" && Number.isFinite(v);
const isVec3 = (v) => Array.isArray(v) && v.length === 3 && v.every(isNum);

function isHttpUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

// All components across the main tree and every drill-down scene, tagged with their scene.
export function allComponents(chip) {
  const out = (chip.components ?? []).map((c) => ({ c, scene: null }));
  for (const [sid, s] of Object.entries(chip.scenes ?? {}))
    for (const c of s?.components ?? []) out.push({ c, scene: sid });
  return out;
}

function checkComponent(c, where, groups, sceneIds, errs) {
  if (!isObj(c)) { errs.push(`${where}: must be an object`); return; }
  if (!isStr(c.id) || !idPattern.test(c.id)) errs.push(`${where}: id must be kebab-case`);
  if (!isStr(c.name)) errs.push(`${where}: name is required`);
  if (!groups.has(c.group)) errs.push(`${where}: group "${c.group}" is not declared in groups`);
  if (c.color !== undefined && !(typeof c.color === "string" && colorPattern.test(c.color)))
    errs.push(`${where}: color must be #rrggbb`);
  if (!isObj(c.geom)) errs.push(`${where}: geom is required`);
  else {
    if (!GEOM_TYPES.includes(c.geom.type)) errs.push(`${where}: geom.type must be one of ${GEOM_TYPES.join(", ")}`);
    if (!isVec3(c.geom.size) || c.geom.size.some((n) => n <= 0)) errs.push(`${where}: geom.size must be 3 positive numbers`);
    if (!isVec3(c.geom.pos)) errs.push(`${where}: geom.pos must be 3 numbers`);
  }
  if (c.explode !== undefined && !isNum(c.explode)) errs.push(`${where}: explode must be a number`);
  if (c.repeat !== undefined && c.repeat !== null) {
    const r = c.repeat;
    if (!isObj(r) || !isVec3(r.count) || r.count.some((n) => !Number.isInteger(n) || n < 1) || !isVec3(r.step))
      errs.push(`${where}: repeat must be { count: [int>=1 ×3], step: [3 numbers], spread?: number }`);
    else if (r.spread !== undefined && !isNum(r.spread)) errs.push(`${where}: repeat.spread must be a number`);
  }
  if (c.desc !== undefined && typeof c.desc !== "string") errs.push(`${where}: desc must be a string`);
  if (c.drill !== undefined && !sceneIds.has(c.drill)) errs.push(`${where}: drill "${c.drill}" is not a declared scene`);
  if (c.specs !== undefined) {
    if (!Array.isArray(c.specs)) errs.push(`${where}: specs must be an array`);
    else c.specs.forEach((s, i) => {
      const w = `${where}.specs[${i}]`;
      if (!isObj(s) || !isStr(s.label) || !isStr(s.value)) { errs.push(`${w}: needs label and value strings`); return; }
      const sourced = typeof s.source === "string" && isHttpUrl(s.source);
      if (s.source !== undefined && !sourced) errs.push(`${w}: source must be an http(s) URL`);
      if (!sourced && s.estimate !== true) errs.push(`${w}: needs a source URL or estimate: true`);
    });
  }
}

export function validateChip(chip) {
  const errs = [];
  if (!isObj(chip)) return ["chip must be a JSON object"];
  if (chip.schemaVersion !== SCHEMA_VERSION) errs.push(`schemaVersion must be ${SCHEMA_VERSION}`);
  for (const k of ["id", "name", "vendor", "arch"]) if (!isStr(chip[k])) errs.push(`${k} is required`);
  if (isStr(chip.id) && !idPattern.test(chip.id)) errs.push("id must be kebab-case");
  if (!isObj(chip.groups) || !Object.values(chip.groups).every((v) => typeof v === "string" && colorPattern.test(v)))
    errs.push("groups must map names to #rrggbb colors");
  if (!Array.isArray(chip.components) || chip.components.length === 0) errs.push("components must be a non-empty array");
  if (chip.scenes !== undefined && !isObj(chip.scenes)) errs.push("scenes must be an object");
  if (errs.length) return errs;

  const groups = new Set(Object.keys(chip.groups));
  const sceneIds = new Set(Object.keys(chip.scenes ?? {}));
  const all = allComponents(chip);
  const ids = new Set();
  for (const { c, scene } of all) {
    const where = scene ? `scenes.${scene}.${c?.id ?? "?"}` : `components.${c?.id ?? "?"}`;
    checkComponent(c, where, groups, sceneIds, errs);
    if (c?.id) {
      if (ids.has(c.id)) errs.push(`${where}: duplicate id`);
      ids.add(c.id);
    }
  }
  const mainIds = new Set(chip.components.map((c) => c.id));
  for (const { c, scene } of all) {
    if (c?.parent === undefined || c.parent === null) continue;
    const pool = scene ? new Set(chip.scenes[scene].components.map((x) => x.id)) : mainIds;
    if (!pool.has(c.parent)) errs.push(`${c.id}: parent "${c.parent}" not found in the same scene`);
  }
  for (const [sid, s] of Object.entries(chip.scenes ?? {})) {
    if (!isObj(s) || !Array.isArray(s.components)) { errs.push(`scenes.${sid}: needs a components array`); continue; }
    if (!mainIds.has(s.anchor)) errs.push(`scenes.${sid}: anchor "${s.anchor}" must be a main component id`);
    if (s.offset !== undefined && !isVec3(s.offset)) errs.push(`scenes.${sid}: offset must be 3 numbers`);
  }
  (chip.flows ?? []).forEach((f, i) => {
    const w = `flows[${i}]`;
    if (!isObj(f)) { errs.push(`${w}: must be an object`); return; }
    if (!mainIds.has(f.from) || !mainIds.has(f.to)) errs.push(`${w}: from/to must be main component ids`);
    if (!groups.has(f.group)) errs.push(`${w}: group "${f.group}" not declared`);
    if (f.particles !== undefined && !(Number.isInteger(f.particles) && f.particles > 0)) errs.push(`${w}: particles must be a positive integer`);
  });
  (chip.steps ?? []).forEach((s, i) => {
    const w = `steps[${i}]`;
    if (!isObj(s) || !isStr(s.text)) { errs.push(`${w}: needs text`); return; }
    for (const id of s.focus ?? []) if (!ids.has(id)) errs.push(`${w}: focus id "${id}" not found`);
    if (s.dot !== undefined && !ids.has(s.dot)) errs.push(`${w}: dot id "${s.dot}" not found`);
    if (s.drill !== undefined && s.drill !== null && !sceneIds.has(s.drill)) errs.push(`${w}: drill "${s.drill}" not a scene`);
    if (s.explode !== undefined && !(isNum(s.explode) && s.explode >= 0 && s.explode <= 1)) errs.push(`${w}: explode must be 0..1`);
  });
  return errs;
}
