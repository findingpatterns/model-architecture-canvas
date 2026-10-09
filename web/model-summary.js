// Standard spec keys for comparing two model architectures, shared by the catalog
// build (validation) and the web compare view (rendering). Pure — no DOM, no fs.
//
// A model's meta.json may carry `summary`: [{ key, value, number?, note?, source?, estimate? }]
//   value  — display text ("16.1B", "MLA + DSA, 64 heads")
//   number — the comparable magnitude in `unit` (enables the bar and the A ÷ B ratio)
//   note   — how the value was derived, shown as a tooltip
//   source — http(s) URL the value comes from; without one the row must say estimate: true
//
// higherIsBetter is null for every key: a bigger architecture is neither better nor worse,
// so the table shows magnitudes and ratios without declaring a winner.
export const MODEL_SUMMARY_KEYS = {
  params_total: { label: "Stored parameters", unit: "B" },
  params_active: { label: "Active per text token", unit: "B" },
  layers: { label: "Decoder layers", unit: "" },
  hidden_size: { label: "Hidden size", unit: "" },
  attention: { label: "Attention", unit: null },
  experts: { label: "Routed experts", unit: "" },
  experts_active: { label: "Experts per token", unit: "" },
  vocab: { label: "Vocabulary", unit: "" },
  context: { label: "Max context", unit: "tokens" },
  weight_dtype: { label: "Main weight dtype", unit: null },
  modalities: { label: "Inputs", unit: null },
  drafting: { label: "Built-in drafting", unit: null },
};

const isStr = (v) => typeof v === "string" && v.trim().length > 0;
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export function isHttpUrl(value) {
  try {
    const u = new URL(value);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}

// Returns a list of problems (empty when valid).
export function validateModelSummary(summary) {
  if (!Array.isArray(summary)) return ["summary must be an array of rows"];
  const errs = [];
  const seen = new Set();
  summary.forEach((r, i) => {
    const w = `summary[${i}]`;
    if (r === null || typeof r !== "object") { errs.push(`${w}: must be an object`); return; }
    if (!Object.hasOwn(MODEL_SUMMARY_KEYS, r.key)) errs.push(`${w}: unknown key "${r.key}" (see MODEL_SUMMARY_KEYS)`);
    if (seen.has(r.key)) errs.push(`${w}: duplicate key "${r.key}"`);
    seen.add(r.key);
    if (!isStr(r.value)) errs.push(`${w}: value is required (non-empty string)`);
    if (r.number !== undefined && !(isNum(r.number) && r.number >= 0)) errs.push(`${w}: number must be a finite number ≥ 0`);
    if (r.note !== undefined && !isStr(r.note)) errs.push(`${w}: note must be a non-empty string`);
    const sourced = typeof r.source === "string" && isHttpUrl(r.source);
    if (r.source !== undefined && !sourced) errs.push(`${w}: source must be an http(s) URL`);
    if (!sourced && r.estimate !== true) errs.push(`${w}: needs a source URL or estimate: true`);
  });
  return errs;
}

// Rows in MODEL_SUMMARY_KEYS order, keeping keys at least one side reports.
// ratio = a / b (left ÷ right, matching the "A vs B" title) when both are numeric and non-zero.
export function joinModelSummary(a, b) {
  const byKey = (s) => new Map((Array.isArray(s) ? s : []).map((r) => [r.key, r]));
  const ma = byKey(a);
  const mb = byKey(b);
  return Object.entries(MODEL_SUMMARY_KEYS)
    .filter(([k]) => ma.has(k) || mb.has(k))
    .map(([key, meta]) => {
      const ra = ma.get(key);
      const rb = mb.get(key);
      const na = ra?.number;
      const nb = rb?.number;
      const ratio = isNum(na) && isNum(nb) && na !== 0 && nb !== 0 ? na / nb : null;
      return { key, label: meta.label, unit: meta.unit, a: ra, b: rb, ratio };
    });
}

// "2.6×" = left is 2.6 times the right; "0.62×" = about 3/5 of it.
export function formatRatio(r) {
  return `${r >= 10 ? r.toFixed(0) : r >= 1 ? r.toFixed(1) : r.toFixed(2)}×`;
}

// Parse "?compare=a,b" into two distinct ids, or null.
export function parseComparePair(param) {
  if (typeof param !== "string") return null;
  const ids = param.split(",").map((s) => s.trim()).filter(Boolean);
  return ids.length === 2 && ids[0] !== ids[1] ? ids : null;
}
