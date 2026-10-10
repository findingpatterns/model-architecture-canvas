// Gallery search + facet filters. Pure (no DOM): facet values are derived from each
// catalog entry so contributors never have to fill in extra fields.
//   models   → architecture (dense / MoE), stored size bucket, inputs (text / multimodal)
//   hardware → vendor, chip type (GPU / accelerator)
// Filter state lives in the URL (?q=&arch=&size=&input=&vendor=&type=) so a view can be shared.

const summaryRow = (entry, key) => (Array.isArray(entry.summary) ? entry.summary.find((r) => r.key === key) : undefined);

const SIZE_BUCKETS = [
  { value: "lt15", label: "< 15B", max: 15 },
  { value: "15-70", label: "15–70B", max: 70 },
  { value: "70-500", label: "70–500B", max: 500 },
  { value: "500plus", label: "500B+", max: Infinity },
];

const VENDORS = {
  nvidia: "NVIDIA", amd: "AMD", google: "Google", aws: "AWS", intel: "Intel",
  cerebras: "Cerebras", groq: "Groq", sambanova: "SambaNova", tenstorrent: "Tenstorrent",
};

// Facet definitions per category: each maps an entry to one value (or null = unknown).
export const FACETS = {
  models: [
    {
      param: "arch",
      label: "Architecture",
      options: [{ value: "dense", label: "Dense" }, { value: "moe", label: "MoE" }],
      valueOf: (e) => {
        const n = summaryRow(e, "experts")?.number;
        return typeof n === "number" ? (n > 0 ? "moe" : "dense") : null;
      },
    },
    {
      param: "size",
      label: "Stored params",
      options: SIZE_BUCKETS.map(({ value, label }) => ({ value, label })),
      valueOf: (e) => {
        const n = summaryRow(e, "params_total")?.number;
        return typeof n === "number" ? SIZE_BUCKETS.find((b) => n < b.max).value : null;
      },
    },
    {
      param: "input",
      label: "Inputs",
      options: [{ value: "text", label: "Text only" }, { value: "multimodal", label: "Multimodal" }],
      valueOf: (e) => {
        const v = summaryRow(e, "modalities")?.value;
        if (typeof v !== "string") return null;
        return /image|video|audio|vision/i.test(v) ? "multimodal" : "text";
      },
    },
  ],
  hardware: [
    {
      param: "vendor",
      label: "Vendor",
      options: Object.entries(VENDORS).map(([value, label]) => ({ value, label })),
      valueOf: (e) => (e.tags ?? []).find((t) => Object.hasOwn(VENDORS, t)) ?? null,
    },
    {
      param: "type",
      label: "Type",
      options: [{ value: "gpu", label: "GPU" }, { value: "accelerator", label: "Other accelerator" }],
      valueOf: (e) => ((e.tags ?? []).includes("gpu") ? "gpu" : "accelerator"),
    },
  ],
};

// Text that the search box matches against.
function haystack(e) {
  return [e.name, e.id, e.description, e.arch, ...(e.tags ?? []), ...(e.summary ?? []).map((r) => r.value)]
    .filter((s) => typeof s === "string")
    .join(" ")
    .toLowerCase();
}

// Read filter state for a category from URLSearchParams; unknown values are dropped.
export function readFilters(category, params) {
  const state = { q: (params.get("q") ?? "").trim() };
  for (const f of FACETS[category] ?? []) {
    const v = params.get(f.param);
    state[f.param] = f.options.some((o) => o.value === v) ? v : "";
  }
  return state;
}

// Write filter state into URLSearchParams (empty values are removed).
export function writeFilters(category, state, params) {
  for (const key of ["q", ...(FACETS[category] ?? []).map((f) => f.param)]) {
    if (state[key]) params.set(key, state[key]);
    else params.delete(key);
  }
}

// Entries matching the search text (every word must appear) and every selected facet.
export function filterEntries(entries, category, state) {
  const words = (state.q ?? "").toLowerCase().split(/\s+/).filter(Boolean);
  const facets = (FACETS[category] ?? []).filter((f) => state[f.param]);
  return entries.filter((e) => {
    if (words.length) {
      const text = haystack(e);
      if (!words.every((w) => text.includes(w))) return false;
    }
    return facets.every((f) => f.valueOf(e) === state[f.param]);
  });
}

// Per-option counts within the entries that match every *other* active filter,
// so each chip shows how many results selecting it would give.
export function facetCounts(entries, category, state) {
  const counts = {};
  for (const f of FACETS[category] ?? []) {
    const base = filterEntries(entries, category, { ...state, [f.param]: "" });
    counts[f.param] = Object.fromEntries(f.options.map((o) => [o.value, base.filter((e) => f.valueOf(e) === o.value).length]));
  }
  return counts;
}

export const hasActiveFilters = (category, state) =>
  Boolean(state.q) || (FACETS[category] ?? []).some((f) => state[f.param]);
