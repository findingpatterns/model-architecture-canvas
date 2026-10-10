import { test } from "node:test";
import assert from "node:assert/strict";
import { facetCounts, filterEntries, hasActiveFilters, readFilters, writeFilters } from "../web/gallery-filters.js";

const row = (key, value, number) => ({ key, value, number, estimate: true });
const models = [
  { id: "dense-small", name: "Gemma Small", tags: ["gemma"], summary: [row("experts", "dense", 0), row("params_total", "12B", 12), row("modalities", "text, image")] },
  { id: "moe-big", name: "Kimi Big", tags: ["moe"], description: "KDA hybrid", summary: [row("experts", "384", 384), row("params_total", "1T", 1000), row("modalities", "text")] },
  { id: "no-summary", name: "Mystery", tags: [] },
];

test("search matches every word across name, tags, description and summary values", () => {
  assert.deepEqual(filterEntries(models, "models", { q: "kda" }).map((e) => e.id), ["moe-big"]);
  assert.deepEqual(filterEntries(models, "models", { q: "gemma image" }).map((e) => e.id), ["dense-small"]);
  assert.deepEqual(filterEntries(models, "models", { q: "gemma kda" }), []);
});

test("facets derive architecture, size bucket and inputs from the summary", () => {
  assert.deepEqual(filterEntries(models, "models", { q: "", arch: "moe" }).map((e) => e.id), ["moe-big"]);
  assert.deepEqual(filterEntries(models, "models", { q: "", size: "lt15" }).map((e) => e.id), ["dense-small"]);
  assert.deepEqual(filterEntries(models, "models", { q: "", input: "text" }).map((e) => e.id), ["moe-big"]);
  // An entry without a summary only drops out once a facet is selected.
  assert.equal(filterEntries(models, "models", { q: "" }).length, 3);
});

test("facet counts ignore the facet's own selection but respect the others", () => {
  const c = facetCounts(models, "models", { q: "", arch: "dense", size: "" });
  assert.deepEqual(c.arch, { dense: 1, moe: 1 });
  assert.equal(c.size.lt15, 1);
  assert.equal(c.size["500plus"], 0);
});

test("hardware facets come from tags", () => {
  const chips = [{ id: "h100", name: "H100", tags: ["nvidia", "gpu"] }, { id: "lpu", name: "LPU", tags: ["groq", "lpu"] }];
  assert.deepEqual(filterEntries(chips, "hardware", { q: "", vendor: "groq" }).map((e) => e.id), ["lpu"]);
  assert.deepEqual(filterEntries(chips, "hardware", { q: "", type: "gpu" }).map((e) => e.id), ["h100"]);
});

test("URL state round-trips and drops unknown values", () => {
  const params = new URLSearchParams("q=kimi&arch=moe&size=bogus&vendor=amd");
  const state = readFilters("models", params);
  assert.deepEqual(state, { q: "kimi", arch: "moe", size: "", input: "" });
  assert.ok(hasActiveFilters("models", state));
  const out = new URLSearchParams("category=models&size=lt15");
  writeFilters("models", state, out);
  assert.equal(out.toString(), "category=models&q=kimi&arch=moe");
});
