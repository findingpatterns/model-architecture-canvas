import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { alignStages, effectiveStepView, formatRatio, joinSummary, roleDiff } from "./compare-diff.ts";
import type { Chip } from "../chip-types.ts";

const load = (id: string): Chip => JSON.parse(readFileSync(new URL(`../../hardware/${id}/chip.json`, import.meta.url), "utf8"));
const h200 = load("h200");
const b200 = load("b200");

test("summary rows align by key with ratio and better side", () => {
  const rows = joinSummary(h200, b200);
  const bw = rows.find((r) => r.key === "hbm_bandwidth")!;
  assert.ok(Math.abs(bw.ratio! - 8 / 4.8) < 1e-9);
  assert.equal(bw.better, "b");
  const sm = rows.find((r) => r.key === "sm_count")!;
  assert.ok(sm.a && !sm.b, "keys reported by only one chip are kept");
  assert.equal(sm.ratio, null);
  const proc = rows.find((r) => r.key === "process")!;
  assert.equal(proc.better, null, "non-directional keys never pick a side");
});

test("role diff finds Blackwell-only parts and nothing H200-only", () => {
  const d = roleDiff(h200, b200);
  assert.deepEqual(d.onlyA, []);
  assert.deepEqual(d.onlyB.map((x) => x.role).sort(), ["die-to-die-link", "second-compute-die", "tensor-memory"]);
  assert.ok(d.shared.includes("tensor-core"));
});

test("walkthrough stages align; missing stages are null", () => {
  const pairs = alignStages(h200, b200);
  const d2d = pairs.find((p) => p.stage === "die-to-die")!;
  assert.equal(d2d.a, null);
  assert.equal(d2d.b!.index, 2);
  assert.deepEqual(pairs.slice(0, 3).map((p) => p.stage), ["hbm", "memory-controller", "die-to-die"]);
  assert.ok(pairs.every((p) => p.a || p.b));
});

test("step drill/explode carry forward until a later step changes them", () => {
  const tc = b200.steps!.findIndex((s) => s.stage === "accumulate");
  assert.deepEqual(effectiveStepView(b200, tc), { drill: "sm-detail", explode: 0.6 });
  assert.equal(effectiveStepView(b200, b200.steps!.length - 1).drill, null);
  assert.equal(effectiveStepView(b200, 1).drill, null);
});

test("ratios format both ways", () => {
  assert.equal(formatRatio(2), "×2.0");
  assert.equal(formatRatio(0.5), "÷2.0");
  assert.equal(formatRatio(12), "×12");
});
