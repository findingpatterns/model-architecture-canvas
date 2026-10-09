import { test } from "node:test";
import assert from "node:assert/strict";
import { formatRatio, joinModelSummary, parseComparePair, validateModelSummary } from "../web/model-summary.js";

const SRC = "https://huggingface.co/org/repo/blob/main/config.json";

test("validateModelSummary accepts sourced and estimated rows", () => {
  assert.deepEqual(validateModelSummary([
    { key: "layers", value: "40", number: 40, source: SRC },
    { key: "attention", value: "GQA 32/8", estimate: true },
  ]), []);
});

test("validateModelSummary rejects unknown, duplicate, unsourced and malformed rows", () => {
  const errs = validateModelSummary([
    { key: "flops", value: "1", source: SRC },
    { key: "layers", value: "40", source: SRC },
    { key: "layers", value: "40", source: SRC },
    { key: "vocab", value: "1" },
    { key: "context", value: "1M", number: -1, source: SRC },
    { key: "hidden_size", value: "", source: "ftp://x" },
  ]);
  assert.ok(errs.some((e) => e.includes('unknown key "flops"')));
  assert.ok(errs.some((e) => e.includes('duplicate key "layers"')));
  assert.ok(errs.some((e) => e.startsWith("summary[3]") && e.includes("estimate: true")));
  assert.ok(errs.some((e) => e.startsWith("summary[4]") && e.includes("number")));
  assert.ok(errs.some((e) => e.startsWith("summary[5]") && e.includes("value is required")));
  assert.ok(errs.some((e) => e.startsWith("summary[5]") && e.includes("http(s)")));
  assert.deepEqual(validateModelSummary({}), ["summary must be an array of rows"]);
});

test("joinModelSummary aligns by standard key order and computes left ÷ right", () => {
  const a = [{ key: "layers", value: "40", number: 40 }, { key: "params_total", value: "763B", number: 763 }];
  const b = [{ key: "params_total", value: "27.8B", number: 27.8 }, { key: "attention", value: "GQA" }];
  const rows = joinModelSummary(a, b);
  assert.deepEqual(rows.map((r) => r.key), ["params_total", "layers", "attention"]);
  assert.ok(Math.abs(rows[0].ratio - 763 / 27.8) < 1e-9);
  assert.equal(rows[1].b, undefined);
  assert.equal(rows[1].ratio, null);
  assert.equal(rows[2].ratio, null);
});

test("joinModelSummary gives no ratio for zero values and tolerates a missing summary", () => {
  const rows = joinModelSummary([{ key: "experts", value: "dense", number: 0 }], undefined);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].ratio, null);
});

test("formatRatio and parseComparePair", () => {
  assert.equal(formatRatio(27.45), "27×");
  assert.equal(formatRatio(2.64), "2.6×");
  assert.equal(formatRatio(0.618), "0.62×");
  assert.deepEqual(parseComparePair("a, b"), ["a", "b"]);
  assert.equal(parseComparePair("a,a"), null);
  assert.equal(parseComparePair("a"), null);
  assert.equal(parseComparePair(null), null);
});
