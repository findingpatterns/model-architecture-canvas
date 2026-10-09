import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { chipToCanvas } from "./chip-to-canvas.mjs";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));

for (const id of ["b200", "h200"]) {
  test(`${id}: generated canvas is valid JSON Canvas with resolvable edges`, () => {
    const chip = load(`../hardware/${id}/chip.json`);
    const canvas = chipToCanvas(chip);
    assert.ok(Array.isArray(canvas.nodes) && canvas.nodes.length > 0);
    const ids = new Set(canvas.nodes.map((n) => n.id));
    assert.equal(ids.size, canvas.nodes.length, "node ids are unique");
    for (const e of canvas.edges) {
      assert.ok(ids.has(e.fromNode), `edge ${e.id} fromNode`);
      assert.ok(ids.has(e.toNode), `edge ${e.id} toNode`);
    }
    const componentCount = chip.components.length + Object.values(chip.scenes ?? {}).reduce((n, s) => n + s.components.length, 0);
    assert.equal(canvas.nodes.filter((n) => n.type === "text").length, componentCount + 1, "one node per component + title");
  });
}

test("output is deterministic", () => {
  const chip = load("../hardware/b200/chip.json");
  assert.deepEqual(chipToCanvas(chip), chipToCanvas(chip));
});
