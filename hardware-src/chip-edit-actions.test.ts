import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateChip } from "../scripts/chip-schema.mjs";
import { addComponent, deleteComponent, duplicateComponent, moveComponent, resizeComponent } from "./chip-edit-actions.ts";
import { serializeChip } from "./chip-serialize.ts";
import { instanceOffsets, liftOf } from "./chip-geometry.ts";
import type { Chip } from "./chip-types.ts";

const path = (id: string) => new URL(`../hardware/${id}/chip.json`, import.meta.url);
const load = (id: string): Chip => JSON.parse(readFileSync(path(id), "utf8"));

test("load → serialize is byte-identical for shipped chips", () => {
  for (const id of ["b200", "h200"]) assert.equal(serializeChip(load(id)), readFileSync(path(id), "utf8"), id);
});

test("every edit keeps the chip valid", () => {
  let c = load("b200");
  c = moveComponent(c, "die0", [-2.12345, 0.8, 0.1]);
  c = resizeComponent(c, "nv-hbi", [0.3, 0.35, 4]);
  const added = addComponent(c, "die0");
  c = added.chip;
  const dup = duplicateComponent(c, "tensor-core");
  c = dup.chip;
  assert.deepEqual(validateChip(c), []);
  assert.deepEqual(c.components.find((x) => x.id === "die0")!.geom.pos, [-2.123, 0.8, 0.1]);
  assert.ok(c.scenes!["sm-detail"].components.some((x) => x.id === dup.id));
});

test("deleting a component cascades to children, flows, steps and anchored scenes", () => {
  const c = deleteComponent(load("b200"), "die0");
  assert.deepEqual(validateChip(c), []);
  const ids = new Set(c.components.map((x) => x.id));
  for (const gone of ["die0", "l2-die0", "mc-die0", "sm-die0-a"]) assert.ok(!ids.has(gone), gone);
  assert.equal(c.scenes!["sm-detail"], undefined, "scene anchored on sm-die0-a removed");
  assert.ok(!c.flows!.some((f) => f.from === "die0" || f.to === "die0"));
  assert.ok(c.steps!.every((s) => s.drill !== "sm-detail"));
});

test("repeat expands to count instances and spread lifts upper layers", () => {
  const dram = load("b200").components.find((x) => x.id === "hbm-dram-left")!;
  const inst = instanceOffsets(dram);
  assert.equal(inst.length, 32);
  assert.ok(liftOf(dram, 1, 7) > liftOf(dram, 1, 0));
});
