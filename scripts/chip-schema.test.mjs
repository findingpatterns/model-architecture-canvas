import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateChip } from "./chip-schema.mjs";

const load = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const template = () => load("../hardware/_template/chip.json");

test("shipped chips and the template are valid", () => {
  for (const p of ["../hardware/_template/chip.json", "../hardware/b200/chip.json", "../hardware/h200/chip.json"])
    assert.deepEqual(validateChip(load(p)), [], p);
});

test("dangling parent, flow and step refs are reported", () => {
  const c = template();
  c.components[1].parent = "nope";
  c.flows[0].to = "ghost";
  c.steps[0].focus = ["missing"];
  const errs = validateChip(c).join("\n");
  assert.match(errs, /parent "nope"/);
  assert.match(errs, /flows\[0\]/);
  assert.match(errs, /focus id "missing"/);
});

test("bad vectors and undeclared groups are reported", () => {
  const c = template();
  c.components[0].geom.pos = [0, "1", 0];
  c.components[0].group = "unknown";
  const errs = validateChip(c).join("\n");
  assert.match(errs, /geom.pos/);
  assert.match(errs, /group "unknown"/);
});

test("specs need a source URL or estimate flag", () => {
  const c = template();
  c.components[1].specs = [{ label: "X", value: "1" }];
  assert.match(validateChip(c).join("\n"), /source URL or estimate/);
  c.components[1].specs = [{ label: "X", value: "1", estimate: true }];
  assert.deepEqual(validateChip(c), []);
});

test("duplicate ids across scenes are reported", () => {
  const c = template();
  c.scenes = { s: { anchor: "die", components: [{ ...c.components[1], parent: undefined }] } };
  assert.match(validateChip(c).join("\n"), /duplicate id/);
});

test("summary rows need a known key, no duplicates, and a source or estimate", () => {
  const c = template();
  c.summary = [{ key: "nope", value: "1" }, { key: "dies", value: "1", estimate: true }, { key: "dies", value: "2", estimate: true }];
  const errs = validateChip(c).join("\n");
  assert.match(errs, /unknown key "nope"/);
  assert.match(errs, /duplicate key "dies"/);
  assert.match(errs, /summary\[0\]: needs a source URL or estimate/);
});

test("roles and stages must be kebab-case; stages unique", () => {
  const c = template();
  c.components[1].role = "Bad Role";
  c.steps[0].stage = "hbm";
  c.steps[1].stage = "hbm";
  const errs = validateChip(c).join("\n");
  assert.match(errs, /role must be kebab-case/);
  assert.match(errs, /stage values must be unique/);
});
