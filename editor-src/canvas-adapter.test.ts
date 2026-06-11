import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { toReactFlow, toCanvas, type Canvas } from "./canvas-adapter.ts";

const here = dirname(fileURLToPath(import.meta.url));
const load = (p: string): Canvas => JSON.parse(readFileSync(join(here, p), "utf8"));

const demo = load("../models/deepseek-v4-flash-dtype/model.canvas");
const synthetic = load("./fixtures/all-features.canvas");

test("real demo round-trips with zero data loss", () => {
  const back = toCanvas(toReactFlow(demo));
  assert.deepStrictEqual(back, demo);
});

test("synthetic fixture round-trips (all node types, colors, group bg, edge ends)", () => {
  const back = toCanvas(toReactFlow(synthetic));
  assert.deepStrictEqual(back, synthetic);
});

test("group.background + backgroundStyle survive the round-trip", () => {
  const back = toCanvas(toReactFlow(synthetic));
  const g = back.nodes.find((n) => n.id === "g1")!;
  assert.equal(g.background, "https://example.com/bg.png");
  assert.equal(g.backgroundStyle, "cover");
});

test("hex and preset colors both preserved verbatim", () => {
  const back = toCanvas(toReactFlow(synthetic));
  assert.equal(back.nodes.find((n) => n.id === "thex")!.color, "#3344ff");
  assert.equal(back.nodes.find((n) => n.id === "t2")!.color, "2");
  assert.ok(!("color" in back.nodes.find((n) => n.id === "tnone")!));
});

test("dimensions are byte-identical on an unedited round-trip", () => {
  const back = toCanvas(toReactFlow(demo));
  for (const orig of demo.nodes) {
    const got = back.nodes.find((n) => n.id === orig.id)!;
    assert.equal(got.width, orig.width);
    assert.equal(got.height, orig.height);
    assert.equal(got.x, orig.x);
    assert.equal(got.y, orig.y);
  }
});

test("edge handles get s-/t- prefixes; absent sides stay absent", () => {
  const { edges } = toReactFlow(synthetic);
  const e1 = edges.find((e) => e.id === "e1")!;
  assert.equal(e1.sourceHandle, "s-right");
  assert.equal(e1.targetHandle, "t-left");
  const e3 = edges.find((e) => e.id === "e3")!; // no sides
  assert.equal(e3.sourceHandle, undefined);
  assert.equal(e3.targetHandle, undefined);
  // export keeps e3 side-less
  const back = toCanvas({ nodes: [], edges });
  const e3back = back.edges.find((e) => e.id === "e3")!;
  assert.ok(!("fromSide" in e3back));
  assert.ok(!("toSide" in e3back));
});

test("a side used as both source and target is disambiguated by handle id", () => {
  const { edges } = toReactFlow(synthetic);
  const e4 = edges.find((e) => e.id === "e4")!; // t1 source top, t4 target top
  assert.equal(e4.sourceHandle, "s-top");
  assert.equal(e4.targetHandle, "t-top");
});

test("toEnd:'none' suppresses the arrow marker but survives export", () => {
  const { edges } = toReactFlow(synthetic);
  const e2 = edges.find((e) => e.id === "e2")!;
  assert.equal(e2.markerEnd, undefined);
  const back = toCanvas({ nodes: [], edges });
  assert.equal(back.edges.find((e) => e.id === "e2")!.toEnd, "none");
});
