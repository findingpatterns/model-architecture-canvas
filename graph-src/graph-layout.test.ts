import { test } from "node:test";
import assert from "node:assert/strict";
import { barWidth, layoutGraph, liftEdges } from "./graph-layout.ts";
import type { ModelGraph } from "./graph-types.ts";

const g: ModelGraph = {
  model: "t", name: "t", source: "t", totalParams: 0,
  nodes: [
    { id: "in", label: "in", kind: "io", dimOut: 512 },
    { id: "blk", label: "blk", kind: "group", repeat: 2 },
    { id: "a", label: "a", kind: "linear", parent: "blk", dimOut: 1280 },
    { id: "b", label: "b", kind: "linear", parent: "blk", dimOut: 32768 },
    { id: "out", label: "out", kind: "io" },
  ],
  edges: [["in", "blk"], ["in", "a"], ["a", "b"], ["b", "out"], ["blk", "out"]],
};

test("bar width grows with channels in every size mode and is constant when off", () => {
  for (const m of ["sqrt", "log", "linear"] as const) assert.ok(barWidth(32768, m) > barWidth(512, m));
  assert.equal(barWidth(512, "off"), barWidth(32768, "off"));
});

test("collapsed group swallows its children's edges", () => {
  const e = liftEdges(g, new Set());
  assert.deepEqual(e.map((x) => x.id).sort(), ["blk->out", "in->blk"]);
});

test("expanded group exposes children and keeps them inside the parent", () => {
  const l = layoutGraph(g, new Set(["blk"]), "sqrt");
  const ids = l.nodes.map((n) => n.id);
  assert.ok(ids.indexOf("blk") < ids.indexOf("a"), "parent precedes child");
  assert.equal(l.nodes.find((n) => n.id === "a")!.parent, "blk");
  assert.ok(l.edges.some((e) => e.id === "a->b"));
  const a = l.nodes.find((n) => n.id === "a")!;
  const b = l.nodes.find((n) => n.id === "b")!;
  assert.ok(b.y > a.y, "dataflow goes top to bottom");
  assert.ok(b.bar > a.bar, "wider tensor gets a wider bar");
});
