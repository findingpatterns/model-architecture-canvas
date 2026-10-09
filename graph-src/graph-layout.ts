// Pure layout for the graph viewer: collapse/expand groups, lift edges to the visible
// representative, then lay each container out in longest-path layers (top → bottom).
// Positions of children are relative to their parent group (React Flow convention).
import type { GraphEdge, GraphNode, ModelGraph, SizeMode } from "./graph-types.ts";

export const LABEL_W = 170; // min footprint so captions stay readable at any bar width
export const LEAF_H = 64;
const GROUP_PAD = 24;
const GROUP_HEAD = 40;
const ROW_GAP = 46;
const COL_GAP = 36;
const LINEAR_REF = 32768; // linear mode: this many channels = 600px
const MAX_BAR = 1100;

/** Bar width in px for a tensor of `dim` channels. */
export function barWidth(dim: number | undefined, mode: SizeMode): number {
  if (mode === "off" || !dim) return 150;
  if (mode === "sqrt") return Math.max(4, Math.round(3 * Math.sqrt(dim)));
  if (mode === "log") return Math.max(4, Math.round(36 * Math.log2(dim + 1)));
  return Math.min(MAX_BAR, Math.max(2, Math.round((dim / LINEAR_REF) * 600)));
}

/** Edge stroke width for a tensor of `dim` channels (only in size modes). */
export function edgeWidth(dim: number | undefined, mode: SizeMode): number {
  if (mode === "off" || !dim) return 1.5;
  return Math.min(14, 1 + Math.sqrt(dim) / 16);
}

export interface PlacedNode {
  id: string;
  node: GraphNode;
  parent?: string;
  x: number;
  y: number;
  w: number;
  h: number;
  bar: number;
  expanded: boolean; // only meaningful for groups
}

export interface PlacedEdge {
  id: string;
  from: string;
  to: string;
  label?: string;
  dim?: number;
}

export interface Layout {
  nodes: PlacedNode[]; // parents always precede their children
  edges: PlacedEdge[];
}

export function isGroup(g: ModelGraph, id: string): boolean {
  return g.nodes.some((n) => n.parent === id);
}

/** Nearest visible stand-in for `id`: the outermost collapsed ancestor, else itself. */
export function representative(byId: Map<string, GraphNode>, id: string, expanded: Set<string>): string {
  const chain: string[] = [];
  for (let cur = byId.get(id)?.parent; cur; cur = byId.get(cur)?.parent) chain.unshift(cur);
  for (const anc of chain) if (!expanded.has(anc)) return anc;
  return id;
}

/** Edges between visible nodes, deduplicated, self-loops dropped. */
export function liftEdges(g: ModelGraph, expanded: Set<string>): PlacedEdge[] {
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const seen = new Set<string>();
  const out: PlacedEdge[] = [];
  for (const [a, b, label] of g.edges as GraphEdge[]) {
    const ra = representative(byId, a, expanded);
    const rb = representative(byId, b, expanded);
    if (ra === rb || ra === b || rb === a) continue; // collapsed into one box, or an edge into own container
    const key = `${ra}->${rb}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ id: key, from: ra, to: rb, label, dim: byId.get(a)?.dimOut });
  }
  return out;
}

export function layoutGraph(g: ModelGraph, expanded: Set<string>, mode: SizeMode): Layout {
  const byId = new Map(g.nodes.map((n) => [n.id, n]));
  const kids = new Map<string | undefined, GraphNode[]>();
  for (const n of g.nodes) {
    const list = kids.get(n.parent) ?? [];
    list.push(n);
    kids.set(n.parent, list);
  }
  const edges = liftEdges(g, expanded);
  const placed: PlacedNode[] = [];

  // ancestor of `id` that is a direct child of `container` (undefined = top level)
  const childOf = (id: string, container: string | undefined): string | undefined => {
    for (let cur: string | undefined = id; cur; cur = byId.get(cur)?.parent) {
      if (byId.get(cur)?.parent === container) return cur;
    }
    return undefined;
  };

  // Lays out `container`'s children; returns its inner size. Pushes placed nodes.
  function place(container: string | undefined): { w: number; h: number; items: PlacedNode[] } {
    const items = kids.get(container) ?? [];
    const sized: PlacedNode[] = [];
    const inner = new Map<string, PlacedNode[]>(); // nested results, appended after their parent
    for (const n of items) {
      const group = kids.has(n.id);
      const open = group && expanded.has(n.id);
      const bar = barWidth(n.dimOut, mode);
      let w = Math.max(bar, LABEL_W);
      let h = LEAF_H;
      if (open) {
        const sub = place(n.id);
        w = Math.max(sub.w + 2 * GROUP_PAD, LABEL_W);
        h = sub.h + GROUP_HEAD + GROUP_PAD;
        inner.set(n.id, sub.items);
      }
      sized.push({ id: n.id, node: n, parent: container, x: 0, y: 0, w, h, bar, expanded: open });
    }

    // longest-path layering over sibling edges (graph is a DAG)
    const ids = sized.map((s) => s.id);
    const preds = new Map<string, string[]>(ids.map((id) => [id, []]));
    for (const e of edges) {
      const a = childOf(e.from, container);
      const b = childOf(e.to, container);
      if (a && b && a !== b) preds.get(b)!.push(a);
    }
    const layer = new Map<string, number>();
    const depth = (id: string, stack = new Set<string>()): number => {
      if (layer.has(id)) return layer.get(id)!;
      if (stack.has(id)) return 0; // defensive: ignore cycles
      stack.add(id);
      const d = Math.max(-1, ...preds.get(id)!.map((p) => depth(p, stack))) + 1;
      layer.set(id, d);
      return d;
    };
    ids.forEach((id) => depth(id));

    const rows: PlacedNode[][] = [];
    for (const s of sized) (rows[layer.get(s.id)!] ??= []).push(s);
    const rowW = rows.map((r) => r.reduce((t, s) => t + s.w, 0) + COL_GAP * (r.length - 1));
    const width = Math.max(0, ...rowW);
    let y = container ? GROUP_HEAD : 0;
    rows.forEach((r, i) => {
      let x = (width - rowW[i]) / 2 + (container ? GROUP_PAD : 0);
      const rowH = Math.max(...r.map((s) => s.h));
      for (const s of r) {
        s.x = x;
        s.y = y + (rowH - s.h) / 2;
        x += s.w + COL_GAP;
      }
      y += rowH + ROW_GAP;
    });
    const height = y - ROW_GAP - (container ? GROUP_HEAD : 0);

    const ordered: PlacedNode[] = [];
    for (const s of sized) {
      ordered.push(s);
      ordered.push(...(inner.get(s.id) ?? []));
    }
    return { w: width, h: Math.max(height, 0), items: ordered };
  }

  placed.push(...place(undefined).items);
  return { nodes: placed, edges };
}
