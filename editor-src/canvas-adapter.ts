// JSON Canvas (jsoncanvas.org 1.0) ⇄ React Flow, both directions.
//
// Correctness rule (red-team #3): an UNEDITED round-trip must not lose data.
// Every original field is preserved by spreading the whole node/edge into RF
// `data` on import and merging it back on export — so unmodeled fields
// (group.background, backgroundStyle, future spec additions, …) survive verbatim.
import type { Node as RFNode, Edge as RFEdge } from "@xyflow/react";

// ---- JSON Canvas types ----
export type CanvasColor = string; // preset "1".."6" or hex "#rrggbb"
export type Side = "top" | "right" | "bottom" | "left";
export type End = "none" | "arrow";

export interface CanvasNode {
  id: string;
  type: "text" | "file" | "link" | "group";
  x: number;
  y: number;
  width: number;
  height: number;
  color?: CanvasColor;
  // type-specific (only one applies)
  text?: string;
  file?: string;
  url?: string;
  label?: string;
  background?: string;
  backgroundStyle?: string;
  [extra: string]: unknown; // preserve unknown fields
}

export interface CanvasEdge {
  id: string;
  fromNode: string;
  toNode: string;
  fromSide?: Side;
  toSide?: Side;
  fromEnd?: End;
  toEnd?: End;
  color?: CanvasColor;
  label?: string;
  [extra: string]: unknown;
}

export interface Canvas {
  nodes: CanvasNode[];
  edges: CanvasEdge[];
  [extra: string]: unknown;
}

// RF node/edge data carries the full original object under `canvas`, plus the
// editable surface (color, text, label) hoisted for convenient mutation.
export interface CanvasNodeData extends Record<string, unknown> {
  canvas: CanvasNode;
  color?: CanvasColor;
  label: string; // display text (text/label/file/url) — editable
}
export interface CanvasEdgeData extends Record<string, unknown> {
  canvas: CanvasEdge;
  color?: CanvasColor;
}

export type EditorNode = RFNode<CanvasNodeData>;
export type EditorEdge = RFEdge<CanvasEdgeData>;

// Display string shown inside a node.
function nodeLabel(n: CanvasNode): string {
  return n.text ?? n.label ?? n.file ?? n.url ?? "";
}

// ---- import: JSON Canvas → React Flow ----
export function toReactFlow(canvas: Canvas): { nodes: EditorNode[]; edges: EditorEdge[] } {
  const nodes: EditorNode[] = (canvas.nodes ?? []).map((n) => ({
    id: n.id,
    type: n.type,
    position: { x: n.x, y: n.y },
    width: n.width,
    height: n.height,
    // set style too so RF renders at the authored size before any measuring
    style: { width: n.width, height: n.height },
    // groups sit behind everything so they don't cover/eat clicks (red-team #6)
    zIndex: n.type === "group" ? 0 : 1,
    data: { canvas: n, color: n.color, label: nodeLabel(n) },
  }));

  const edges: EditorEdge[] = (canvas.edges ?? []).map((e) => ({
    id: e.id,
    source: e.fromNode,
    target: e.toNode,
    ...(e.fromSide ? { sourceHandle: `s-${e.fromSide}` } : {}),
    ...(e.toSide ? { targetHandle: `t-${e.toSide}` } : {}),
    ...(e.label ? { label: e.label } : {}),
    // toEnd defaults to "arrow" per spec; only suppress the marker for "none"
    markerEnd: e.toEnd === "none" ? undefined : "arrowclosed",
    data: { canvas: e, color: e.color },
  }));

  return { nodes, edges };
}

// strip a handle prefix ("s-top" → "top"); undefined if no handle
function sideFromHandle(handle: string | null | undefined, prefix: "s-" | "t-"): Side | undefined {
  if (!handle || !handle.startsWith(prefix)) return undefined;
  return handle.slice(prefix.length) as Side;
}

// ---- export: React Flow → JSON Canvas ----
export function toCanvas(state: { nodes: EditorNode[]; edges: EditorEdge[] }): Canvas {
  const nodes: CanvasNode[] = state.nodes.map((n) => {
    const orig = n.data.canvas;
    const node: CanvasNode = {
      ...orig, // preserve every original field
      id: n.id,
      x: Math.round(n.position.x),
      y: Math.round(n.position.y),
      // user-set size lives in n.width/height (NodeResizer updates it); never read measured
      width: Math.round((n.width as number | undefined) ?? orig.width),
      height: Math.round((n.height as number | undefined) ?? orig.height),
    };
    // editable surface overrides
    if (n.data.color !== undefined) node.color = n.data.color;
    else delete node.color;
    applyLabel(node, n.data.label);
    return node;
  });

  const edges: CanvasEdge[] = state.edges.map((e) => {
    const orig = e.data?.canvas ?? ({ id: e.id } as CanvasEdge);
    const edge: CanvasEdge = {
      ...orig,
      id: e.id,
      fromNode: e.source,
      toNode: e.target,
    };
    assignOrDelete(edge, "fromSide", sideFromHandle(e.sourceHandle, "s-") ?? orig.fromSide);
    assignOrDelete(edge, "toSide", sideFromHandle(e.targetHandle, "t-") ?? orig.toSide);
    assignOrDelete(edge, "label", typeof e.label === "string" ? e.label : orig.label);
    if (e.data?.color !== undefined) edge.color = e.data.color;
    return edge;
  });

  return { ...canvasExtras(state), nodes, edges };
}

// write the edited display string back onto the right type-specific field
function applyLabel(node: CanvasNode, label: string) {
  switch (node.type) {
    case "text": node.text = label; break;
    case "group": if (label) node.label = label; else delete node.label; break;
    case "link": node.url = label; break;
    case "file": node.file = label; break;
  }
}

function assignOrDelete<T extends object, K extends keyof T>(obj: T, key: K, value: T[K] | undefined) {
  if (value === undefined) delete obj[key];
  else obj[key] = value;
}

// carry any top-level canvas fields beyond nodes/edges (none today, future-proof)
function canvasExtras(state: { nodes: EditorNode[] }): Record<string, unknown> {
  void state;
  return {};
}
