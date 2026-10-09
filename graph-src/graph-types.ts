// Schema of models/<id>/graph.json — the single source of truth for the graph viewer.
// Sizes are DATA (channels, params), not pixels; the viewer decides how to draw them.

export type NodeKind =
  | "io" | "embed" | "op" | "hc" | "norm" | "group" | "attn" | "linear" | "moe" | "out";

export interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  parent?: string;        // containing group; omitted = top level
  dimIn?: number;         // channels entering this op
  dimOut?: number;        // channels leaving this op (drives width in size mode)
  repeat?: number;        // ×N (layers, experts, …)
  params?: number;        // stored params (from checkpoint shapes)
  activeParams?: number;  // params a token actually touches, when it differs
  dtype?: string;         // checkpoint dtype of the main tensor
  tensors?: string[];     // regexes over checkpoint tensor names (traceability)
  note?: string;
}

/** [from, to, label?] */
export type GraphEdge = [string, string, string?];

export interface LayerRow {
  id: string;
  mode: string;
  experts: number;
  shared: number;
  attention: number;
  engram: number;
  other: number;
  active: number;
}

export interface ModelGraph {
  model: string;
  name: string;
  source: string;
  totalParams: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
  layers?: LayerRow[];
}

export type SizeMode = "off" | "sqrt" | "log" | "linear";
