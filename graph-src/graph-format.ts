// Small display helpers shared by the graph viewer components.
import type { NodeKind } from "./graph-types.ts";

export function fmtParams(n: number | undefined): string {
  if (n === undefined) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 1 : 2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function fmtDim(n: number | undefined): string {
  return n === undefined ? "—" : n.toLocaleString("en-US");
}

// One color per semantic kind, consistent across graph, info panel and layer chart.
export const KIND_COLOR: Record<NodeKind, string> = {
  io: "#f59e0b",
  embed: "#a78bfa",
  op: "#94a3b8",
  hc: "#f87171",
  norm: "#facc15",
  group: "#64748b",
  attn: "#38bdf8",
  linear: "#60a5fa",
  moe: "#4ade80",
  out: "#fb923c",
};
