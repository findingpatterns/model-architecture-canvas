// Client-side IO + id helpers for the editor. NOTHING here touches a server —
// download/import are pure browser operations (no DB persistence, by design).
import { toReactFlow, toCanvas, type Canvas, type EditorNode, type EditorEdge, type CanvasNode } from "./canvas-adapter";

// Collision-free id vs the live id set (red-team #7): bare counters collide with
// imported ids like "n-5". crypto.randomUUID() is effectively unique; we still
// re-roll on the astronomically unlikely clash.
export function makeId(prefix: string, existing: Set<string>): string {
  let id = `${prefix}-${crypto.randomUUID().slice(0, 8)}`;
  while (existing.has(id)) id = `${prefix}-${crypto.randomUUID().slice(0, 8)}`;
  existing.add(id);
  return id;
}

// Download the current editor state as a .canvas file (adapter preserves all fields).
export function downloadCanvas(state: { nodes: EditorNode[]; edges: EditorEdge[] }, modelId: string | null) {
  const canvas = toCanvas(state);
  const blob = new Blob([JSON.stringify(canvas, null, 2) + "\n"], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${modelId ?? "model"}-edited.canvas`;
  a.click();
  URL.revokeObjectURL(url);
}

// Parse a local .canvas file's text into editor state. Throws on invalid JSON.
export function parseCanvasText(text: string): { nodes: EditorNode[]; edges: EditorEdge[] } {
  const canvas = JSON.parse(text) as Canvas;
  if (!Array.isArray(canvas.nodes)) throw new Error("Not a JSON Canvas file (missing nodes array).");
  return toReactFlow(canvas);
}

// A fresh text node centred in the current viewport.
export function newTextNode(id: string, x: number, y: number): EditorNode {
  const canvas: CanvasNode = { id, type: "text", x, y, width: 220, height: 90, text: "New node" };
  return {
    id,
    type: "text",
    position: { x, y },
    width: 220,
    height: 90,
    style: { width: 220, height: 90 },
    zIndex: 1,
    data: { canvas, color: undefined, label: "New node" },
  };
}
