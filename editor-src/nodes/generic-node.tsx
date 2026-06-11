import { useState, useRef, useEffect } from "react";
import { NodeResizer, useReactFlow, type NodeProps } from "@xyflow/react";
import { NodeHandles } from "./node-handles";
import { resolveColor } from "../canvas-colors";
import { useEditorContext } from "../editor-context";
import type { CanvasNodeData } from "../canvas-adapter";

// Generic renderer for file / link / group nodes (red-team #7 — no bespoke
// per-type code; full data still round-trips via the adapter).
//
// GROUP (red-team #6): rendered as a translucent labelled box BEHIND everything
// (zIndex 0 set on import) with `pointer-events:none` on the body so enclosed
// nodes stay visible and clickable. Only the label chip is interactive (for the
// explicit edit affordance — double-click conflicts with selection otherwise).
export function GenericNode({ id, data, selected }: NodeProps) {
  const d = data as CanvasNodeData;
  const type = d.canvas.type;
  const accent = resolveColor(d.color) ?? "var(--cv-accent)";

  const { updateNodeData } = useReactFlow();
  const { markDirty } = useEditorContext();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(d.label);
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => { if (editing) ref.current?.focus(); }, [editing]);
  function commit() {
    setEditing(false);
    if (draft !== d.label) { updateNodeData(id, { label: draft }); markDirty(); }
  }

  if (type === "group") {
    return (
      <div className="cv-group" style={{ borderColor: accent }}>
        <NodeResizer isVisible={selected} minWidth={120} minHeight={80} />
        <NodeHandles />
        <div className="cv-group-label" style={{ background: accent }}>
          {editing ? (
            <input
              ref={ref}
              className="cv-group-label-edit"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onBlur={commit}
              onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") { setDraft(d.label); setEditing(false); } }}
            />
          ) : (
            <span onDoubleClick={() => { setDraft(d.label); setEditing(true); }}>{d.label || "group"}</span>
          )}
        </div>
      </div>
    );
  }

  // file / link card
  return (
    <div className="cv-node cv-file-node" style={{ borderLeftColor: accent }}>
      <NodeResizer isVisible={selected} minWidth={80} minHeight={40} />
      <NodeHandles />
      <span className="cv-node-kind mono">{type}</span>
      <div className="cv-node-text">{d.label}</div>
    </div>
  );
}
