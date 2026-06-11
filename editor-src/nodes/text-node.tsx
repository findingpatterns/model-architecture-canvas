import { useState, useRef, useEffect } from "react";
import { NodeResizer, useReactFlow, type NodeProps } from "@xyflow/react";
import { NodeHandles } from "./node-handles";
import { resolveColor } from "../canvas-colors";
import { useEditorContext } from "../editor-context";
import type { CanvasNodeData } from "../canvas-adapter";

// Text node: colored left accent, raw (multiline) text. Double-click to edit
// inline via a textarea (red-team #12 — multiline content like the demo title).
// NodeResizer (visible when selected) drives width/height; RF persists them.
export function TextNode({ id, data, selected }: NodeProps) {
  const d = data as CanvasNodeData;
  const { updateNodeData } = useReactFlow();
  const { markDirty } = useEditorContext();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(d.label);
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (editing) ref.current?.focus();
  }, [editing]);

  const accent = resolveColor(d.color) ?? "var(--cv-accent)";

  function commit() {
    setEditing(false);
    if (draft !== d.label) { updateNodeData(id, { label: draft }); markDirty(); }
  }

  return (
    <div className="cv-node cv-text-node" style={{ borderLeftColor: accent }}>
      <NodeResizer isVisible={selected} minWidth={80} minHeight={40} />
      <NodeHandles />
      {editing ? (
        <textarea
          ref={ref}
          className="cv-node-edit"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) commit();
            if (e.key === "Escape") { setDraft(d.label); setEditing(false); }
          }}
        />
      ) : (
        <div className="cv-node-text" onDoubleClick={() => { setDraft(d.label); setEditing(true); }}>
          {d.label}
        </div>
      )}
    </div>
  );
}
