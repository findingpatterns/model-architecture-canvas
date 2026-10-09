// React Flow node renderers: a leaf op (bar sized by output channels + caption) and a
// group box (collapsed = clickable card, expanded = container with a header).
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { PlacedNode } from "./graph-layout.ts";
import { fmtDim, fmtParams, KIND_COLOR } from "./graph-format.ts";

export interface NodeData extends Record<string, unknown> {
  placed: PlacedNode;
  selected: boolean;
  onToggle: (id: string) => void;
  onSelect: (id: string) => void;
}

const repeatTag = (r?: number) => (r ? ` ×${r}` : "");

export function OpNode({ data }: NodeProps) {
  const { placed, selected, onSelect } = data as NodeData;
  const n = placed.node;
  const color = KIND_COLOR[n.kind];
  return (
    <div className={`op-node${selected ? " is-selected" : ""}`} onClick={() => onSelect(n.id)}>
      <Handle type="target" position={Position.Top} />
      <div className="op-bar" style={{ width: placed.bar, background: color }} title={`${fmtDim(n.dimOut)} channels`} />
      <div className="op-label">
        <span className="op-name">{n.label}{repeatTag(n.repeat)}</span>
        {n.dimOut !== undefined && <span className="op-dim">{fmtDim(n.dimOut)}</span>}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export function GroupNode({ data }: NodeProps) {
  const { placed, selected, onToggle, onSelect } = data as NodeData;
  const n = placed.node;
  const head = (
    <button
      type="button"
      className="group-head"
      onClick={(e) => { e.stopPropagation(); onToggle(n.id); onSelect(n.id); }}
      title={placed.expanded ? "Collapse" : "Expand"}
    >
      <span className="caret">{placed.expanded ? "▾" : "▸"}</span>
      <span className="op-name">{n.label}{repeatTag(n.repeat)}</span>
      <span className="op-dim">{fmtParams(n.params)}</span>
    </button>
  );
  if (placed.expanded) {
    return (
      <div className={`group-box${selected ? " is-selected" : ""}`}>
        <Handle type="target" position={Position.Top} />
        {head}
        <Handle type="source" position={Position.Bottom} />
      </div>
    );
  }
  return (
    <div className={`op-node group-collapsed${selected ? " is-selected" : ""}`}>
      <Handle type="target" position={Position.Top} />
      <div className="op-bar" style={{ width: placed.bar, background: KIND_COLOR.group }} />
      {head}
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}
