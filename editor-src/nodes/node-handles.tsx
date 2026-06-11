import { Handle, Position } from "@xyflow/react";

// 8 handles per node: a source + a target on each of the 4 sides.
// IDs are `s-<side>` / `t-<side>` so the adapter can map JSON Canvas edge
// fromSide/toSide unambiguously even when one side is both a source and a
// target across different edges (red-team #4). Handles are dimmed until hover
// so they don't clutter the diagram but stay available for drawing edges.
const SIDES: { side: string; pos: Position }[] = [
  { side: "top", pos: Position.Top },
  { side: "right", pos: Position.Right },
  { side: "bottom", pos: Position.Bottom },
  { side: "left", pos: Position.Left },
];

export function NodeHandles() {
  return (
    <>
      {SIDES.map(({ side, pos }) => (
        <span key={side}>
          <Handle id={`t-${side}`} type="target" position={pos} className="cv-handle" />
          <Handle id={`s-${side}`} type="source" position={pos} className="cv-handle" />
        </span>
      ))}
    </>
  );
}
