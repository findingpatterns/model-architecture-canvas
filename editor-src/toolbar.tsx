import { useRef } from "react";
import { useReactFlow } from "@xyflow/react";
import { ColorPicker } from "./color-picker";
import { useEditorContext } from "./editor-context";
import { downloadCanvas, parseCanvasText, newTextNode, makeId } from "./editor-io";
import type { EditorNode, EditorEdge } from "./canvas-adapter";

// Floating toolbar: add node, recolor selection, delete selection, download, open.
export function Toolbar({ modelId }: { modelId: string | null }) {
  const rf = useReactFlow<EditorNode, EditorEdge>();
  const { markDirty } = useEditorContext();
  const fileInput = useRef<HTMLInputElement>(null);

  function idSet(): Set<string> {
    return new Set([...rf.getNodes().map((n) => n.id), ...rf.getEdges().map((e) => e.id)]);
  }

  function addNode() {
    const id = makeId("n", idSet());
    const c = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    const p = rf.screenToFlowPosition(c);
    rf.addNodes(newTextNode(id, Math.round(p.x), Math.round(p.y)));
    markDirty();
  }

  function recolor(token: string | undefined) {
    for (const n of rf.getNodes()) if (n.selected) rf.updateNodeData(n.id, { color: token });
    markDirty();
  }

  function deleteSelection() {
    const nodes = rf.getNodes().filter((n) => n.selected);
    const edges = rf.getEdges().filter((e) => e.selected);
    if (nodes.length || edges.length) { rf.deleteElements({ nodes, edges }); markDirty(); }
  }

  function openFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    file.text().then((text) => {
      try {
        const { nodes, edges } = parseCanvasText(text);
        rf.setNodes(nodes);
        rf.setEdges(edges);
        rf.fitView();
        markDirty();
      } catch (err) {
        alert(`Could not open file: ${(err as Error).message}`);
      }
    });
    e.target.value = "";
  }

  return (
    <div className="cv-toolbar">
      <button className="cv-btn" onClick={addNode}>＋ Node</button>
      <ColorPicker onPick={recolor} />
      <button className="cv-btn" onClick={deleteSelection}>🗑 Delete</button>
      <span className="cv-toolbar-sep" />
      <button className="cv-btn" onClick={() => fileInput.current?.click()}>📂 Open</button>
      <button className="cv-btn cv-btn-primary" onClick={() => downloadCanvas({ nodes: rf.getNodes(), edges: rf.getEdges() }, modelId)}>⬇ Download .canvas</button>
      <input ref={fileInput} type="file" accept=".canvas,application/json" style={{ display: "none" }} onChange={openFile} />
    </div>
  );
}
