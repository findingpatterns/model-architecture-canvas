import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type NodeTypes,
  type Connection,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import { toReactFlow, type Canvas, type EditorNode, type EditorEdge } from "./canvas-adapter";
import { TextNode } from "./nodes/text-node";
import { GenericNode } from "./nodes/generic-node";
import { Toolbar } from "./toolbar";
import { EditorContext } from "./editor-context";
import { makeId } from "./editor-io";

const nodeTypes: NodeTypes = { text: TextNode, file: GenericNode, link: GenericNode, group: GenericNode };

function initTheme() {
  const saved = localStorage.getItem("modelcanvas-theme");
  document.documentElement.setAttribute("data-theme", saved === "light" ? "light" : "dark");
}

// change types that represent a real user edit (not initial load / selection)
const DIRTYING = new Set(["position", "dimensions", "remove", "add", "replace"]);

function Editor() {
  const [nodes, setNodes, onNodesChange] = useNodesState<EditorNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<EditorEdge>([]);
  const [error, setError] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string | null>(null);
  const dirty = useRef(false);
  const loaded = useRef(false);
  const markDirty = useCallback(() => { if (loaded.current) dirty.current = true; }, []);

  useEffect(() => {
    initTheme();
    const id = new URLSearchParams(location.search).get("model");
    if (!id) { setError("No ?model specified."); return; }
    setModelId(id);
    fetch(`/canvases/${encodeURIComponent(id)}.canvas`, { cache: "no-cache" })
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() as Promise<Canvas>; })
      .then((canvas) => {
        const conv = toReactFlow(canvas);
        setNodes(conv.nodes);
        setEdges(conv.edges);
        setTimeout(() => { loaded.current = true; }, 0); // ignore load-time changes
      })
      .catch((e) => setError(`Could not load model "${id}": ${e.message}`));
  }, [setNodes, setEdges]);

  // Warn before leaving with unsaved edits (red-team #9).
  useEffect(() => {
    const onBeforeUnload = (e: BeforeUnloadEvent) => { if (dirty.current) { e.preventDefault(); e.returnValue = ""; } };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, []);

  const handleNodesChange = useCallback((changes: NodeChange<EditorNode>[]) => {
    if (changes.some((c) => DIRTYING.has(c.type))) markDirty();
    onNodesChange(changes);
  }, [onNodesChange, markDirty]);

  const handleEdgesChange = useCallback((changes: EdgeChange<EditorEdge>[]) => {
    if (changes.some((c) => DIRTYING.has(c.type))) markDirty();
    onEdgesChange(changes);
  }, [onEdgesChange, markDirty]);

  const onConnect = useCallback((conn: Connection) => {
    setEdges((eds) => {
      const id = makeId("e", new Set(eds.map((e) => e.id)));
      const edge: EditorEdge = {
        id, source: conn.source, target: conn.target,
        sourceHandle: conn.sourceHandle ?? undefined, targetHandle: conn.targetHandle ?? undefined,
        markerEnd: "arrowclosed",
        data: { canvas: { id, fromNode: conn.source, toNode: conn.target }, color: undefined },
      };
      return addEdge(edge, eds);
    });
    markDirty();
  }, [setEdges, markDirty]);

  // Guarded "← Preview" navigation.
  const onBack = useCallback((e: React.MouseEvent) => {
    if (dirty.current && !confirm("Discard unsaved edits and return to preview?")) e.preventDefault();
  }, []);

  if (error) return <div className="cv-message">{error}</div>;

  return (
    <EditorContext.Provider value={{ markDirty }}>
      <div className="cv-app">
        <div className="cv-topbar">
          <a className="cv-back mono" href={modelId ? `../?model=${encodeURIComponent(modelId)}` : "../"} onClick={onBack}>← Preview</a>
          <span className="cv-title mono">{modelId}</span>
          <Toolbar modelId={modelId} />
        </div>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          deleteKeyCode={["Delete", "Backspace"]}
          fitView
          minZoom={0.05}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
          <MiniMap />
          <Controls />
        </ReactFlow>
      </div>
    </EditorContext.Provider>
  );
}

export function App() {
  return useMemo(() => (
    <ReactFlowProvider>
      <Editor />
    </ReactFlowProvider>
  ), []);
}
