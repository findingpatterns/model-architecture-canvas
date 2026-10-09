// Graph viewer: loads graph-data/<id>.json (built from models/<id>/graph.json) and renders it
// with React Flow. Layout is recomputed from data whenever groups open/close or the size mode
// changes — sizes are never baked into pixels.
import { useEffect, useMemo, useState } from "react";
import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from "@xyflow/react";
import type { ModelGraph, SizeMode } from "./graph-types.ts";
import { edgeWidth, layoutGraph } from "./graph-layout.ts";
import { GroupNode, OpNode, type NodeData } from "./graph-node.tsx";
import { InfoPanel, LayersPanel } from "./side-panels.tsx";
import { fmtParams, KIND_COLOR } from "./graph-format.ts";

const nodeTypes = { op: OpNode, box: GroupNode };
const MODES: [SizeMode, string][] = [["off", "uniform"], ["sqrt", "√ channels"], ["log", "log channels"], ["linear", "linear"]];
const modelId = new URLSearchParams(location.search).get("model") ?? "";

export function App() {
  const [graph, setGraph] = useState<ModelGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set(["stack"]));
  const [mode, setMode] = useState<SizeMode>("sqrt");
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<"info" | "layers">("info");

  useEffect(() => {
    if (!/^[a-z0-9-]+$/.test(modelId)) { setError("Missing or invalid ?model=<id>"); return; }
    fetch(`../graph-data/${modelId}.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((g: ModelGraph) => { setGraph(g); document.title = `${g.name} — graph · ModelCanvas`; })
      .catch((e) => setError(`Could not load graph for "${modelId}": ${e.message}`));
  }, []);

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });

  const groupIds = useMemo(() => new Set(graph?.nodes.filter((n) => n.parent).map((n) => n.parent!)), [graph]);

  const { nodes, edges } = useMemo(() => {
    if (!graph) return { nodes: [] as Node[], edges: [] as Edge[] };
    const l = layoutGraph(graph, expanded, mode);
    const nodes: Node[] = l.nodes.map((p) => ({
      id: p.id,
      type: groupIds.has(p.id) ? "box" : "op",
      position: { x: p.x, y: p.y },
      parentId: p.parent,
      draggable: false,
      style: { width: p.w, height: p.h },
      data: { placed: p, selected: p.id === selected, onToggle: toggle, onSelect: setSelected } satisfies NodeData,
    }));
    const edges: Edge[] = l.edges.map((e) => ({
      id: e.id,
      source: e.from,
      target: e.to,
      label: e.label,
      type: "smoothstep",
      zIndex: 1,
      style: { strokeWidth: edgeWidth(e.dim, mode), stroke: "var(--edge)" },
    }));
    return { nodes, edges };
  }, [graph, expanded, mode, selected, groupIds]);

  if (error) return <p className="fatal">{error}</p>;
  if (!graph) return <p className="fatal">Loading…</p>;
  const sel = graph.nodes.find((n) => n.id === selected);

  return (
    <div className="shell">
      <header className="bar">
        <a className="back mono" href={`../?model=${modelId}`}>← canvas view</a>
        <h1>{graph.name}</h1>
        <span className="sub mono">{fmtParams(graph.totalParams)} params · {graph.source}</span>
        <div className="seg" role="group" aria-label="Size mode">
          {MODES.map(([m, label]) => (
            <button key={m} type="button" className={m === mode ? "on" : ""} onClick={() => setMode(m)}>{label}</button>
          ))}
        </div>
        <button type="button" className="ghost" onClick={() => setExpanded(new Set(groupIds))}>expand all</button>
        <button type="button" className="ghost" onClick={() => setExpanded(new Set())}>collapse all</button>
      </header>
      <main className="body">
        <div className="flow">
          <ReactFlow
            key={`${mode}-${[...expanded].sort().join(",")}`}
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            fitView
            minZoom={0.05}
            nodesConnectable={false}
            onPaneClick={() => setSelected(null)}
            proOptions={{ hideAttribution: true }}
          >
            <Background gap={24} color="var(--grid)" />
            <Controls showInteractive={false} />
            <MiniMap pannable zoomable bgColor="#121514" maskColor="#0b0d0cb0" nodeColor={(n) => KIND_COLOR[(n.data as NodeData).placed.node.kind]} />
          </ReactFlow>
          {mode !== "off" && (
            <div className="scale-note mono">
              bar width ∝ {mode === "sqrt" ? "√channels" : mode === "log" ? "log₂ channels" : "channels (clipped at 1100px)"} · edge thickness ∝ √channels
            </div>
          )}
        </div>
        <aside className="side">
          <div className="tabs">
            <button type="button" className={tab === "info" ? "on" : ""} onClick={() => setTab("info")}>Details</button>
            {graph.layers && <button type="button" className={tab === "layers" ? "on" : ""} onClick={() => setTab("layers")}>Layers</button>}
          </div>
          {tab === "info" ? <InfoPanel node={sel} /> : graph.layers && <LayersPanel layers={graph.layers} />}
        </aside>
      </main>
    </div>
  );
}
