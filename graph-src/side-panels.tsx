// Right-hand panels: details of the selected node, and the per-layer parameter chart.
import type { GraphNode, LayerRow } from "./graph-types.ts";
import { fmtDim, fmtParams, KIND_COLOR } from "./graph-format.ts";

export function InfoPanel({ node }: { node: GraphNode | undefined }) {
  if (!node) return <p className="hint">Click a block to see its shapes, params and checkpoint tensors. Click a ▸ group to open it.</p>;
  const rows: [string, string][] = [
    ["kind", node.kind],
    ["channels in → out", `${fmtDim(node.dimIn)} → ${fmtDim(node.dimOut)}`],
  ];
  if (node.repeat) rows.push(["repeat", `×${node.repeat}`]);
  if (node.params !== undefined) rows.push(["stored params", fmtParams(node.params)]);
  if (node.activeParams !== undefined) rows.push(["active / token", fmtParams(node.activeParams)]);
  if (node.dtype) rows.push(["dtype", node.dtype]);
  return (
    <div className="info">
      <h2><span className="swatch" style={{ background: KIND_COLOR[node.kind] }} />{node.label}</h2>
      <dl>{rows.map(([k, v]) => (<div key={k}><dt>{k}</dt><dd>{v}</dd></div>))}</dl>
      {node.note && <p className="note">{node.note}</p>}
      {node.tensors && (
        <>
          <h3>checkpoint tensors</h3>
          <ul className="tensors">{node.tensors.map((t) => <li key={t}><code>{t}</code></li>)}</ul>
        </>
      )}
    </div>
  );
}

const PARTS: [keyof LayerRow, string, string][] = [
  ["experts", "routed experts", KIND_COLOR.moe],
  ["shared", "shared + gate", "#16a34a"],
  ["attention", "attention", KIND_COLOR.attn],
  ["engram", "Engram", KIND_COLOR.embed],
];

export function LayersPanel({ layers }: { layers: LayerRow[] }) {
  const total = (r: LayerRow) => r.experts + r.shared + r.attention + r.engram + r.other;
  const maxTotal = Math.max(...layers.map(total));
  const maxActive = Math.max(...layers.map((r) => r.active));
  return (
    <div className="layers">
      <div className="legend">{PARTS.map(([, lab, c]) => <span key={lab}><i style={{ background: c }} />{lab}</span>)}</div>
      <div className="layer-head"><span /><span>stored (linear)</span><span>active / token</span></div>
      {layers.map((r) => (
        <div className="layer-row" key={r.id} title={`${r.id} · ${r.mode} · ${fmtParams(total(r))} stored · ${fmtParams(r.active)} active`}>
          <span className="layer-id">{r.id}<em>{r.mode}</em></span>
          <span className="bars">
            {PARTS.map(([k, , c]) => {
              const v = r[k] as number;
              return v ? <i key={k} style={{ width: `${(v / maxTotal) * 100}%`, background: c }} /> : null;
            })}
            <b>{fmtParams(total(r))}</b>
          </span>
          <span className="bars">
            <i style={{ width: `${((r.active - r.attention) / maxActive) * 100}%`, background: KIND_COLOR.moe }} />
            <i style={{ width: `${(r.attention / maxActive) * 100}%`, background: KIND_COLOR.attn }} />
            <b>{fmtParams(r.active)}</b>
          </span>
        </div>
      ))}
    </div>
  );
}
