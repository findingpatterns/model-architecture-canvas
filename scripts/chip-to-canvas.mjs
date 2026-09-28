// chip.json → JSON Canvas 1.0 (2D tree diagram) so every hardware entry also
// works in the existing viewer, the React Flow editor and Obsidian.
//
// Deterministic layout (stable git diffs): each scene is a group node; inside it
// components are laid out as a tree — depth → column, depth-first order → row.
// Parent→child edges are grey; `flows` become colored "data" edges.

const COL_W = 320;
const ROW_H = 150;
const NODE_W = 260;
const PAD = 40;

function nodeText(c) {
  const lines = [`**${c.name}**`];
  if (c.repeat) {
    const n = c.repeat.count.reduce((a, b) => a * b, 1);
    if (n > 1) lines.push(`× ${n} (symbolic)`);
  }
  for (const s of (c.specs ?? []).slice(0, 2)) lines.push(`${s.label}: ${s.value}${s.estimate ? " (est.)" : ""}`);
  return lines.join("\n");
}

// Lay out one scene's components; returns { nodes, edges, width, height }.
function layoutScene(components, groups, ox, oy) {
  const byParent = new Map();
  const ids = new Set(components.map((c) => c.id));
  for (const c of components) {
    const key = c.parent && ids.has(c.parent) ? c.parent : null;
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(c);
  }
  const nodes = [];
  const edges = [];
  let row = 0;
  let maxDepth = 0;
  const visit = (c, depth) => {
    maxDepth = Math.max(maxDepth, depth);
    nodes.push({
      id: c.id,
      type: "text",
      text: nodeText(c),
      x: ox + PAD + depth * COL_W,
      y: oy + PAD + row * ROW_H,
      width: NODE_W,
      height: 110,
      color: c.color ?? groups[c.group],
    });
    row++;
    for (const child of byParent.get(c.id) ?? []) {
      edges.push({ id: `tree-${c.id}-${child.id}`, fromNode: c.id, fromSide: "right", toNode: child.id, toSide: "left" });
      visit(child, depth + 1);
    }
  };
  for (const root of byParent.get(null) ?? []) visit(root, 0);
  return { nodes, edges, width: PAD * 2 + maxDepth * COL_W + NODE_W, height: PAD * 2 + row * ROW_H };
}

export function chipToCanvas(chip) {
  const nodes = [];
  const edges = [];
  nodes.push({
    id: "title",
    type: "text",
    text: `# ${chip.name}\n${chip.arch} · ${chip.vendor}\n${chip.disclaimer ?? ""}`.trim(),
    x: 0,
    y: -200,
    width: 520,
    height: 140,
  });

  let ox = 0;
  const scenes = [["Main package", chip.components, null], ...Object.entries(chip.scenes ?? {}).map(([id, s]) => [`Inside: ${id}`, s.components, s.anchor])];
  for (const [label, comps, anchor] of scenes) {
    const lay = layoutScene(comps, chip.groups, ox, 0);
    const gid = `group-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    nodes.push({ id: gid, type: "group", label, x: ox, y: 0, width: lay.width, height: lay.height });
    nodes.push(...lay.nodes);
    edges.push(...lay.edges);
    if (anchor) {
      const first = lay.nodes[0];
      if (first) edges.push({ id: `drill-${gid}`, fromNode: anchor, toNode: first.id, toSide: "left", label: "look inside" });
    }
    ox += lay.width + 120;
  }

  for (const f of chip.flows ?? [])
    edges.push({ id: `flow-${f.id}`, fromNode: f.from, toNode: f.to, color: chip.groups[f.group], label: "data" });

  return { nodes, edges };
}
