// Pure, immutable chip mutations used by edit mode (and its undo history).
// Every function returns a NEW chip and keeps all cross-references valid:
// deleting a component cascades to its descendants, flows, steps and scenes.
import type { Chip, Component, SceneKey, Step, Vec3 } from "./chip-types.ts";
import { round3 } from "./chip-geometry.ts";

export function locate(chip: Chip, id: string): { scene: SceneKey; comp: Component } | null {
  const c = chip.components.find((x) => x.id === id);
  if (c) return { scene: null, comp: c };
  for (const [sid, s] of Object.entries(chip.scenes ?? {})) {
    const f = s.components.find((x) => x.id === id);
    if (f) return { scene: sid, comp: f };
  }
  return null;
}

function allIds(chip: Chip): Set<string> {
  const ids = new Set(chip.components.map((c) => c.id));
  for (const s of Object.values(chip.scenes ?? {})) for (const c of s.components) ids.add(c.id);
  return ids;
}

export function uniqueId(chip: Chip, base: string): string {
  const ids = allIds(chip);
  if (!ids.has(base)) return base;
  let n = 2;
  while (ids.has(`${base}-${n}`)) n++;
  return `${base}-${n}`;
}

// Apply fn to the component list of one scene.
function mapScene(chip: Chip, scene: SceneKey, fn: (list: Component[]) => Component[]): Chip {
  if (scene === null) return { ...chip, components: fn(chip.components) };
  const s = chip.scenes![scene];
  return { ...chip, scenes: { ...chip.scenes, [scene]: { ...s, components: fn(s.components) } } };
}

export function updateComponent(chip: Chip, id: string, patch: Partial<Component>): Chip {
  const loc = locate(chip, id);
  if (!loc) return chip;
  return mapScene(chip, loc.scene, (list) => list.map((c) => (c.id === id ? { ...c, ...patch } : c)));
}

export function moveComponent(chip: Chip, id: string, pos: Vec3): Chip {
  const loc = locate(chip, id);
  if (!loc) return chip;
  return updateComponent(chip, id, { geom: { ...loc.comp.geom, pos: pos.map(round3) as Vec3 } });
}

export function resizeComponent(chip: Chip, id: string, size: Vec3): Chip {
  const loc = locate(chip, id);
  if (!loc) return chip;
  return updateComponent(chip, id, { geom: { ...loc.comp.geom, size: size.map((n) => Math.max(0.01, round3(n))) as Vec3 } });
}

// New box as a child of `parentId` (or a root when null), placed just above it.
export function addComponent(chip: Chip, parentId: string | null, scene: SceneKey = null): { chip: Chip; id: string } {
  const id = uniqueId(chip, "part");
  const parent = parentId ? locate(chip, parentId) : null;
  const sc = parent ? parent.scene : scene;
  const p = parent?.comp.geom;
  const comp: Component = {
    id,
    ...(parentId ? { parent: parentId } : {}),
    name: "New part",
    group: Object.keys(chip.groups)[0],
    geom: { type: "box", size: [1, 0.2, 1], pos: p ? [p.pos[0], round3(p.pos[1] + p.size[1] / 2 + 0.1), p.pos[2]] : [0, 1, 0] },
    explode: sc === null ? 1 : undefined,
    desc: "",
  };
  if (comp.explode === undefined) delete comp.explode;
  return { chip: mapScene(chip, sc, (list) => [...list, comp]), id };
}

export function duplicateComponent(chip: Chip, id: string): { chip: Chip; id: string } {
  const loc = locate(chip, id);
  if (!loc) return { chip, id };
  const newId = uniqueId(chip, `${id}-copy`);
  const g = loc.comp.geom;
  const copy: Component = { ...structuredClone(loc.comp), id: newId, geom: { ...g, pos: [round3(g.pos[0] + 0.5), g.pos[1], g.pos[2]] } };
  delete copy.drill;
  return { chip: mapScene(chip, loc.scene, (list) => [...list, copy]), id: newId };
}

// Delete a component and its descendants, then repair every reference to them.
export function deleteComponent(chip: Chip, id: string): Chip {
  const loc = locate(chip, id);
  if (!loc) return chip;
  const list = loc.scene === null ? chip.components : chip.scenes![loc.scene].components;
  const gone = new Set([id]);
  for (let grew = true; grew; ) {
    grew = false;
    for (const c of list) if (c.parent && gone.has(c.parent) && !gone.has(c.id)) { gone.add(c.id); grew = true; }
  }
  let next = mapScene(chip, loc.scene, (l) => l.filter((c) => !gone.has(c.id)));

  // Scenes anchored on a deleted component go too (with their ids).
  const deadScenes = new Set(Object.entries(next.scenes ?? {}).filter(([, s]) => gone.has(s.anchor)).map(([k]) => k));
  for (const k of deadScenes) for (const c of next.scenes![k].components) gone.add(c.id);
  if (next.scenes) {
    const scenes = Object.fromEntries(Object.entries(next.scenes).filter(([k]) => !deadScenes.has(k)));
    next = { ...next, scenes };
  }
  const fixDrill = (c: Component) => (c.drill && deadScenes.has(c.drill) ? (({ drill: _d, ...rest }) => rest)(c) : c);
  next = { ...next, components: next.components.map(fixDrill) };
  if (next.scenes)
    next = {
      ...next,
      scenes: Object.fromEntries(Object.entries(next.scenes).map(([k, s]) => [k, { ...s, components: s.components.map(fixDrill) }])),
    };

  if (next.flows) next = { ...next, flows: next.flows.filter((f) => !gone.has(f.from) && !gone.has(f.to)) };
  if (next.steps)
    next = {
      ...next,
      steps: next.steps.map((s) => {
        const out: Step = { ...s };
        if (s.focus) out.focus = s.focus.filter((f) => !gone.has(f));
        if (s.dot && gone.has(s.dot)) delete out.dot;
        if (s.drill && deadScenes.has(s.drill)) out.drill = null;
        return out;
      }),
    };
  return next;
}

export function setSteps(chip: Chip, steps: Step[]): Chip {
  return { ...chip, steps };
}
