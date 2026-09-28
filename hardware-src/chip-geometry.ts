// Pure geometry helpers: expand `repeat` into instance offsets, apply explode
// lift, and place drill-down scenes above their anchor. No three.js here so it
// stays unit-testable in Node.
import type { Chip, Component, SceneKey, Vec3 } from "./chip-types.ts";

// Local offsets of every instance relative to geom.pos, plus that instance's y-layer index.
export function instanceOffsets(c: Component): { offset: Vec3; layer: number }[] {
  const r = c.repeat;
  if (!r) return [{ offset: [0, 0, 0], layer: 0 }];
  const out: { offset: Vec3; layer: number }[] = [];
  for (let i = 0; i < r.count[0]; i++)
    for (let j = 0; j < r.count[1]; j++)
      for (let k = 0; k < r.count[2]; k++)
        out.push({ offset: [i * r.step[0], j * r.step[1], k * r.step[2]], layer: j });
  return out;
}

// Vertical lift of a main-scene component (and one of its layers) at explode ∈ [0,1].
export function liftOf(c: Component, explode: number, layer = 0): number {
  return explode * ((c.explode ?? 0) + layer * (c.repeat?.spread ?? 0));
}

// World-space origin of a drill-down scene: anchor's first instance, lifted, plus offset.
export function sceneOrigin(chip: Chip, sceneId: string, explode: number): Vec3 {
  const s = chip.scenes?.[sceneId];
  const a = s && chip.components.find((c) => c.id === s.anchor);
  if (!s || !a) return [0, 0, 0];
  const o = s.offset ?? [0, 3, 0];
  return [a.geom.pos[0] + o[0], a.geom.pos[1] + liftOf(a, explode) + o[1], a.geom.pos[2] + o[2]];
}

// Base position (world) of a component's group node: pos + lift (main) or scene origin + pos.
export function groupPosition(chip: Chip, c: Component, scene: SceneKey, explode: number): Vec3 {
  if (scene === null) return [c.geom.pos[0], c.geom.pos[1] + liftOf(c, explode), c.geom.pos[2]];
  const o = sceneOrigin(chip, scene, explode);
  return [o[0] + c.geom.pos[0], o[1] + c.geom.pos[1], o[2] + c.geom.pos[2]];
}

// Center of a component's first instance in world space (targets for dots / flows / camera).
export function worldCenter(chip: Chip, id: string, explode: number): Vec3 | null {
  const main = chip.components.find((c) => c.id === id);
  if (main) return groupPosition(chip, main, null, explode);
  for (const [sid, s] of Object.entries(chip.scenes ?? {})) {
    const c = s.components.find((x) => x.id === id);
    if (c) return groupPosition(chip, c, sid, explode);
  }
  return null;
}

// Largest horizontal extent (x or z) of the main scene, in scene units.
export function footprint(chip: Chip): number {
  let m = 0;
  for (const c of chip.components) {
    const r = c.repeat;
    for (const axis of [0, 2] as const) {
      const lo = c.geom.pos[axis] - c.geom.size[axis] / 2 + Math.min(0, r ? r.step[axis] * (r.count[axis] - 1) : 0);
      const hi = c.geom.pos[axis] + c.geom.size[axis] / 2 + Math.max(0, r ? r.step[axis] * (r.count[axis] - 1) : 0);
      m = Math.max(m, Math.abs(lo), Math.abs(hi));
    }
  }
  return 2 * m;
}

// World scale per chip. "fit": each chip fills the same frame (structure view).
// "real": one shared mm-per-pixel so sizes are comparable; requires every chip
// to declare scale.mmPerUnit (otherwise falls back to fit).
export const FRAME = 15;
export function worldScales(chips: Chip[], mode: "fit" | "real"): number[] {
  const fit = chips.map((c) => FRAME / (footprint(c) || 1));
  if (mode === "fit" || chips.some((c) => !c.scale)) return fit;
  const biggestMm = Math.max(...chips.map((c) => footprint(c) * c.scale!.mmPerUnit));
  return chips.map((c) => (FRAME * c.scale!.mmPerUnit) / biggestMm);
}

// Physical size (mm) of the die + HBM area — the parts drawn to scale. Uses
// components carrying these roles; null when the chip has no scale info.
const SCALED_ROLES = new Set(["compute-die", "second-compute-die", "die-to-die-link", "hbm-stack"]);
export function scaledFootprintMm(chip: Chip): { w: number; d: number } | null {
  if (!chip.scale) return null;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (const c of chip.components) {
    if (!c.role || !SCALED_ROLES.has(c.role)) continue;
    for (const { offset } of instanceOffsets(c)) {
      const x = c.geom.pos[0] + offset[0], z = c.geom.pos[2] + offset[2];
      x0 = Math.min(x0, x - c.geom.size[0] / 2); x1 = Math.max(x1, x + c.geom.size[0] / 2);
      z0 = Math.min(z0, z - c.geom.size[2] / 2); z1 = Math.max(z1, z + c.geom.size[2] / 2);
    }
  }
  if (!Number.isFinite(x0)) return null;
  const k = chip.scale.mmPerUnit;
  return { w: (x1 - x0) * k, d: (z1 - z0) * k };
}

export const round3 = (n: number) => Math.round(n * 1000) / 1000;
