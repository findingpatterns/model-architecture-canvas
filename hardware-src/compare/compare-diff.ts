// Pure comparison helpers (no React/three): align two chips' summary rows by
// standard key, diff their component roles, and align walkthroughs by stage.
import { SUMMARY_KEYS } from "../../scripts/chip-schema.mjs";
import type { Chip, Component, Step, SummaryRow } from "../chip-types.ts";

export interface SummaryPair {
  key: string;
  label: string;
  unit: string | null;
  a?: SummaryRow;
  b?: SummaryRow;
  ratio: number | null; // a / b (left ÷ right, matching the "A vs B" title) when both numeric and non-zero
  better: "a" | "b" | null; // side with the better value, when the key is directional
}

// Rows in SUMMARY_KEYS order, keeping only keys at least one chip reports.
export function joinSummary(a: Chip, b: Chip): SummaryPair[] {
  const byKey = (c: Chip) => new Map((c.summary ?? []).map((r) => [r.key, r]));
  const ma = byKey(a);
  const mb = byKey(b);
  return Object.entries(SUMMARY_KEYS)
    .filter(([k]) => ma.has(k) || mb.has(k))
    .map(([key, meta]) => {
      const ra = ma.get(key);
      const rb = mb.get(key);
      const na = ra?.number;
      const nb = rb?.number;
      const ratio = na !== undefined && nb !== undefined && na !== 0 && nb !== 0 ? na / nb : null;
      let better: "a" | "b" | null = null;
      if (ratio !== null && meta.higherIsBetter !== null && ratio !== 1)
        better = (ratio > 1) === meta.higherIsBetter ? "a" : "b";
      return { key, label: meta.label, unit: meta.unit, a: ra, b: rb, ratio, better };
    });
}

function roleNames(c: Chip): Map<string, string> {
  const all: Component[] = [...c.components, ...Object.values(c.scenes ?? {}).flatMap((s) => s.components)];
  const out = new Map<string, string>();
  for (const comp of all) if (comp.role && !out.has(comp.role)) out.set(comp.role, comp.name);
  return out;
}

export interface RoleDiff {
  shared: string[];
  onlyA: { role: string; name: string }[];
  onlyB: { role: string; name: string }[];
}

// Roles present in one chip's model but not the other's. Only parts that carry
// a role are compared, so unmodeled parts never show up as "missing".
export function roleDiff(a: Chip, b: Chip): RoleDiff {
  const ra = roleNames(a);
  const rb = roleNames(b);
  return {
    shared: [...ra.keys()].filter((r) => rb.has(r)),
    onlyA: [...ra].filter(([r]) => !rb.has(r)).map(([role, name]) => ({ role, name })),
    onlyB: [...rb].filter(([r]) => !ra.has(r)).map(([role, name]) => ({ role, name })),
  };
}

export interface StagePair {
  stage: string;
  a: { index: number; step: Step } | null;
  b: { index: number; step: Step } | null;
}

// Union of stages, ordered by chip A then B's extras inserted after their
// predecessor stage. Steps without a stage are skipped.
export function alignStages(a: Chip, b: Chip): StagePair[] {
  const idx = (c: Chip) => new Map((c.steps ?? []).flatMap((s, i) => (s.stage ? [[s.stage, { index: i, step: s }] as const] : [])));
  const ia = idx(a);
  const ib = idx(b);
  const order = [...ia.keys()];
  let prev: string | null = null;
  for (const st of ib.keys()) {
    if (!order.includes(st)) order.splice(prev ? order.indexOf(prev) + 1 : 0, 0, st);
    prev = st;
  }
  return order.map((stage) => ({ stage, a: ia.get(stage) ?? null, b: ib.get(stage) ?? null }));
}

// A step's `drill`/`explode` are "sticky": undefined means "keep what the
// previous step set". Resolve what is in effect at step `index` (walking from 0).
export function effectiveStepView(chip: Chip, index: number): { drill: string | null; explode?: number } {
  let drill: string | null = null;
  let explode: number | undefined;
  for (const s of (chip.steps ?? []).slice(0, index + 1)) {
    if (s.drill !== undefined) drill = s.drill;
    if (s.explode !== undefined) explode = s.explode;
  }
  return { drill, explode };
}

// "2.6×" = left chip is 2.6 times the right one; "0.62×" = about 3/5 of it.
export function formatRatio(r: number): string {
  return `${r >= 10 ? r.toFixed(0) : r >= 1 ? r.toFixed(1) : r.toFixed(2)}×`;
}
