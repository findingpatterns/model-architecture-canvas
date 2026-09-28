// TypeScript mirror of the chip.json schema (validated at runtime by scripts/chip-schema.mjs).
export type Vec3 = [number, number, number];

export interface Spec {
  label: string;
  value: string;
  source?: string;
  estimate?: boolean;
}

export interface Component {
  id: string;
  parent?: string | null;
  name: string;
  group: string;
  color?: string;
  geom: { type: "box" | "cylinder"; size: Vec3; pos: Vec3 };
  explode?: number;
  repeat?: { count: Vec3; step: Vec3; spread?: number } | null;
  desc?: string;
  specs?: Spec[];
  drill?: string;
  role?: string; // shared vocabulary for comparing chips (e.g. "tensor-memory")
}

export interface Scene {
  anchor: string;
  offset?: Vec3;
  components: Component[];
}

export interface Flow {
  id: string;
  from: string;
  to: string;
  group: string;
  particles?: number;
}

export interface Step {
  text: string;
  focus?: string[];
  dot?: string;
  explode?: number;
  drill?: string | null;
  stage?: string; // aligns walkthroughs across chips when comparing
}

export interface SummaryRow {
  key: string; // one of SUMMARY_KEYS
  value: string; // display text, including basis (e.g. "per 8-GPU DGX / 8")
  number?: number; // in the key's unit, for ratios/bars
  role?: string; // component role to highlight from this row
  source?: string;
  estimate?: boolean;
}

export interface Chip {
  schemaVersion: 1;
  id: string;
  name: string;
  vendor: string;
  arch: string;
  disclaimer?: string;
  groups: Record<string, string>;
  components: Component[];
  scenes?: Record<string, Scene>;
  flows?: Flow[];
  steps?: Step[];
  summary?: SummaryRow[];
  scale?: { mmPerUnit: number; basis: string; source?: string; estimate?: boolean };
}

// Where a component lives: the main scene (null) or a drill-down scene id.
export type SceneKey = string | null;
