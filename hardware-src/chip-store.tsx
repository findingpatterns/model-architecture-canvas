// App state for the 3D chip view: the chip being shown/edited (with undo
// history) plus view state (explode, drill-down, selection, steps, toggles).
import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { Chip } from "./chip-types.ts";

export interface ViewState {
  chip: Chip;
  published: Chip; // as loaded from the site — "Reset" target and dirty check
  past: Chip[];
  future: Chip[];
  explode: number;
  drill: string | null;
  selected: string | null;
  edit: boolean;
  flows: boolean;
  autoRotate: boolean;
  step: number; // -1 = no guided step active
  playing: boolean;
  revision: number; // bumps on undo/redo/load/gizmo edits → edit forms re-read their values
}

export type Action =
  | { type: "load"; chip: Chip; published?: boolean }
  | { type: "commit"; chip: Chip; select?: string | null; external?: boolean }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "set"; patch: Partial<Omit<ViewState, "chip" | "published" | "past" | "future" | "revision">> };

export type ViewPatch = Extract<Action, { type: "set" }>["patch"];

const HISTORY_LIMIT = 100;

export function initialState(chip: Chip, reducedMotion: boolean): ViewState {
  return {
    chip,
    published: chip,
    past: [],
    future: [],
    explode: 0.55,
    drill: null,
    selected: null,
    edit: false,
    flows: !reducedMotion,
    autoRotate: false,
    step: -1,
    playing: false,
    revision: 0,
  };
}

// Keep view state consistent after the chip changes (ids may have disappeared).
function sanitize(s: ViewState): ViewState {
  const ids = new Set(s.chip.components.map((c) => c.id));
  for (const sc of Object.values(s.chip.scenes ?? {})) for (const c of sc.components) ids.add(c.id);
  const drill = s.drill && s.chip.scenes?.[s.drill] ? s.drill : null;
  const selected = s.selected && ids.has(s.selected) ? s.selected : null;
  const step = s.step >= (s.chip.steps?.length ?? 0) ? -1 : s.step;
  return { ...s, drill, selected, step };
}

export function reducer(s: ViewState, a: Action): ViewState {
  switch (a.type) {
    case "load":
      return sanitize({
        ...s,
        chip: a.chip,
        published: a.published ? a.chip : s.published,
        past: a.published ? [] : [...s.past, s.chip],
        future: [],
        revision: s.revision + 1,
      });
    case "commit":
      if (a.chip === s.chip) return s;
      return sanitize({
        ...s,
        chip: a.chip,
        past: [...s.past, s.chip].slice(-HISTORY_LIMIT),
        future: [],
        selected: a.select === undefined ? s.selected : a.select,
        revision: a.external || a.select !== undefined ? s.revision + 1 : s.revision,
      });
    case "undo":
      if (!s.past.length) return s;
      return sanitize({ ...s, chip: s.past[s.past.length - 1], past: s.past.slice(0, -1), future: [s.chip, ...s.future], revision: s.revision + 1 });
    case "redo":
      if (!s.future.length) return s;
      return sanitize({ ...s, chip: s.future[0], past: [...s.past, s.chip], future: s.future.slice(1), revision: s.revision + 1 });
    case "set":
      return { ...s, ...a.patch };
  }
}

const Ctx = createContext<{ state: ViewState; dispatch: Dispatch<Action> } | null>(null);

export function ChipStoreProvider({ init, children }: { init: ViewState; children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, init);
  return <Ctx.Provider value={{ state, dispatch }}>{children}</Ctx.Provider>;
}

export function useChipStore() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useChipStore outside ChipStoreProvider");
  return v;
}
