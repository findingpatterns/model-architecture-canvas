// Edit-mode actions: structure (add / duplicate / delete), history (undo /
// redo + keyboard shortcuts), and file IO (load a local chip.json, reset).
import { useEffect, useRef, useState } from "react";
import { addComponent, deleteComponent, duplicateComponent, locate } from "../chip-edit-actions.ts";
import { useChipStore } from "../chip-store.tsx";
import { validateChip } from "../../scripts/chip-schema.mjs";
import type { Chip } from "../chip-types.ts";

export function EditToolbar() {
  const { state, dispatch } = useChipStore();
  const { chip, selected } = state;
  const fileRef = useRef<HTMLInputElement>(null);
  const [error, setError] = useState<string | null>(null);
  const loc = selected ? locate(chip, selected) : null;
  const errors = validateChip(chip);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select")) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        dispatch({ type: e.shiftKey ? "redo" : "undo" });
      } else if ((e.key === "Delete" || e.key === "Backspace") && state.selected) {
        e.preventDefault();
        dispatch({ type: "commit", chip: deleteComponent(state.chip, state.selected), select: null });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [dispatch, state.chip, state.selected]);

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    try {
      const parsed = JSON.parse(await f.text()) as Chip;
      const errs = validateChip(parsed);
      if (errs.length) throw new Error(errs.slice(0, 3).join("; "));
      setError(null);
      dispatch({ type: "load", chip: parsed });
    } catch (e) {
      setError(`Could not load: ${(e as Error).message}`);
    }
  };

  return (
    <div className="edit-toolbar">
      <button className="btn small" onClick={() => dispatch({ type: "undo" })} disabled={!state.past.length}>↶ Undo</button>
      <button className="btn small" onClick={() => dispatch({ type: "redo" })} disabled={!state.future.length}>↷ Redo</button>
      <button
        className="btn small"
        onClick={() => {
          const r = addComponent(chip, selected, state.drill ?? null);
          dispatch({ type: "commit", chip: r.chip, select: r.id });
        }}
      >
        + Add {selected ? "child" : "part"}
      </button>
      <button className="btn small" disabled={!loc} onClick={() => { const r = duplicateComponent(chip, selected!); dispatch({ type: "commit", chip: r.chip, select: r.id }); }}>
        Duplicate
      </button>
      <button
        className="btn small danger"
        disabled={!loc}
        onClick={() => {
          const kids = [...chip.components, ...Object.values(chip.scenes ?? {}).flatMap((s) => s.components)].filter((c) => c.parent === selected).length;
          if (kids && !confirm(`Delete "${loc!.comp.name}" and its ${kids} child part(s)?`)) return;
          dispatch({ type: "commit", chip: deleteComponent(chip, selected!), select: null });
        }}
      >
        Delete
      </button>
      <span className="spacer" />
      <button className="btn small" onClick={() => fileRef.current?.click()}>Load file…</button>
      <input ref={fileRef} type="file" accept=".json,application/json" hidden onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
      <button
        className="btn small ghost"
        disabled={chip === state.published}
        onClick={() => confirm("Discard all edits and reload the published chip?") && dispatch({ type: "load", chip: state.published })}
      >
        Reset
      </button>
      {error && <p className="error small">{error}</p>}
      {!!errors.length && <p className="error small" title={errors.join("\n")}>⚠ {errors.length} validation issue(s): {errors[0]}</p>}
    </div>
  );
}
