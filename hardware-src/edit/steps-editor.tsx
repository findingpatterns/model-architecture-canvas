// Edit-mode editor for chip.steps (the guided walkthrough): text, focus ids,
// dot target, drill scene and explode; add / delete / reorder; preview a step.
import type { Step } from "../chip-types.ts";
import { setSteps } from "../chip-edit-actions.ts";
import { useChipStore } from "../chip-store.tsx";

export function StepsEditor() {
  const { state, dispatch } = useChipStore();
  const { chip } = state;
  const steps = chip.steps ?? [];
  const ids = [...chip.components, ...Object.values(chip.scenes ?? {}).flatMap((s) => s.components)].map((c) => c.id);
  const scenes = Object.keys(chip.scenes ?? {});
  // Structural changes (add/delete/reorder) re-key the list; text edits don't.
  const save = (next: Step[], external = false) => dispatch({ type: "commit", chip: setSteps(chip, next), external });
  const upd = (i: number, patch: Partial<Step>) => save(steps.map((s, j) => (j === i ? { ...s, ...patch } : s)));
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= steps.length) return;
    const next = [...steps];
    [next[i], next[j]] = [next[j], next[i]];
    save(next, true);
  };

  return (
    <div className="panel-body edit" key={state.revision}>
      <datalist id="component-ids">{ids.map((id) => <option key={id} value={id} />)}</datalist>
      {steps.map((s, i) => (
        <div className={`step-card ${state.step === i ? "active" : ""}`} key={i}>
          <div className="step-card-head">
            <span className="mono">#{i + 1}</span>
            <button className="btn ghost small" onClick={() => dispatch({ type: "set", patch: { step: i, playing: false } })}>Preview</button>
            <button className="btn ghost small" aria-label="Move up" onClick={() => move(i, -1)}>↑</button>
            <button className="btn ghost small" aria-label="Move down" onClick={() => move(i, 1)}>↓</button>
            <button className="btn ghost small" aria-label="Delete step" onClick={() => save(steps.filter((_, j) => j !== i), true)}>✕</button>
          </div>
          <textarea rows={2} defaultValue={s.text} onBlur={(e) => e.target.value !== s.text && upd(i, { text: e.target.value })} />
          <label className="field">
            <span>Focus (comma-separated ids)</span>
            <input
              defaultValue={(s.focus ?? []).join(", ")}
              onBlur={(e) => {
                const focus = e.target.value.split(",").map((x) => x.trim()).filter(Boolean);
                if (focus.join(",") !== (s.focus ?? []).join(",")) upd(i, { focus });
              }}
            />
          </label>
          <div className="row">
            <label className="field">
              <span>Dot</span>
              <input list="component-ids" defaultValue={s.dot ?? ""} onBlur={(e) => e.target.value !== (s.dot ?? "") && upd(i, { dot: e.target.value || undefined })} />
            </label>
            <label className="field">
              <span>Inside view</span>
              <select value={s.drill === undefined ? "__keep" : s.drill ?? "__close"} onChange={(e) => upd(i, { drill: e.target.value === "__keep" ? undefined : e.target.value === "__close" ? null : e.target.value })}>
                <option value="__keep">keep as is</option>
                <option value="__close">close</option>
                {scenes.map((sc) => <option key={sc} value={sc}>{sc}</option>)}
              </select>
            </label>
            <label className="field">
              <span>Explode</span>
              <input type="number" min={0} max={1} step={0.05} defaultValue={s.explode ?? ""} onBlur={(e) => upd(i, { explode: e.target.value === "" ? undefined : Math.min(1, Math.max(0, Number(e.target.value))) })} />
            </label>
          </div>
        </div>
      ))}
      <button className="btn small" onClick={() => save([...steps, { text: "New step", focus: state.selected ? [state.selected] : [], ...(state.selected ? { dot: state.selected } : {}) }], true)}>
        + Add step{state.selected ? " (focus selected part)" : ""}
      </button>
    </div>
  );
}
