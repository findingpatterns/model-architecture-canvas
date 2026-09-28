// Edit-mode form for the selected component: identity, look, geometry, repeat,
// description and specs. Every commit goes through the store's undo history.
import type { Component, Spec, Vec3 } from "../chip-types.ts";
import { locate, updateComponent, moveComponent, resizeComponent } from "../chip-edit-actions.ts";
import { useChipStore } from "../chip-store.tsx";
import { TextField, NumberField, Vec3Field, ColorField } from "./form-fields.tsx";

export function PropertyPanel() {
  const { state, dispatch } = useChipStore();
  const { chip } = state;
  const loc = state.selected ? locate(chip, state.selected) : null;
  if (!loc) return <div className="panel-body"><p className="muted">Select a part to edit it. Drag the gizmo to move; double-click a part to switch move / resize.</p></div>;

  const c = loc.comp;
  // `external` re-keys the uncontrolled fields — needed when rows are added/removed.
  const commit = (patch: Partial<Component>, external = false) => dispatch({ type: "commit", chip: updateComponent(chip, c.id, patch), external });
  const setSpecs = (specs: Spec[], external = false) => commit({ specs }, external);
  // Re-key uncontrolled fields only on external changes (undo/redo/load/gizmo/selection),
  // so committing one field on blur never steals focus from the next one.
  const k = `${c.id}:${state.revision}`;

  return (
    <div className="panel-body edit" key={k}>
      <div className="panel-kicker mono">{c.id}{loc.scene && ` · inside ${loc.scene}`}</div>
      <TextField label="Name" value={c.name} onCommit={(name) => commit({ name })} />
      <label className="field">
        <span>Group</span>
        <select value={c.group} onChange={(e) => commit({ group: e.target.value })}>
          {Object.keys(chip.groups).map((g) => <option key={g} value={g}>{g}</option>)}
        </select>
      </label>
      <label className="field inline">
        <span>Color</span>
        <ColorField key={c.color ?? c.group} value={c.color ?? chip.groups[c.group]} onCommit={(color) => commit({ color })} />
        {c.color && <button className="btn ghost small" onClick={() => commit({ color: undefined })}>use group color</button>}
      </label>
      <label className="field">
        <span>Shape</span>
        <select value={c.geom.type} onChange={(e) => commit({ geom: { ...c.geom, type: e.target.value as "box" | "cylinder" } })}>
          <option value="box">box</option>
          <option value="cylinder">cylinder</option>
        </select>
      </label>
      <Vec3Field label="Position" value={c.geom.pos} onCommit={(p) => dispatch({ type: "commit", chip: moveComponent(chip, c.id, p) })} />
      <Vec3Field label="Size" value={c.geom.size} min={0.01} onCommit={(s) => dispatch({ type: "commit", chip: resizeComponent(chip, c.id, s) })} />
      {loc.scene === null && <NumberField label="Explode lift" value={c.explode ?? 0} onCommit={(explode) => commit({ explode })} />}

      <details open={!!c.repeat}>
        <summary>Repeat (grid of identical parts)</summary>
        {c.repeat ? (
          <>
            <Vec3Field label="Count" integer min={1} value={c.repeat.count} onCommit={(count) => commit({ repeat: { ...c.repeat!, count } })} />
            <Vec3Field label="Step" value={c.repeat.step} onCommit={(step) => commit({ repeat: { ...c.repeat!, step } })} />
            <NumberField label="Layer spread" value={c.repeat.spread ?? 0} onCommit={(spread) => commit({ repeat: { ...c.repeat!, spread } })} />
            <button className="btn ghost small" onClick={() => commit({ repeat: undefined }, true)}>Remove repeat</button>
          </>
        ) : (
          <button className="btn small" onClick={() => commit({ repeat: { count: [2, 1, 1] as Vec3, step: [c.geom.size[0] * 1.2, 0, 0] as Vec3 } })}>Add repeat</button>
        )}
      </details>

      <TextField label="Description (markdown)" multiline value={c.desc ?? ""} onCommit={(desc) => commit({ desc })} />

      <div className="field">
        <span>Specs</span>
        {(c.specs ?? []).map((s, i) => {
          const upd = (patch: Partial<Spec>) => setSpecs((c.specs ?? []).map((x, j) => (j === i ? { ...x, ...patch } : x)));
          return (
            <div className="spec-row" key={i}>
              <input defaultValue={s.label} placeholder="label" onBlur={(e) => e.target.value !== s.label && upd({ label: e.target.value })} />
              <input defaultValue={s.value} placeholder="value" onBlur={(e) => e.target.value !== s.value && upd({ value: e.target.value })} />
              <input defaultValue={s.source ?? ""} placeholder="https://source" type="url" onBlur={(e) => e.target.value !== (s.source ?? "") && upd({ source: e.target.value || undefined })} />
              <label className="inline small"><input type="checkbox" checked={!!s.estimate} onChange={(e) => upd({ estimate: e.target.checked || undefined })} /> est.</label>
              <button className="btn ghost small" aria-label="Remove spec" onClick={() => setSpecs((c.specs ?? []).filter((_, j) => j !== i), true)}>✕</button>
            </div>
          );
        })}
        <button className="btn small" onClick={() => setSpecs([...(c.specs ?? []), { label: "Label", value: "Value", estimate: true }], true)}>+ Add spec</button>
      </div>
    </div>
  );
}
