// Read-only details for the selected component: description, specs with
// source links / estimate badges, and a button to open its drill-down scene.
import { renderMarkdown } from "../../editor-src/render-markdown.ts";
import { locate } from "../chip-edit-actions.ts";
import { useChipStore } from "../chip-store.tsx";

export function InspectorPanel() {
  const { state, dispatch } = useChipStore();
  const { chip } = state;
  const loc = state.selected ? locate(chip, state.selected) : null;

  if (!loc) {
    return (
      <div className="panel-body">
        <p className="muted">Click any part of the chip to see what it is.</p>
        {chip.disclaimer && <p className="muted small">{chip.disclaimer}</p>}
      </div>
    );
  }
  const c = loc.comp;
  const count = c.repeat ? c.repeat.count.reduce((a, b) => a * b, 1) : 1;
  return (
    <div className="panel-body">
      <div className="panel-kicker mono">
        <span className="swatch" style={{ background: c.color ?? chip.groups[c.group] }} />
        {c.group}
        {count > 1 && ` · ×${count} (symbolic)`}
        {loc.scene && ` · inside ${loc.scene}`}
      </div>
      <h2 className="panel-title">{c.name}</h2>
      {c.desc && <p className="panel-desc" dangerouslySetInnerHTML={{ __html: renderMarkdown(c.desc) }} />}
      {!!c.specs?.length && (
        <table className="specs">
          <tbody>
            {c.specs.map((s, i) => (
              <tr key={i}>
                <th>{s.label}</th>
                <td>
                  {s.value}{" "}
                  {s.estimate && <span className="badge">estimate</span>}
                  {s.source && /^https?:\/\//i.test(s.source) && (
                    <a className="src" href={s.source} target="_blank" rel="noopener noreferrer">
                      source ↗
                    </a>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {c.drill && (
        <button className="btn accent" onClick={() => dispatch({ type: "set", patch: { drill: state.drill === c.drill ? null : c.drill } })}>
          {state.drill === c.drill ? "Close inside view" : "Open inside ↗"}
        </button>
      )}
    </div>
  );
}
