// Top bar: navigation, view toggles (explode, flows, auto-rotate), cross-links
// to the 2D diagram, download, edit-mode toggle and theme.
import { useChipStore, type ViewPatch } from "../chip-store.tsx";
import { downloadChip } from "../chip-serialize.ts";

interface Props {
  onResetView: () => void;
  theme: "dark" | "light";
  onToggleTheme: () => void;
}

export function ViewerToolbar({ onResetView, theme, onToggleTheme }: Props) {
  const { state, dispatch } = useChipStore();
  const { chip } = state;
  const set = (patch: ViewPatch) => dispatch({ type: "set", patch });

  return (
    <header className="topbar">
      <a className="btn ghost" href="../">← Gallery</a>
      <div className="title">
        <span className="title-name">{chip.name}</span>
        <span className="title-sub mono">{chip.arch} · 3D</span>
      </div>
      <label className="slider">
        <span className="mono">Explode</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={state.explode}
          onChange={(e) => set({ explode: Number(e.target.value) })}
          aria-label="Explode layers"
        />
      </label>
      <button className={`btn ${state.flows ? "on" : ""}`} aria-pressed={state.flows} onClick={() => set({ flows: !state.flows })}>
        Data flow
      </button>
      <button className={`btn ${state.autoRotate ? "on" : ""}`} aria-pressed={state.autoRotate} onClick={() => set({ autoRotate: !state.autoRotate })}>
        Auto-rotate
      </button>
      <button className="btn" onClick={onResetView}>Reset view</button>
      <span className="spacer" />
      <a className="btn" href={`../?model=${encodeURIComponent(chip.id)}`}>2D diagram</a>
      <button className="btn" onClick={() => downloadChip(chip)}>Download chip.json</button>
      <button
        className={`btn ${state.edit ? "on" : "accent"}`}
        aria-pressed={state.edit}
        onClick={() => set({ edit: !state.edit, step: -1, playing: false })}
      >
        {state.edit ? "Done editing" : "✎ Edit"}
      </button>
      <button className="btn icon" aria-label="Toggle dark / light theme" onClick={onToggleTheme}>
        {theme === "light" ? "🌙" : "☀"}
      </button>
    </header>
  );
}
