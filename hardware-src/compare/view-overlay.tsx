// HTML overlay for one compare view: chip name + to-scale size label, and a
// floating details card for the part selected in THAT view (so clicking a part
// shows its details right where you clicked, not at the bottom of the page).
//
// Rendered in a layer OUTSIDE the Canvas event container (see compare-views):
// clicks on the card never orbit/pick the 3D view, and still reach React.
import { useChipStore } from "../chip-store.tsx";
import { scaledFootprintMm } from "../chip-geometry.ts";
import { InspectorPanel } from "../ui/inspector-panel.tsx";

export function ViewOverlay({ realSize }: { realSize: boolean }) {
  const { state, dispatch } = useChipStore();
  const size = scaledFootprintMm(state.chip);
  return (
    <div className="view-overlay">
      <div className="compare-label">
        <span className="mono">{state.chip.name}</span>
        {size && (
          <span className="size-label mono" title="Bounding box of the dies and HBM stacks, the parts drawn to scale (estimates where noted)">
            die + HBM ≈ {Math.round(size.w)} × {Math.round(size.d)} mm{realSize ? "" : " · not to scale here"}
          </span>
        )}
      </div>
      {state.selected && (
        <div className="view-card">
          <button className="btn ghost small view-card-close" aria-label="Close details" onClick={() => dispatch({ type: "set", patch: { selected: null } })}>
            ✕
          </button>
          <InspectorPanel />
        </div>
      )}
    </div>
  );
}
