// 3D chip explorer: loads hardware-data/<id>.json (?chip=<id>), validates it,
// then renders the scene + toolbar + side panel (inspector, or editors in Edit mode).
import { useEffect, useMemo, useState } from "react";
import { worldScales } from "./chip-geometry.ts";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import type { Chip } from "./chip-types.ts";
import { loadChip } from "./chip-loader.ts";
import { CompareApp } from "./compare/compare-app.tsx";
import { ChipStoreProvider, initialState, useChipStore } from "./chip-store.tsx";
import { ChipScene } from "./scene/chip-scene.tsx";
import { ViewerToolbar } from "./ui/viewer-toolbar.tsx";
import { InspectorPanel } from "./ui/inspector-panel.tsx";
import { StepBar, useApplyStep } from "./ui/step-bar.tsx";
import { EditToolbar } from "./edit/edit-toolbar.tsx";
import { PropertyPanel } from "./edit/property-panel.tsx";
import { StepsEditor } from "./edit/steps-editor.tsx";
import { useTheme } from "./use-theme.ts";

const reducedMotion = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

export function App() {
  const [chip, setChip] = useState<Chip | null>(null);
  const [error, setError] = useState<string | null>(null);

  const params = new URLSearchParams(location.search);
  const compareIds = params.get("compare")?.split(",") ?? null;

  useEffect(() => {
    if (compareIds) return;
    const id = params.get("chip");
    if (!id) { setError("No chip selected. Open one from the gallery."); return; }
    loadChip(id)
      .then((data) => {
        document.title = `ModelCanvas — ${data.name} (3D)`;
        setChip(data);
      })
      .catch((e) => setError(e.message));
    // Route params are read once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (compareIds) return <CompareApp ids={compareIds} reducedMotion={reducedMotion} />;
  if (error) return <div className="center-msg"><p>{error}</p><a className="btn" href="../?category=hardware">← Gallery</a></div>;
  if (!chip) return <div className="center-msg"><p className="muted">Loading chip…</p></div>;
  return (
    <ChipStoreProvider init={initialState(chip, reducedMotion)}>
      <Explorer />
    </ChipStoreProvider>
  );
}

function Explorer() {
  const { state } = useChipStore();
  const [theme, toggleTheme] = useTheme();
  const [viewKey, setViewKey] = useState(0);
  const [tip, setTip] = useState<{ name: string; x: number; y: number } | null>(null);
  const [tab, setTab] = useState<"part" | "steps">("part");
  useApplyStep();
  // Fit scale is frozen per published chip: edits that grow the footprint must
  // not rescale the whole scene under the user's gizmo.
  const worldScale = useMemo(() => worldScales([state.published], "fit")[0], [state.published]);

  const dirty = state.chip !== state.published;
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const onHover = (name: string | null, e?: ThreeEvent<PointerEvent>) =>
    setTip(name && e ? { name, x: e.nativeEvent.offsetX, y: e.nativeEvent.offsetY } : null);

  return (
    <div className={`app ${state.edit ? "is-editing" : ""}`}>
      <ViewerToolbar chipId={state.chip.id} onResetView={() => setViewKey((k) => k + 1)} theme={theme} onToggleTheme={toggleTheme} />
      {state.edit && <EditToolbar />}
      <main className="stage">
        <div className="canvas-wrap" onPointerLeave={() => setTip(null)}>
          <Canvas key={viewKey} camera={{ position: [14, 13, 16], fov: 40 }} dpr={[1, 2]}>
            <ChipScene onHover={onHover} reducedMotion={reducedMotion} worldScale={worldScale} />
          </Canvas>
          {tip && <div className="tip" style={{ left: tip.x + 14, top: tip.y + 14 }}>{tip.name}</div>}
          <p className="hint mono">Drag to rotate · scroll / pinch to zoom · right-drag to pan · click a part</p>
        </div>
        <aside className="panel" aria-label={state.edit ? "Editor" : "Details"}>
          {state.edit ? (
            <>
              <div className="tabs" role="tablist">
                <button role="tab" aria-selected={tab === "part"} className={tab === "part" ? "on" : ""} onClick={() => setTab("part")}>Part</button>
                <button role="tab" aria-selected={tab === "steps"} className={tab === "steps" ? "on" : ""} onClick={() => setTab("steps")}>Walkthrough</button>
              </div>
              {tab === "part" ? <PropertyPanel /> : <StepsEditor />}
            </>
          ) : (
            <InspectorPanel />
          )}
        </aside>
      </main>
      <StepBar />
    </div>
  );
}
