// Compare page (?compare=a,b): loads two chips, owns one store per chip plus the
// shared view state (explode, flows, highlighted role, aligned walkthrough
// stage), and pushes shared changes into both stores.
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Chip } from "../chip-types.ts";
import { loadChip } from "../chip-loader.ts";
import { initialState, useChipReducer, ChipStoreBridge, type ChipStore } from "../chip-store.tsx";
import { useTheme } from "../use-theme.ts";
import { InspectorPanel } from "../ui/inspector-panel.tsx";
import { alignStages, effectiveStepView } from "./compare-diff.ts";
import { CompareViews } from "./compare-views.tsx";
import { CompareTable } from "./compare-table.tsx";
import { RoleDiffList } from "./role-diff-list.tsx";
import { CompareStepBar } from "./compare-step-bar.tsx";

export function CompareApp({ ids, reducedMotion }: { ids: string[]; reducedMotion: boolean }) {
  const [chips, setChips] = useState<[Chip, Chip] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (ids.length !== 2 || ids[0] === ids[1]) { setError("Compare needs two different chips: ?compare=first,second"); return; }
    Promise.all(ids.map(loadChip))
      .then(([a, b]) => { document.title = `ModelCanvas — ${a.name} vs ${b.name}`; setChips([a, b]); })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (error) return <div className="center-msg"><p>{error}</p><a className="btn" href="../">← Gallery</a></div>;
  if (!chips) return <div className="center-msg"><p className="muted">Loading chips…</p></div>;
  return <CompareLoaded chips={chips} reducedMotion={reducedMotion} />;
}

// Scene id that contains a component with this role (for auto-opening drill-downs).
function sceneWithRole(chip: Chip, role: string): string | null {
  for (const [sid, s] of Object.entries(chip.scenes ?? {})) if (s.components.some((c) => c.role === role)) return sid;
  return null;
}

function CompareLoaded({ chips, reducedMotion }: { chips: [Chip, Chip]; reducedMotion: boolean }) {
  const [a, b] = chips;
  const storeA = useChipReducer(initialState(a, reducedMotion));
  const storeB = useChipReducer(initialState(b, reducedMotion));
  const stores: [ChipStore, ChipStore] = [storeA, storeB];
  const [theme, toggleTheme] = useTheme();
  const [explode, setExplode] = useState(0.55);
  const [flows, setFlows] = useState(!reducedMotion);
  const [role, setRole] = useState<string | null>(null);
  const [stage, setStage] = useState({ index: -1, playing: false });
  const [viewKey, setViewKey] = useState(0);
  const pairs = useMemo(() => alignStages(a, b), [a, b]);

  const broadcast = (patch: Parameters<ChipStore["dispatch"]>[0]) => { storeA.dispatch(patch); storeB.dispatch(patch); };
  useEffect(() => broadcast({ type: "set", patch: { explode } }), [explode]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => broadcast({ type: "set", patch: { flows } }), [flows]); // eslint-disable-line react-hooks/exhaustive-deps
  // One place decides each chip's step + drill-down (and the shared explode):
  // - an active walkthrough stage wins; a chip without that stage keeps the view
  //   its own walkthrough would have at the nearest earlier stage it does have;
  // - otherwise a highlighted role opens the drill-down that contains it (or closes).
  useEffect(() => {
    const p = stage.index >= 0 ? pairs[stage.index] : null;
    let explodeFromStep: number | undefined;
    stores.forEach((s, i) => {
      const side = i === 0 ? "a" : "b";
      const roleScene = role ? sceneWithRole(chips[i], role) : null;
      if (!p) {
        s.dispatch({ type: "set", patch: { step: -1, drill: roleScene } });
        return;
      }
      let k = -1; // this chip's step index at, or nearest before, the current stage
      for (let j = stage.index; j >= 0 && k < 0; j--) k = pairs[j][side]?.index ?? -1;
      const view = effectiveStepView(chips[i], k);
      explodeFromStep ??= view.explode;
      s.dispatch({ type: "set", patch: { step: p[side]?.index ?? -1, drill: view.drill ?? roleScene } });
    });
    if (explodeFromStep !== undefined) setExplode(explodeFromStep); // keeps the slider truthful
  }, [stage.index, role]); // eslint-disable-line react-hooks/exhaustive-deps

  const onStage = useCallback((index: number, playing: boolean) => setStage({ index, playing }), []);

  return (
    <div className="compare-app">
      <header className="topbar">
        <a className="btn ghost" href="../">← Gallery</a>
        <div className="title">
          <span className="title-name">{a.name} vs {b.name}</span>
          <span className="title-sub mono">Compare · structure, not scale</span>
        </div>
        <a className="btn icon" href={`?compare=${b.id},${a.id}`} aria-label="Swap sides" title="Swap sides">⇄</a>
        <label className="slider">
          <span className="mono">Explode</span>
          <input type="range" min={0} max={1} step={0.01} value={explode} onChange={(e) => setExplode(Number(e.target.value))} aria-label="Explode layers" />
        </label>
        <button className={`btn ${flows ? "on" : ""}`} aria-pressed={flows} onClick={() => setFlows(!flows)}>Data flow</button>
        <button className="btn" onClick={() => setViewKey((k) => k + 1)}>Reset view</button>
        <span className="spacer" />
        <a className="btn" href={`?chip=${a.id}`}>Open {a.name}</a>
        <a className="btn" href={`?chip=${b.id}`}>Open {b.name}</a>
        <button className="btn icon" aria-label="Toggle dark / light theme" onClick={toggleTheme}>{theme === "light" ? "🌙" : "☀"}</button>
      </header>

      <CompareViews stores={stores} highlightRole={role} reducedMotion={reducedMotion} viewKey={viewKey} />
      <CompareStepBar pairs={pairs} names={[a.name, b.name]} index={stage.index} playing={stage.playing} onChange={onStage} />

      <main className="compare-body">
        <RoleDiffList a={a} b={b} activeRole={role} onRole={setRole} />
        <CompareTable a={a} b={b} activeRole={role} onRole={setRole} />
        <div className="compare-inspectors">
          {stores.map((s, i) => (
            <section key={i} className="panel-card" aria-label={`${chips[i].name} details`}>
              <ChipStoreBridge store={s}>
                <InspectorPanel />
              </ChipStoreBridge>
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
