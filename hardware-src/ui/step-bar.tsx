// Guided walkthrough controls for chip.steps: prev / play-pause / next + caption.
// Applying a step (explode, drill) happens in useApplyStep so edits and
// playback share one code path.
import { useEffect } from "react";
import { useChipStore } from "../chip-store.tsx";

const STEP_MS = 3200;

export function useApplyStep() {
  const { state, dispatch } = useChipStore();
  const step = state.step >= 0 ? state.chip.steps?.[state.step] : undefined;
  useEffect(() => {
    if (!step) return;
    const patch: { explode?: number; drill?: string | null } = {};
    if (step.explode !== undefined) patch.explode = step.explode;
    if (step.drill !== undefined) patch.drill = step.drill;
    if (Object.keys(patch).length) dispatch({ type: "set", patch });
    // Only when the active step changes — not on every chip edit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.step]);

  useEffect(() => {
    if (!state.playing) return;
    const n = state.chip.steps?.length ?? 0;
    const t = setTimeout(() => {
      if (state.step + 1 >= n) dispatch({ type: "set", patch: { playing: false } });
      else dispatch({ type: "set", patch: { step: state.step + 1 } });
    }, state.step < 0 ? 0 : STEP_MS);
    return () => clearTimeout(t);
  }, [state.playing, state.step, state.chip.steps?.length, dispatch]);
}

export function StepBar() {
  const { state, dispatch } = useChipStore();
  const steps = state.chip.steps ?? [];
  if (!steps.length) return null;
  const go = (i: number) => dispatch({ type: "set", patch: { step: Math.max(0, Math.min(steps.length - 1, i)), playing: false } });
  const cur = state.step >= 0 ? steps[state.step] : null;
  return (
    <div className="step-bar">
      <button className="btn icon" aria-label="Previous step" onClick={() => go(state.step - 1)} disabled={state.step <= 0}>‹</button>
      <button
        className="btn"
        onClick={() => {
          const restart = state.step >= steps.length - 1;
          dispatch({ type: "set", patch: { playing: !state.playing, step: restart && !state.playing ? -1 : state.step } });
        }}
      >
        {state.playing ? "Pause" : state.step < 0 ? "▶ Play walkthrough" : "▶ Play"}
      </button>
      <button className="btn icon" aria-label="Next step" onClick={() => go(state.step + 1)} disabled={state.step >= steps.length - 1}>›</button>
      <span className="step-count mono">{state.step >= 0 ? `${state.step + 1}/${steps.length}` : `${steps.length} steps`}</span>
      <span className="step-text">{cur?.text ?? "A guided tour of how data moves through the chip."}</span>
      {state.step >= 0 && (
        <button className="btn ghost" onClick={() => dispatch({ type: "set", patch: { step: -1, playing: false } })}>
          Exit
        </button>
      )}
    </div>
  );
}
