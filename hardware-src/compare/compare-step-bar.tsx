// Walkthrough controls for compare mode: steps are aligned by `stage`, so the
// same moment of the data path plays on both chips at once. A chip without
// that stage shows "no equivalent" and its view is left undimmed.
import { useEffect } from "react";
import type { StagePair } from "./compare-diff.ts";

const STEP_MS = 3600;

interface Props {
  pairs: StagePair[];
  names: [string, string];
  index: number; // -1 = inactive
  playing: boolean;
  onChange: (index: number, playing: boolean) => void;
}

export function CompareStepBar({ pairs, names, index, playing, onChange }: Props) {
  useEffect(() => {
    if (!playing) return;
    const t = setTimeout(() => {
      if (index + 1 >= pairs.length) onChange(index, false);
      else onChange(index + 1, true);
    }, index < 0 ? 0 : STEP_MS);
    return () => clearTimeout(t);
  }, [playing, index, pairs.length, onChange]);

  if (!pairs.length) return null;
  const cur = index >= 0 ? pairs[index] : null;
  const text = (side: "a" | "b", name: string) =>
    cur?.[side] ? cur[side]!.step.text : `No equivalent in ${name}.`;

  return (
    <div className="step-bar compare-steps">
      <button className="btn icon" aria-label="Previous stage" disabled={index <= 0} onClick={() => onChange(index - 1, false)}>‹</button>
      <button className="btn" onClick={() => onChange(!playing && index >= pairs.length - 1 ? -1 : index, !playing)}>
        {playing ? "Pause" : index < 0 ? "▶ Play both walkthroughs" : "▶ Play"}
      </button>
      <button className="btn icon" aria-label="Next stage" disabled={index >= pairs.length - 1} onClick={() => onChange(index + 1, false)}>›</button>
      <span className="step-count mono">{cur ? `${index + 1}/${pairs.length} · ${cur.stage}` : `${pairs.length} stages`}</span>
      {cur ? (
        <div className="step-pair">
          <p><b>{names[0]}:</b> <span className={cur.a ? "" : "muted"}>{text("a", names[0])}</span></p>
          <p><b>{names[1]}:</b> <span className={cur.b ? "" : "muted"}>{text("b", names[1])}</span></p>
        </div>
      ) : (
        <span className="step-text">The same data path, stage by stage, on both chips.</span>
      )}
      {index >= 0 && <button className="btn ghost" onClick={() => onChange(-1, false)}>Exit</button>}
    </div>
  );
}
