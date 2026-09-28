// Small uncontrolled form fields that commit on blur / Enter, so typing does
// not flood the undo history with one entry per keystroke. Callers re-key them
// when the underlying value changes (undo/redo) to refresh defaults.
import { useEffect, useRef } from "react";
import type { Vec3 } from "../chip-types.ts";

const onEnterBlur = (e: React.KeyboardEvent<HTMLInputElement>) => {
  if (e.key === "Enter") (e.target as HTMLInputElement).blur();
};

export function TextField({ label, value, onCommit, multiline }: { label: string; value: string; onCommit: (v: string) => void; multiline?: boolean }) {
  return (
    <label className="field">
      <span>{label}</span>
      {multiline ? (
        <textarea defaultValue={value} rows={4} onBlur={(e) => e.target.value !== value && onCommit(e.target.value)} />
      ) : (
        <input defaultValue={value} onKeyDown={onEnterBlur} onBlur={(e) => e.target.value !== value && onCommit(e.target.value)} />
      )}
    </label>
  );
}

export function NumberField({ label, value, onCommit, step = 0.1, min }: { label: string; value: number; onCommit: (v: number) => void; step?: number; min?: number }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        min={min}
        defaultValue={value}
        onKeyDown={onEnterBlur}
        onBlur={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n) && n !== value && (min === undefined || n >= min)) onCommit(n);
        }}
      />
    </label>
  );
}

export function Vec3Field({ label, value, onCommit, step = 0.1, min, integer }: { label: string; value: Vec3; onCommit: (v: Vec3) => void; step?: number; min?: number; integer?: boolean }) {
  return (
    <fieldset className="field vec3">
      <legend>{label}</legend>
      {value.map((v, i) => (
        <input
          key={i}
          type="number"
          step={integer ? 1 : step}
          min={min}
          aria-label={`${label} ${"xyz"[i]}`}
          defaultValue={v}
          onKeyDown={onEnterBlur}
          onBlur={(e) => {
            let n = Number(e.target.value);
            if (!Number.isFinite(n) || (min !== undefined && n < min)) return;
            if (integer) n = Math.round(n);
            if (n === v) return;
            const next = [...value] as Vec3;
            next[i] = n;
            onCommit(next);
          }}
        />
      ))}
    </fieldset>
  );
}

// Native "change" fires once when the picker closes; React's onChange fires on
// every drag step and would flood the undo history.
export function ColorField({ value, onCommit }: { value: string; onCommit: (v: string) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const input = ref.current!;
    const onChange = () => input.value !== value && onCommit(input.value);
    input.addEventListener("change", onChange);
    return () => input.removeEventListener("change", onChange);
  }, [value, onCommit]);
  return <input ref={ref} type="color" defaultValue={value} aria-label="Color" />;
}
