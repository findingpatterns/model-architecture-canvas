import { useState } from "react";
import { PRESET_COLORS } from "./canvas-colors";

// Color picker: 6 JSON Canvas presets + a custom hex input. Applies the chosen
// token to the current selection (handled by the parent's onPick).
export function ColorPicker({ onPick }: { onPick: (token: string | undefined) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="cv-color-wrap">
      <button className="cv-btn" onClick={() => setOpen((o) => !o)} title="Color of selection">🎨 Color</button>
      {open && (
        <div className="cv-color-pop" onMouseLeave={() => setOpen(false)}>
          {PRESET_COLORS.map(({ token, hex }) => (
            <button
              key={token}
              className="cv-swatch"
              style={{ background: hex }}
              title={`preset ${token}`}
              onClick={() => { onPick(token); setOpen(false); }}
            />
          ))}
          <input
            type="color"
            className="cv-swatch cv-swatch-hex"
            title="custom hex"
            onChange={(e) => { onPick(e.target.value); setOpen(false); }}
          />
          <button className="cv-swatch cv-swatch-none" title="no color" onClick={() => { onPick(undefined); setOpen(false); }}>∅</button>
        </div>
      )}
    </div>
  );
}
