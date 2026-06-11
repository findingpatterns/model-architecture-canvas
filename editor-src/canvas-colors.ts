// JSON Canvas preset color digits → hex, for RENDERING only.
// The adapter preserves the original `color` token (digit or hex) verbatim on
// export; this map just turns a preset into a paintable color in the editor.
// Values match the Obsidian Canvas defaults.
export const CANVAS_COLORS: Record<string, string> = {
  "1": "#fb464c", // red
  "2": "#e9973f", // orange
  "3": "#e0de71", // yellow
  "4": "#44cf6e", // green
  "5": "#53dfdd", // cyan
  "6": "#a882ff", // purple
};

// Resolve a JSON Canvas color token (preset digit or hex) to a CSS color.
// Returns undefined when no color is set (caller falls back to a theme default).
export function resolveColor(color?: string): string | undefined {
  if (!color) return undefined;
  return CANVAS_COLORS[color] ?? color; // hex (or any CSS color) passes through
}

// The 6 presets, for the color-picker UI.
export const PRESET_COLORS = Object.entries(CANVAS_COLORS).map(([token, hex]) => ({ token, hex }));
