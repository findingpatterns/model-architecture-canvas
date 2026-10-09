// Canonical chip.json text: 2-space JSON + trailing newline, key order as authored,
// arrays of primitives kept on one line ("pos": [0, 1, 0]) so files stay hand-editable.
// Loading a shipped file and serializing it unchanged is byte-identical (tested),
// so downloads diff cleanly against the repo.
import type { Chip } from "./chip-types.ts";

export function serializeChip(chip: Chip): string {
  const text = JSON.stringify(chip, null, 2).replace(/\[\n\s+([^[\]{}]*?)\n\s*\]/g, (_m, inner: string) =>
    `[${inner.split(/,\n\s*/).join(", ")}]`,
  );
  return text + "\n";
}

export function downloadChip(chip: Chip): void {
  const blob = new Blob([serializeChip(chip)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${chip.id}.chip.json`;
  a.click();
  URL.revokeObjectURL(url);
}
