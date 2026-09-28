// "What exists in one chip but not the other", from component roles.
// Each item toggles a highlight of that part in both 3D views.
import type { Chip } from "../chip-types.ts";
import { roleDiff } from "./compare-diff.ts";

function Group({ title, items, activeRole, onRole }: { title: string; items: { role: string; name: string }[]; activeRole: string | null; onRole: (r: string | null) => void }) {
  return (
    <div className="diff-group">
      <h3 className="mono">{title}</h3>
      {items.length ? (
        <div className="chips">
          {items.map((it) => (
            <button key={it.role} className={`btn small ${activeRole === it.role ? "on" : ""}`} aria-pressed={activeRole === it.role} onClick={() => onRole(activeRole === it.role ? null : it.role)}>
              {it.name}
            </button>
          ))}
        </div>
      ) : (
        <p className="muted small">Nothing — every modeled part has a counterpart.</p>
      )}
    </div>
  );
}

export function RoleDiffList({ a, b, activeRole, onRole }: { a: Chip; b: Chip; activeRole: string | null; onRole: (r: string | null) => void }) {
  const d = roleDiff(a, b);
  return (
    <section className="role-diff" aria-label="Structural differences">
      <Group title={`Only in ${b.name}`} items={d.onlyB} activeRole={activeRole} onRole={onRole} />
      <Group title={`Only in ${a.name}`} items={d.onlyA} activeRole={activeRole} onRole={onRole} />
      <p className="muted small">
        {d.shared.length} part types exist in both. Compares the modeled structure only; parts without a role are not compared.
      </p>
    </section>
  );
}
