// Side-by-side spec table from both chips' `summary`, aligned by standard key.
// Numeric rows get a ratio and proportional bars; every cell shows its source
// or an "estimate" badge. Clicking a row highlights the related part in 3D.
import type { Chip, SummaryRow } from "../chip-types.ts";
import { formatRatio, joinSummary } from "./compare-diff.ts";

function Cell({ row, max, unit, better }: { row?: SummaryRow; max: number; unit: string | null; better: boolean }) {
  if (!row) return <td className="muted">—</td>;
  return (
    <td className={better ? "better" : ""}>
      <div>{row.value}</div>
      {row.number !== undefined && max > 0 && (
        <div className="bar" aria-hidden="true">
          <span style={{ width: `${Math.max(2, (row.number / max) * 100)}%` }} />
        </div>
      )}
      <div className="cell-meta">
        {row.estimate && <span className="badge">estimate</span>}
        {row.source && /^https?:\/\//i.test(row.source) && (
          <a className="src" href={row.source} target="_blank" rel="noopener noreferrer">source ↗</a>
        )}
        {unit && row.number !== undefined && <span className="muted small">{row.number} {unit}</span>}
      </div>
    </td>
  );
}

export function CompareTable({ a, b, onRole, activeRole }: { a: Chip; b: Chip; onRole: (role: string | null) => void; activeRole: string | null }) {
  const rows = joinSummary(a, b);
  if (!rows.length) return <p className="muted">Neither chip has a summary yet.</p>;
  return (
    <table className="compare-table">
      <thead>
        <tr>
          <th scope="col">Spec</th>
          <th scope="col">{a.name}</th>
          <th scope="col">{b.name}</th>
          <th scope="col">{b.name} vs {a.name}</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const role = r.a?.role ?? r.b?.role ?? null;
          const max = Math.max(r.a?.number ?? 0, r.b?.number ?? 0);
          return (
            <tr
              key={r.key}
              className={role ? `clickable ${role === activeRole ? "active" : ""}` : ""}
              onClick={() => role && onRole(role === activeRole ? null : role)}
              title={role ? "Highlight this part in both 3D views" : undefined}
            >
              <th scope="row">{r.label}</th>
              <Cell row={r.a} max={max} unit={r.unit} better={r.better === "a"} />
              <Cell row={r.b} max={max} unit={r.unit} better={r.better === "b"} />
              <td className="ratio mono">{r.ratio !== null ? formatRatio(r.ratio) : "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
