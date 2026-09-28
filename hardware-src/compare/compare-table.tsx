// Side-by-side spec table from both chips' `summary`, aligned by standard key.
// Columns follow the page title (left chip, right chip); the ratio is left ÷ right.
// Every cell shows its source and/or an explained "estimate" badge. Clicking a
// row highlights the related part in both 3D views.
import type { Chip, SummaryRow } from "../chip-types.ts";
import { EstimateBadge } from "../ui/estimate-badge.tsx";
import { formatRatio, joinSummary } from "./compare-diff.ts";
import { isHttpUrl as isHttp } from "../url-utils.ts";

function Cell({ row, max, better, chipName }: { row?: SummaryRow; max: number; better: boolean; chipName: string }) {
  if (!row)
    return (
      <td className="muted" title={`No value in ${chipName}'s data for this spec — usually because the vendor has not published it.`}>
        not listed
      </td>
    );
  return (
    <td className={better ? "better" : ""}>
      <div>{row.value}</div>
      {row.number !== undefined && max > 0 && (
        <div className="bar" aria-hidden="true">
          <span style={{ width: `${Math.max(2, (row.number / max) * 100)}%` }} />
        </div>
      )}
      <div className="cell-meta">
        {row.estimate && <EstimateBadge sourced={isHttp(row.source)} />}
        {isHttp(row.source) && (
          <a className="src" href={row.source} target="_blank" rel="noopener noreferrer">source ↗</a>
        )}
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
          <th scope="col" title={`${a.name} divided by ${b.name}`}>{a.name} ÷ {b.name}</th>
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
              <Cell row={r.a} max={max} better={r.better === "a"} chipName={a.name} />
              <Cell row={r.b} max={max} better={r.better === "b"} chipName={b.name} />
              <td className="ratio mono">{r.ratio !== null ? formatRatio(r.ratio) : "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
