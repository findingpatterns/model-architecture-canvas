// "Compare with…" dropdown in the single-chip toolbar: lists the other chips in
// the catalog and opens ?compare=<this>,<other>.
import { useEffect, useState } from "react";
import { listChips, type ChipListing } from "../chip-loader.ts";

export function ComparePicker({ chipId }: { chipId: string }) {
  const [others, setOthers] = useState<ChipListing[]>([]);
  useEffect(() => {
    listChips()
      .then((all) => setOthers(all.filter((c) => c.id !== chipId)))
      .catch(() => setOthers([])); // catalog unavailable → just hide the picker
  }, [chipId]);
  if (!others.length) return null;
  return (
    <select
      className="compare-picker"
      aria-label="Compare with another chip"
      value=""
      onChange={(e) => e.target.value && location.assign(`?compare=${encodeURIComponent(chipId)},${encodeURIComponent(e.target.value)}`)}
    >
      <option value="">Compare with…</option>
      {others.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
    </select>
  );
}
