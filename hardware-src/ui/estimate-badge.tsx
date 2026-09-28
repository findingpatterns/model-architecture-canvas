// "estimate" badge with an explanation. When a value also has a source, the
// cited page describes a closely related part (e.g. GH100 / H100 for H200) or
// a total that was divided — the badge says so instead of looking contradictory.
export function EstimateBadge({ sourced }: { sourced: boolean }) {
  const why = sourced
    ? "Derived, not stated verbatim: the cited page gives a related figure (e.g. for the same die in another product, or a system total). The value text states the basis."
    : "No official figure found; this is an informed estimate.";
  return (
    // A real button: focusable, announced, and shows the explanation on focus (CSS).
    <button type="button" className="badge" title={why} data-tip={why} aria-label={`Estimate. ${why}`} onClick={(e) => e.stopPropagation()}>
      estimate{sourced ? " ⓘ" : ""}
    </button>
  );
}
