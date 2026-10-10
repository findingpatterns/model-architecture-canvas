// Search box + facet chips above the gallery grid. Shows/hides the already-rendered
// cards (matched by data-id) instead of re-rendering them; state syncs to the URL.
import { el, isHardware } from "./gallery.js";
import { FACETS, facetCounts, filterEntries, hasActiveFilters, readFilters, writeFilters } from "./gallery-filters.js";

export function createFilterBar({ root, catalog, grids }) {
  const entries = {
    models: catalog.filter((e) => !isHardware(e)),
    hardware: catalog.filter(isHardware),
  };
  let category = "models";
  let state = { q: "" };

  const search = el("input", "filter-search");
  search.type = "search";
  search.setAttribute("aria-label", "Search the gallery");
  const facetsBox = el("div", "filter-facets");
  const status = el("span", "filter-status mono");
  const clear = el("button", "filter-clear mono", "Clear filters");
  clear.type = "button";
  const meta = el("div", "filter-meta");
  meta.append(status, clear);
  root.replaceChildren(search, facetsBox, meta);

  const empty = {};
  for (const [key, grid] of Object.entries(grids)) {
    empty[key] = el("p", "gallery-empty", "No match — try a different search or clear the filters.");
    empty[key].hidden = true;
    grid.after(empty[key]);
  }

  function syncUrl() {
    const url = new URL(location.href);
    for (const c of Object.keys(FACETS)) writeFilters(c, {}, url.searchParams); // drop the other category's facets
    writeFilters(category, state, url.searchParams);
    history.replaceState(null, "", url);
  }

  function apply() {
    const list = entries[category];
    const shown = new Set(filterEntries(list, category, state).map((e) => e.id));
    for (const card of grids[category].querySelectorAll(".model-card")) card.hidden = !shown.has(card.dataset.id);
    empty[category].hidden = shown.size > 0 || list.length === 0;
    const noun = category === "models" ? "models" : "chips";
    status.textContent = `${shown.size} of ${list.length} ${noun}`;
    clear.hidden = !hasActiveFilters(category, state);
    renderFacets();
  }

  function renderFacets() {
    const counts = facetCounts(entries[category], category, state);
    facetsBox.replaceChildren();
    for (const f of FACETS[category]) {
      const group = el("div", "filter-group");
      group.setAttribute("role", "group");
      group.setAttribute("aria-label", f.label);
      group.appendChild(el("span", "filter-label mono", f.label));
      for (const o of f.options) {
        const n = counts[f.param][o.value];
        const on = state[f.param] === o.value;
        const chip = el("button", `filter-chip${on ? " is-active" : ""}`);
        chip.type = "button";
        chip.setAttribute("aria-pressed", String(on));
        chip.disabled = !n && !on; // nothing would match with this option
        chip.append(o.label, el("span", "filter-count mono", String(n)));
        chip.addEventListener("click", () => {
          state = { ...state, [f.param]: on ? "" : o.value };
          syncUrl();
          apply();
        });
        group.appendChild(chip);
      }
      facetsBox.appendChild(group);
    }
  }

  search.addEventListener("input", () => {
    state = { ...state, q: search.value.trim() };
    syncUrl();
    apply();
  });
  clear.addEventListener("click", () => {
    state = { q: "" };
    search.value = "";
    syncUrl();
    apply();
  });

  return {
    // Switch the bar to a category; filters start fresh (a model query rarely matches a chip).
    show(next, { fromUrl = false } = {}) {
      category = next;
      const params = new URLSearchParams(location.search);
      state = readFilters(category, fromUrl ? params : new URLSearchParams());
      search.value = state.q;
      search.placeholder = category === "models" ? "Search models — name, attention, tag…" : "Search chips — name, vendor, architecture…";
      if (!fromUrl) syncUrl();
      apply();
    },
  };
}
