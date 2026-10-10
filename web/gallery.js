// Gallery (landing) rendering: model cards + a separate Hardware section.
// Entries come from catalog.json; hardware entries have kind: "hardware" and
// open the 3D explorer (hardware/?chip=<id>) instead of the 2D canvas viewer.

// Build an element with a class + text. textContent only (no innerHTML interpolation).
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

export const isHardware = (entry) => entry.kind === "hardware";

// Logo badge: image (path/URL), else a glyph/emoji, else the entry's initial.
function badge(entry) {
  const logo = entry.logo;
  const isImage = typeof logo === "string" && /^(https?:\/\/|logos\/|\/)|\.(svg|png|jpe?g|webp|gif)$/i.test(logo);
  if (isImage) {
    const img = el("img", "card-badge card-badge-img");
    img.src = logo;
    img.alt = `${entry.name} logo`;
    img.loading = "lazy";
    return img;
  }
  const b = el("span", "card-badge");
  b.textContent = logo && logo.trim() ? logo.trim() : entry.name.charAt(0).toUpperCase();
  return b;
}

function card(entry) {
  const hw = isHardware(entry);
  const a = el("a", "model-card");
  a.dataset.id = entry.id; // matched by the search/filter bar
  a.href = hw ? `hardware/?chip=${encodeURIComponent(entry.id)}` : `?model=${encodeURIComponent(entry.id)}`;

  const head = el("div", "card-head");
  head.appendChild(badge(entry));
  const titles = el("div", "card-titles");
  titles.appendChild(el("div", "card-name", entry.name));
  titles.appendChild(el("span", "card-id mono", hw ? `${entry.id} · ${entry.arch ?? "chip"}` : entry.id));
  head.appendChild(titles);
  if (hw) head.appendChild(el("span", "card-3d mono", "3D"));
  a.appendChild(head);

  a.appendChild(el("p", "card-note", entry.description));
  if (Array.isArray(entry.tags) && entry.tags.length) {
    const tags = el("div", "card-tags");
    for (const t of entry.tags) tags.appendChild(el("span", "tag", t));
    a.appendChild(tags);
  }
  const foot = el("div", "card-foot");
  if (entry.author?.name) foot.appendChild(el("span", "card-author mono", `by ${entry.author.name}`));
  foot.appendChild(el("span", "card-open mono", hw ? "Explore in 3D →" : "Open →"));
  a.appendChild(foot);
  return a;
}

export const CATEGORIES = ["models", "hardware"];

// Show one catalog section; the active tab is reflected in ?category= (models is the default).
export function selectCategory(category, els, { updateUrl = true } = {}) {
  const active = CATEGORIES.includes(category) ? category : "models";
  els.modelsSection.hidden = active !== "models";
  els.hardwareSection.hidden = active !== "hardware";
  for (const tab of els.categoryTabs) {
    const on = tab.dataset.category === active;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
  }
  if (updateUrl) {
    const url = new URL(location.href);
    if (active === "models") url.searchParams.delete("category");
    else url.searchParams.set("category", active);
    history.replaceState(null, "", url);
  }
  return active;
}

function lead(models, chips) {
  const parts = [];
  if (models.length === 1) parts.push(`the ${models[0].name} architecture`);
  else if (models.length > 1) parts.push(`${models.length} architectures — ${models.map((m) => m.name).join(", ")}`);
  if (chips.length) parts.push(`${chips.length === 1 ? "one chip" : `${chips.length} chips`} in 3D — ${chips.map((c) => c.name).join(", ")}`);
  return parts.length ? `Currently featuring ${parts.join(", plus ")}.` : "";
}

export function renderGallery(catalog, els) {
  const models = catalog.filter((e) => !isHardware(e));
  const chips = catalog.filter(isHardware);
  els.galleryLead.textContent = lead(models, chips);

  els.galleryGrid.replaceChildren();
  if (models.length === 0) {
    els.galleryGrid.appendChild(el("p", "gallery-empty", "No models yet — open a PR adding a folder under models/ to contribute one."));
  }
  for (const m of models) els.galleryGrid.appendChild(card(m));
  // One-click architecture comparison (the compare view has pickers for any pair).
  const comparable = models.filter((m) => Array.isArray(m.summary));
  els.modelCompareLink.hidden = comparable.length < 2;
  if (comparable.length >= 2) {
    els.modelCompareLink.href = `?compare=${encodeURIComponent(comparable[0].id)},${encodeURIComponent(comparable[1].id)}`;
    els.modelCompareLink.textContent = `Compare ${comparable[0].name} vs ${comparable[1].name} →`;
  }

  els.categoryModelsCount.textContent = String(models.length);
  els.categoryHardwareCount.textContent = String(chips.length);
  els.hardwareGrid.replaceChildren();
  if (chips.length === 0) els.hardwareGrid.appendChild(el("p", "gallery-empty", "No chips yet — open a PR adding a folder under hardware/ to contribute one."));
  for (const c of chips) els.hardwareGrid.appendChild(card(c));
  // One-click comparison of the first two chips (the 3D view lets you pick any pair).
  els.compareLink.hidden = chips.length < 2;
  if (chips.length >= 2) {
    els.compareLink.href = `hardware/?compare=${encodeURIComponent(chips[0].id)},${encodeURIComponent(chips[1].id)}`;
    els.compareLink.textContent = `Compare ${chips[0].name} vs ${chips[1].name} →`;
  }
}
