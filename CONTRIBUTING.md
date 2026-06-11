# Contributing a model diagram

ModelCanvas is a community catalog of model-architecture diagrams. To add one, you open a Pull
Request with a single new folder — you never touch any web or JavaScript code.

## Quick start

1. **Fork** this repo and create a branch.
2. **Copy the template:** `models/_template/` → `models/<your-model-id>/`
   - `<your-model-id>` must be **kebab-case** (lowercase letters, digits, hyphens), e.g. `llama-3-70b-attention`. It becomes the deep-link `?model=<id>` and the downloaded filename, and must be unique.
3. **Add your diagram:** replace `model.canvas` with your [JSON Canvas 1.0](https://jsoncanvas.org/) file. You can export one from a canvas-capable note editor (for example [Obsidian](https://obsidian.md/)) or generate it however you like. Exactly **one** `.canvas` file per folder.
4. **Fill in `meta.json`** (see fields below) and delete the template's `README.md` from your copy.
5. **Open a PR.** CI validates it automatically (see "Validation"). A maintainer reviews the diagram (Vercel posts a live preview link on the PR) and merges. The site redeploys on merge.

## `meta.json` fields

| Field | Required | Type | Notes |
|-------|----------|------|-------|
| `name` | ✅ | string | Human-readable model name shown as the card heading. |
| `description` | ✅ | string | 1–2 sentences: what the diagram shows / how to read it (e.g. what colors mean). |
| `author` | optional | object | `{ "name": "...", "github": "handle" }` — credited in the sidebar. |
| `tags` | optional | string[] | e.g. `["llama", "attention", "moe"]` — for grouping/filtering. |
| `source` | optional | string | `http(s)` URL to the paper, repo, or model card. Shown as a `↗ source` link. |

Example:

```json
{
  "name": "DeepSeek-V4-Flash",
  "description": "Architecture colored by weight dtype — FP4 routed experts, FP8 default linears, BF16 overrides.",
  "author": { "name": "Jane Doe", "github": "janedoe" },
  "tags": ["deepseek", "moe", "quantization"],
  "source": "https://github.com/deepseek-ai"
}
```

## Validation

Every PR runs `npm run validate` (GitHub Action). It fails — with a red ✗ on your PR — if:

- the folder name isn't kebab-case or isn't unique,
- `meta.json` is missing/invalid or lacks `name`/`description`,
- `tags`/`source`/`author` have the wrong shape (`source` must be an `http(s)` URL),
- there isn't exactly one `.canvas` file, or it isn't valid JSON Canvas (no `nodes` array).

Run it locally before pushing:

```bash
npm install           # first time only (installs the editor's build deps)
npm run validate      # validate submissions only, writes nothing — no deps needed
npm run build         # validate + generate catalog + build the editor
npm run serve         # build, then serve the site at http://localhost:8000
```

`npm run validate` (the PR gate for submissions) stays dependency-free. `npm run build`/`serve` also
build the in-browser editor (a React/Vite app under `editor-src/`), which needs `npm install` first.
Adding a model never requires touching the editor.

## How it deploys

`models/` is the source of truth. `scripts/build-catalog.mjs` scans it, validates each model, and emits
`web/catalog.json` plus `web/canvases/<id>.canvas`. Those generated files are **git-ignored** — they're
built fresh by Vercel on every deploy, which is why adding a model never causes merge conflicts. The
deployed site is fully static (HTML + CSS + JS + CDN viewer + JSON), no backend, no secrets.
