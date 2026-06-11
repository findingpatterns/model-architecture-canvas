# web/ — the static viewer

The deployed site. Overview, features, contributing, and deploy instructions live in the repo root:
**[`../README.md`](../README.md)** and **[`../CONTRIBUTING.md`](../CONTRIBUTING.md)**.

This folder is plain static output — `index.html` + `styles.css` + `app.js`, no build tooling. At runtime
it fetches `catalog.json` (generated from `models/` by `scripts/build-catalog.mjs`) and renders the
sidebar from it. `catalog.json` and `canvases/` are git-ignored build artifacts.

```bash
npm run serve   # from repo root: build the catalog, then serve http://localhost:8000
```

## Viewer API note

Uses [`json-canvas-viewer`](https://github.com/hesprs/json-canvas-viewer)@4.3.2 from the unpkg CDN
(pinned). Verified API:

- `new JSONCanvasViewer({ container, canvas, parser }, [Minimap, Controls])` — option key is `parser`
  (not `markdownParser`); modules are the second argument.
- `viewer.load({ canvas })` swaps the diagram on the same instance; `viewer.dispose()` tears it down.
- `fetchCanvas(url)` loads a `.canvas` and returns parsed data.
