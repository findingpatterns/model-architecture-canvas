# web/ — the static viewer

The deployed site. Overview, features, contributing, and deploy instructions live in the repo root:
**[`../README.md`](../README.md)** and **[`../CONTRIBUTING.md`](../CONTRIBUTING.md)**.

The **viewer** (`index.html` + `styles.css` + `app.js`) is plain static output with no bundler. At
runtime it fetches `catalog.json` (generated from `models/` by `scripts/build-catalog.mjs`) and renders
the sidebar from it.

The **editor** lives at `web/editor/` — the built output of the React/Vite app in `editor-src/` (see
`../README.md`). The viewer's topbar "✎ Edit" button links to `editor/?model=<id>`. The editor reads
the shared `localStorage["modelcanvas-theme"]` for light/dark and fetches `/canvases/<id>.canvas`.

`catalog.json`, `canvases/`, and `editor/` are all git-ignored build artifacts.

```bash
npm install && npm run serve   # build catalog + editor, then serve http://localhost:8000
```

## Viewer API note

Uses [`json-canvas-viewer`](https://github.com/hesprs/json-canvas-viewer)@4.3.2 from the unpkg CDN
(pinned). Verified API:

- `new JSONCanvasViewer({ container, canvas, parser }, [Minimap, Controls])` — option key is `parser`
  (not `markdownParser`); modules are the second argument.
- `viewer.load({ canvas })` swaps the diagram on the same instance; `viewer.dispose()` tears it down.
- `fetchCanvas(url)` loads a `.canvas` and returns parsed data.
