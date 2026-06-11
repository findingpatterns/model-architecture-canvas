# Model submission template

Copy this folder to add a model. Steps:

1. Copy `models/_template/` → `models/<your-model-id>/`
   - `<your-model-id>` must be **kebab-case** (lowercase, digits, hyphens) and unique. It becomes the deep-link `?model=<id>` and the download filename.
2. Replace `model.canvas` with your JSON Canvas 1.0 diagram (export from a note editor like Obsidian, or generate it).
3. Edit `meta.json` (see field reference in the repo root `CONTRIBUTING.md`).
4. Delete this `README.md` from your copy.
5. Open a Pull Request. CI validates your submission automatically; a maintainer merges it; the site redeploys.

You never edit any web or JavaScript file — the catalog is generated from this folder.
