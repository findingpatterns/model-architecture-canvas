import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The editor is a separate React app whose source lives in editor-src/ and which
// builds into web/editor/ — alongside (never overwriting) the vanilla static site.
// base: "./" keeps asset paths relative so subpath hosting works.
//
// SAFETY: outDir MUST stay exactly ../web/editor. With emptyOutDir:true, pointing it
// at ../web would wipe the catalog + viewer. scripts/assert-web-intact.mjs verifies
// web/index.html and web/canvases/ survived the build.
export default defineConfig({
  root: "editor-src",
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../web/editor",
    emptyOutDir: true,
  },
});
