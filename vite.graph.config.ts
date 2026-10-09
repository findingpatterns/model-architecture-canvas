import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Data-driven model graph viewer (React Flow). Source in graph-src/, output web/graph/.
//
// SAFETY: outDir MUST stay exactly ../web/graph. With emptyOutDir:true, pointing it
// at ../web would wipe the catalog + viewer. scripts/assert-web-intact.mjs checks.
export default defineConfig({
  root: "graph-src",
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../web/graph",
    emptyOutDir: true,
  },
});
