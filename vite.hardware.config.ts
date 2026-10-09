import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The 3D hardware explorer is a separate React app (three.js lives only in this
// bundle, so model pages stay light). Source in hardware-src/, output web/hardware/.
//
// SAFETY: outDir MUST stay exactly ../web/hardware. With emptyOutDir:true, pointing
// it at ../web would wipe the catalog + viewer. scripts/assert-web-intact.mjs checks.
export default defineConfig({
  root: "hardware-src",
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../web/hardware",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
});
