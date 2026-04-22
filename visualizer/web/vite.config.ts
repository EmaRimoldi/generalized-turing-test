import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Relative asset URLs so the built app works behind any HTTP prefix and with `python -m http.server` / file open.
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});

