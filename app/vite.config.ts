import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  // Runs the real workerd runtime in dev, so `npm run dev` exercises the same
  // code path as production instead of a Node stand-in.
  plugins: [react(), cloudflare()],
  build: {
    sourcemap: true,
  },
});
