import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  // Runs the real workerd runtime in dev, so `npm run dev` exercises the same
  // code path as production instead of a Node stand-in.
  plugins: [react(), cloudflare()],
  build: {
    // The client source map is ~970 KB of publicly served asset that only a
    // reader with devtools open would ever fetch, and the source is on GitHub
    // anyway. Set to true locally if you need to debug a production bundle.
    sourcemap: false,
  },
});
