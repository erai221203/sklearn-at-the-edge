import { defineConfig } from "vitest/config";

// Kept separate from vite.config.ts: the inference modules are plain
// TypeScript, so the tests need neither the React nor the Cloudflare plugin.
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
