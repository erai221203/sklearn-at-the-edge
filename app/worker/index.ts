/**
 * The Cloudflare Worker: a small JSON API in front of the exported models.
 *
 * Routes under /api are handled here; everything else is served from static
 * assets (see `run_worker_first` in wrangler.jsonc).
 */

import { Hono } from "hono";

import type { ApiError } from "../shared/types";
import {
  ArtifactError,
  isTaskId,
  loadAllMeta,
  loadMeta,
  loadTask,
  TASKS,
} from "./artifacts";
import { buildEvidence, buildVector, predictAll, ValidationError } from "./predict";

const app = new Hono<{ Bindings: Env }>();

/** Metadata is immutable per deploy; weights change only when models retrain. */
const CACHE_CONTROL = "public, max-age=300, stale-while-revalidate=3600";

app.get("/api/health", (c) =>
  c.json({ status: "ok", tasks: TASKS, time: new Date().toISOString() }),
);

app.get("/api/models", async (c) => {
  const metas = await loadAllMeta(c.env, c.req.url);
  c.header("Cache-Control", CACHE_CONTROL);
  return c.json({
    tasks: metas.map((meta) => ({
      task: meta.task,
      title: meta.title,
      subtitle: meta.subtitle,
      kind: meta.kind,
      classes: meta.classes,
      dataset: meta.dataset,
      generatedAt: meta.generatedAt,
      sklearnVersion: meta.sklearnVersion,
      models: meta.models.map((model) => ({
        id: model.id,
        name: model.name,
        description: model.description,
        metrics: model.metrics,
      })),
    })),
  });
});

app.get("/api/models/:task", async (c) => {
  const task = c.req.param("task");
  if (!isTaskId(task)) return c.json<ApiError>({ error: `unknown task "${task}"` }, 404);
  const meta = await loadMeta(c.env, c.req.url, task);
  c.header("Cache-Control", CACHE_CONTROL);
  return c.json(meta);
});

app.post("/api/predict/:task", async (c) => {
  const task = c.req.param("task");
  if (!isTaskId(task)) return c.json<ApiError>({ error: `unknown task "${task}"` }, 404);

  let body: unknown;
  try {
    body = await c.req.json();
  } catch {
    return c.json<ApiError>({ error: "request body must be valid JSON" }, 400);
  }

  const loaded = await loadTask(c.env, c.req.url, task);
  try {
    return c.json(predictAll(loaded, buildVector(loaded, body), buildEvidence(loaded, body)));
  } catch (error) {
    if (error instanceof ValidationError) {
      return c.json<ApiError>({ error: "invalid input", detail: error.message }, 400);
    }
    throw error;
  }
});

app.all("/api/*", (c) => c.json<ApiError>({ error: "not found" }, 404));

// Belt and braces: if a non-API request ever reaches the Worker, hand it to
// the static assets so the SPA still loads.
app.all("*", (c) => c.env.ASSETS.fetch(c.req.raw));

app.onError((error, c) => {
  if (error instanceof ArtifactError) {
    return c.json<ApiError>({ error: "model unavailable", detail: error.message }, 503);
  }
  console.error("unhandled worker error", error);
  return c.json<ApiError>({ error: "internal error" }, 500);
});

export default app;
