/**
 * Loads model artifacts from the deployed static assets.
 *
 * Weights are the expensive half of a bundle and most requests never need
 * them, so metadata and weights are separate files and each is fetched at most
 * once per isolate. In-flight promises are cached rather than results, so
 * concurrent first requests share a single fetch instead of racing.
 */

import type { TaskMeta, TabularInputSpec, WeightsBundle } from "../shared/types";
import { TabularEncoder, TfidfVectorizer } from "./ml/features";
import { loadModel, type Model } from "./ml/models";

export const TASKS = ["churn", "spam", "movie-genre"] as const;
export type TaskId = (typeof TASKS)[number];

export function isTaskId(value: string): value is TaskId {
  return (TASKS as readonly string[]).includes(value);
}

export interface LoadedTask {
  meta: TaskMeta;
  models: Map<string, Model>;
  vectorizer: TfidfVectorizer | null;
  encoder: TabularEncoder | null;
}

export class ArtifactError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

const metaCache = new Map<string, Promise<TaskMeta>>();
const taskCache = new Map<string, Promise<LoadedTask>>();

async function fetchJson<T>(env: Env, base: string, path: string): Promise<T> {
  const response = await env.ASSETS.fetch(new URL(path, base));
  if (!response.ok) {
    throw new ArtifactError(`asset ${path} is missing (${response.status})`, 500);
  }
  return (await response.json()) as T;
}

export function loadMeta(env: Env, base: string, task: TaskId): Promise<TaskMeta> {
  let pending = metaCache.get(task);
  if (!pending) {
    pending = fetchJson<TaskMeta>(env, base, `/models/${task}.meta.json`).catch((error) => {
      metaCache.delete(task); // never cache a failure
      throw error;
    });
    metaCache.set(task, pending);
  }
  return pending;
}

export function loadAllMeta(env: Env, base: string): Promise<TaskMeta[]> {
  return Promise.all(TASKS.map((task) => loadMeta(env, base, task)));
}

async function buildTask(env: Env, base: string, task: TaskId): Promise<LoadedTask> {
  const [meta, weights] = await Promise.all([
    loadMeta(env, base, task),
    fetchJson<WeightsBundle>(env, base, `/models/${task}.weights.json`),
  ]);

  const models = new Map<string, Model>();
  for (const id of meta.modelIds) {
    const artifact = weights.models[id];
    if (!artifact) throw new ArtifactError(`weights for model "${id}" are missing`, 500);
    models.set(id, loadModel(artifact));
  }

  return {
    meta,
    models,
    vectorizer: weights.vectorizer ? new TfidfVectorizer(weights.vectorizer) : null,
    encoder:
      meta.input.kind === "tabular" ? new TabularEncoder(meta.input as TabularInputSpec) : null,
  };
}

export function loadTask(env: Env, base: string, task: TaskId): Promise<LoadedTask> {
  let pending = taskCache.get(task);
  if (!pending) {
    pending = buildTask(env, base, task).catch((error) => {
      taskCache.delete(task);
      throw error;
    });
    taskCache.set(task, pending);
  }
  return pending;
}
