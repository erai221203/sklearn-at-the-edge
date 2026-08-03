import type { ApiError, Metrics, PredictResponse, TaskMeta } from "../shared/types";

export interface CatalogTask {
  task: string;
  title: string;
  subtitle: string;
  kind: "binary" | "multiclass";
  classes: string[];
  dataset: TaskMeta["dataset"];
  generatedAt: string;
  sklearnVersion: string;
  models: { id: string; name: string; description: string; metrics: Metrics }[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as ApiError;
      message = body.detail ?? body.error ?? message;
    } catch {
      // Keep the status line when the body is not the JSON error shape.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const fetchCatalog = (): Promise<{ tasks: CatalogTask[] }> => request("/api/models");

export const fetchTask = (task: string): Promise<TaskMeta> => request(`/api/models/${task}`);

export const predict = (task: string, body: unknown): Promise<PredictResponse> =>
  request(`/api/predict/${task}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
