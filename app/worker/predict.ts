/**
 * Request validation and the shape of a prediction response.
 */

import type {
  Contribution,
  Evidence,
  FieldSpec,
  ModelPrediction,
  PredictResponse,
  TabularInputSpec,
} from "../shared/types";
import type { LoadedTask } from "./artifacts";
import type { FeatureVector } from "./ml/vector";

const MAX_TEXT_LENGTH = 20_000;
const MAX_CONTRIBUTIONS = 8;
const MAX_TOP_CLASSES = 5;

export class ValidationError extends Error {}

function requireRecord(body: unknown): Record<string, unknown> {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    throw new ValidationError("request body must be a JSON object");
  }
  return body as Record<string, unknown>;
}

function validateText(body: unknown): string {
  const text = requireRecord(body).text;
  if (typeof text !== "string") throw new ValidationError('"text" must be a string');
  const trimmed = text.trim();
  if (trimmed.length === 0) throw new ValidationError('"text" must not be empty');
  if (text.length > MAX_TEXT_LENGTH) {
    throw new ValidationError(`"text" must be at most ${MAX_TEXT_LENGTH} characters`);
  }
  return text;
}

function validateField(field: FieldSpec, raw: unknown): string | number {
  if (raw === undefined || raw === null) {
    throw new ValidationError(`"${field.name}" is required`);
  }
  switch (field.type) {
    case "number": {
      const value = typeof raw === "string" ? Number(raw.trim()) : raw;
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ValidationError(`"${field.name}" must be a finite number`);
      }
      // Values outside the training range are allowed -- extrapolating is a
      // legitimate thing to ask a model -- but absurd magnitudes are not.
      if (Math.abs(value) > 1e12) {
        throw new ValidationError(`"${field.name}" is out of range`);
      }
      return value;
    }
    case "boolean": {
      if (typeof raw === "boolean") return raw ? 1 : 0;
      const value = Number(raw);
      if (value !== 0 && value !== 1) {
        throw new ValidationError(`"${field.name}" must be 0 or 1`);
      }
      return value;
    }
    case "category": {
      const value = String(raw);
      if (!field.options.includes(value)) {
        throw new ValidationError(
          `"${field.name}" must be one of: ${field.options.join(", ")}`,
        );
      }
      return value;
    }
  }
}

function validateTabular(spec: TabularInputSpec, body: unknown): Record<string, string | number> {
  const record = requireRecord(body);
  const validated: Record<string, string | number> = {};
  for (const field of spec.fields) {
    validated[field.name] = validateField(field, record[field.name]);
  }
  return validated;
}

/** Turns a validated request body into the vector the models consume. */
export function buildVector(task: LoadedTask, body: unknown): FeatureVector {
  if (task.meta.input.kind === "text") {
    if (!task.vectorizer) throw new Error("text task is missing its vectorizer");
    return task.vectorizer.transform(validateText(body));
  }
  if (!task.encoder) throw new Error("tabular task is missing its encoder");
  return task.encoder.encode(validateTabular(task.meta.input, body));
}

/** How much of a text input the models could actually use. */
export function buildEvidence(task: LoadedTask, body: unknown): Evidence | undefined {
  if (task.meta.input.kind !== "text" || !task.vectorizer) return undefined;
  return task.vectorizer.analyse(validateText(body));
}

function labelFor(task: LoadedTask, position: number): string {
  if (task.meta.input.kind === "text") {
    return task.vectorizer?.terms[position] ?? `#${position}`;
  }
  return (task.meta.input as TabularInputSpec).featureLabels[position] ?? `#${position}`;
}

function topContributions(
  task: LoadedTask,
  vector: FeatureVector,
  raw: Float64Array | null,
): Contribution[] {
  if (!raw) return [];
  const entries: Contribution[] = [];
  const { indices, values } = vector;
  const count = indices === null ? values.length : indices.length;
  for (let i = 0; i < count; i += 1) {
    const position = indices === null ? i : indices[i];
    const value = raw[position];
    if (value === 0) continue;
    entries.push({ label: labelFor(task, position), value: Number(value.toFixed(6)) });
  }
  entries.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  return entries.slice(0, MAX_CONTRIBUTIONS);
}

export function predictAll(
  task: LoadedTask,
  vector: FeatureVector,
  evidence?: Evidence,
): PredictResponse {
  const started = Date.now();
  const { classes } = task.meta;
  const predictions: ModelPrediction[] = [];

  for (const meta of task.meta.models) {
    const model = task.models.get(meta.id);
    if (!model) continue;
    const result = model.predict(vector);

    const ranked = classes
      .map((label, index) => ({
        label,
        probability: result.probabilities ? result.probabilities[index] : null,
        // A binary model reports one score describing class 1; mirror it so
        // both rows carry a comparable number.
        score: result.scores.length === classes.length ? result.scores[index]
          : index === 1 ? result.scores[0] : -result.scores[0],
      }))
      .sort((a, b) => (b.probability ?? b.score) - (a.probability ?? a.score))
      .slice(0, task.meta.kind === "binary" ? classes.length : MAX_TOP_CLASSES);

    predictions.push({
      modelId: meta.id,
      modelName: meta.name,
      predictedIndex: result.predictedIndex,
      predictedLabel: classes[result.predictedIndex],
      probabilities: result.probabilities,
      confidence: result.probabilities ? result.probabilities[result.predictedIndex] : null,
      scores: result.scores,
      topClasses: ranked,
      // Binary tasks always explain in terms of the positive class, so a
      // positive contribution means the same thing ("more likely to churn",
      // "more spam-like") no matter which way the model actually voted.
      // Explaining relative to the predicted class instead would silently
      // invert every sign whenever the answer came back negative.
      contributions: topContributions(
        task,
        vector,
        model.contributions(vector, task.meta.kind === "binary" ? 1 : result.predictedIndex),
      ),
    });
  }

  const tally = new Map<string, number>();
  for (const prediction of predictions) {
    tally.set(prediction.predictedLabel, (tally.get(prediction.predictedLabel) ?? 0) + 1);
  }
  let winner = predictions[0]?.predictedLabel ?? "";
  let agreeing = 0;
  for (const [label, count] of tally) {
    if (count > agreeing) {
      winner = label;
      agreeing = count;
    }
  }

  return {
    task: task.meta.task,
    classes,
    models: predictions,
    consensus: { label: winner, agreeing, total: predictions.length },
    evidence,
    elapsedMs: Date.now() - started,
  };
}
