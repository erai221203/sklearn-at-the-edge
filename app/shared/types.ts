/**
 * The contract between the Python trainers, the Worker and the browser.
 *
 * `ml/artifact.py` writes JSON in these shapes, `worker/ml/` reads them, and
 * `src/` renders them. Changing a shape means changing all three.
 */

/** A large numeric array, base64-encoded rather than written as JSON numbers. */
export interface PackedArray {
  __pack: "f32" | "f64" | "i32";
  shape: number[];
  b64: string;
}

export type FloatArrayJson = number[] | PackedArray;
export type IntArrayJson = number[] | PackedArray;
export type MatrixJson = number[][] | PackedArray;

// ---------------------------------------------------------------------------
// Model artifacts (weights)
// ---------------------------------------------------------------------------

export type LinearOutput = "sigmoid" | "softmax" | "margin" | "argmax";

export interface LinearArtifact {
  type: "linear";
  output: LinearOutput;
  coef: MatrixJson;
  intercept: number[];
}

export interface NaiveBayesArtifact {
  type: "multinomial_nb";
  classLogPrior: number[];
  featureLogProb: MatrixJson;
}

interface TreeTable {
  nOutputs: number;
  nNodes: number;
  roots: IntArrayJson;
  feature: IntArrayJson;
  threshold: FloatArrayJson;
  left: IntArrayJson;
  right: IntArrayJson;
  leafValue: FloatArrayJson;
  /** scikit-learn narrows features to float32 before walking a tree. */
  featureDtype: "f32";
}

export interface ForestArtifact extends TreeTable {
  type: "forest";
}

export interface GbdtArtifact extends TreeTable {
  type: "gbdt";
  output: "sigmoid";
  init: number;
}

export type ModelArtifact =
  | LinearArtifact
  | NaiveBayesArtifact
  | ForestArtifact
  | GbdtArtifact;

export interface VectorizerSpec {
  vocabulary: Record<string, number>;
  /** Everyday words removed during training, kept to explain why. */
  stopWords: string[];
  idf: FloatArrayJson;
  analyzer: {
    lowercase: boolean;
    tokenPattern: string;
    sublinearTf: boolean;
    norm: string;
    smoothIdf: boolean;
  };
}

export interface WeightsBundle {
  task: string;
  schemaVersion: number;
  models: Record<string, ModelArtifact>;
  vectorizer?: VectorizerSpec;
}

// ---------------------------------------------------------------------------
// Task metadata
// ---------------------------------------------------------------------------

export interface NumberField {
  name: string;
  label: string;
  help?: string;
  type: "number";
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface BooleanField {
  name: string;
  label: string;
  help?: string;
  type: "boolean";
  default: number;
}

export interface CategoryField {
  name: string;
  label: string;
  help?: string;
  type: "category";
  options: string[];
  default: string;
}

export type FieldSpec = NumberField | BooleanField | CategoryField;

export interface EncodeColumn {
  from: string;
  /** Present for one-hot indicators: emit 1 when the field equals this value. */
  equals?: string;
}

export interface TabularInputSpec {
  kind: "tabular";
  fields: FieldSpec[];
  encode: EncodeColumn[];
  featureLabels: string[];
  scaler?: { mean: number[]; scale: number[] };
}

export interface TextInputSpec {
  kind: "text";
  label: string;
  placeholder: string;
  samples: string[];
}

export type InputSpec = TabularInputSpec | TextInputSpec;

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  f1Weighted?: number;
  rocAuc?: number;
}

export interface TokenWeight {
  term: string;
  weight: number;
}

export interface ModelMeta {
  id: string;
  name: string;
  description: string;
  metrics: Metrics;
  confusion: number[][];
  rocCurve?: { fpr: number; tpr: number }[];
  thresholdSweep?: { threshold: number; precision: number; recall: number; f1: number }[];
  perClass?: { genre: string; precision: number; recall: number; f1: number; support: number }[];
  topTokens?: { spam: TokenWeight[]; ham: TokenWeight[] };
  topTermsPerClass?: { genre: string; terms: TokenWeight[] }[];
  explain: { kind: "coefficients" | "importances" | "tokenWeights"; values?: number[] };
}

export interface DatasetMeta {
  name: string;
  rows: number;
  trainRows: number;
  testRows: number;
  features: number;
  classBalance: Record<string, number>;
}

export interface TaskMeta {
  task: string;
  title: string;
  subtitle: string;
  schemaVersion: number;
  generatedAt: string;
  sklearnVersion: string;
  kind: "binary" | "multiclass";
  classes: string[];
  positiveClass: number | null;
  dataset: DatasetMeta;
  input: InputSpec;
  models: ModelMeta[];
  modelIds: string[];
}

// ---------------------------------------------------------------------------
// API payloads
// ---------------------------------------------------------------------------

export type PredictRequest =
  | { text: string }
  | Record<string, string | number>;

export interface Contribution {
  label: string;
  value: number;
}

export interface ModelPrediction {
  modelId: string;
  modelName: string;
  predictedIndex: number;
  predictedLabel: string;
  /** Null for models with no calibrated probability, such as an uncalibrated SVM. */
  probabilities: number[] | null;
  confidence: number | null;
  scores: number[];
  topClasses: { label: string; probability: number | null; score: number }[];
  contributions: Contribution[];
}

/**
 * How much of the input the models could actually use.
 *
 * A bag-of-words model given nothing it recognises does not abstain — it
 * falls back to whichever class was commonest in training and reports that
 * with a perfectly confident-looking number. Passing the counts to the client
 * lets the answer say so instead of pretending to have read something.
 */
export interface Evidence {
  /** Words in the models' vocabulary; the only ones that affect the answer. */
  recognised: string[];
  /** Everyday words ("this", "from") removed on purpose during training. */
  ignored: string[];
  /** Words the models have never seen — names, typos, rare terms. */
  unknown: string[];
  total: number;
}

export interface PredictResponse {
  task: string;
  classes: string[];
  models: ModelPrediction[];
  consensus: { label: string; agreeing: number; total: number };
  /** Text tasks only. */
  evidence?: Evidence;
  elapsedMs: number;
}

export interface ApiError {
  error: string;
  detail?: string;
}
