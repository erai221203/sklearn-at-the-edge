/**
 * Inference for the four estimator families the trainers export.
 *
 * Every function here mirrors `ml/reference.py`, which in turn is checked
 * against scikit-learn itself inside each trainer. The parity suite in
 * `tests/parity.test.ts` replays scikit-learn's own answers through this code,
 * so a divergence fails the build rather than reaching a user.
 */

import type {
  ForestArtifact,
  GbdtArtifact,
  LinearArtifact,
  ModelArtifact,
  NaiveBayesArtifact,
} from "../../shared/types";
import { readFloats, readInts, readMatrix, type FloatArray, type Matrix } from "./pack";
import {
  argmax,
  dotRow,
  sigmoid,
  softmax,
  toDenseArray,
  type FeatureVector,
} from "./vector";

export interface PredictionResult {
  predictedIndex: number;
  scores: number[];
  probabilities: number[] | null;
}

export interface Model {
  readonly nClasses: number;
  predict(vector: FeatureVector): PredictionResult;
  /**
   * Per-feature contribution to the predicted class, where the model has a
   * meaningful one. Linear families do; tree ensembles do not, and return null.
   */
  contributions(vector: FeatureVector, classIndex: number): Float64Array | null;
}

class LinearModel implements Model {
  private readonly coef: Matrix;
  private readonly intercept: number[];
  private readonly output: LinearArtifact["output"];

  constructor(artifact: LinearArtifact) {
    this.coef = readMatrix(artifact.coef);
    this.intercept = artifact.intercept;
    this.output = artifact.output;
  }

  get nClasses(): number {
    // A binary sigmoid or margin model stores one row of coefficients but
    // still answers with two classes.
    return this.output === "sigmoid" || this.output === "margin" ? 2 : this.coef.rows;
  }

  predict(vector: FeatureVector): PredictionResult {
    const scores: number[] = [];
    for (let row = 0; row < this.coef.rows; row += 1) {
      scores.push(dotRow(this.coef.data, row, this.coef.cols, vector) + this.intercept[row]);
    }

    switch (this.output) {
      case "sigmoid": {
        const probability = sigmoid(scores[0]);
        return {
          predictedIndex: probability >= 0.5 ? 1 : 0,
          scores,
          probabilities: [1 - probability, probability],
        };
      }
      case "margin":
        return { predictedIndex: scores[0] > 0 ? 1 : 0, scores, probabilities: null };
      case "softmax": {
        const probabilities = softmax(scores);
        return { predictedIndex: argmax(probabilities), scores, probabilities };
      }
      case "argmax":
        return { predictedIndex: argmax(scores), scores, probabilities: null };
    }
  }

  contributions(vector: FeatureVector, classIndex: number): Float64Array {
    // One row of coefficients means the single score always describes class 1,
    // so class 0's contributions are its mirror image.
    const single = this.coef.rows === 1;
    const row = single ? 0 : classIndex;
    const sign = single && classIndex === 0 ? -1 : 1;
    const offset = row * this.coef.cols;
    const out = new Float64Array(vector.length);
    const { indices, values } = vector;
    for (let i = 0; i < values.length; i += 1) {
      const position = indices === null ? i : indices[i];
      out[position] = sign * this.coef.data[offset + position] * values[i];
    }
    return out;
  }
}

class NaiveBayesModel implements Model {
  private readonly featureLogProb: Matrix;
  private readonly classLogPrior: number[];

  constructor(artifact: NaiveBayesArtifact) {
    this.featureLogProb = readMatrix(artifact.featureLogProb);
    this.classLogPrior = artifact.classLogPrior;
  }

  get nClasses(): number {
    return this.featureLogProb.rows;
  }

  predict(vector: FeatureVector): PredictionResult {
    const scores: number[] = [];
    for (let row = 0; row < this.featureLogProb.rows; row += 1) {
      scores.push(
        dotRow(this.featureLogProb.data, row, this.featureLogProb.cols, vector) +
          this.classLogPrior[row],
      );
    }
    return { predictedIndex: argmax(scores), scores, probabilities: softmax(scores) };
  }

  contributions(vector: FeatureVector, classIndex: number): Float64Array {
    // How much this term pushed towards the chosen class rather than the most
    // plausible alternative -- a raw log-probability is always negative and
    // would rank every term the same way.
    const rival = this.rivalClass(vector, classIndex);
    const cols = this.featureLogProb.cols;
    const chosen = classIndex * cols;
    const other = rival * cols;
    const out = new Float64Array(vector.length);
    const { indices, values } = vector;
    for (let i = 0; i < values.length; i += 1) {
      const position = indices === null ? i : indices[i];
      out[position] =
        (this.featureLogProb.data[chosen + position] - this.featureLogProb.data[other + position]) *
        values[i];
    }
    return out;
  }

  private rivalClass(vector: FeatureVector, classIndex: number): number {
    const { scores } = this.predict(vector);
    let best = -1;
    for (let i = 0; i < scores.length; i += 1) {
      if (i === classIndex) continue;
      if (best === -1 || scores[i] > scores[best]) best = i;
    }
    return best === -1 ? classIndex : best;
  }
}

/** Shared node-table walk for both tree ensembles. */
class TreeTable {
  readonly roots: Int32Array;
  private readonly feature: Int32Array;
  private readonly threshold: FloatArray;
  private readonly left: Int32Array;
  private readonly right: Int32Array;
  readonly leafValue: FloatArray;
  readonly nOutputs: number;

  constructor(artifact: ForestArtifact | GbdtArtifact) {
    this.roots = readInts(artifact.roots);
    this.feature = readInts(artifact.feature);
    this.threshold = readFloats(artifact.threshold);
    this.left = readInts(artifact.left);
    this.right = readInts(artifact.right);
    this.leafValue = readFloats(artifact.leafValue);
    this.nOutputs = artifact.nOutputs;
  }

  /**
   * scikit-learn narrows X to float32 before traversal, and its split points
   * are midpoints of adjacent float32 values -- some sit only ~2e-9 from real
   * data. Comparing in float64 sends those samples down the wrong branch.
   */
  narrow(vector: FeatureVector): Float32Array {
    return Float32Array.from(toDenseArray(vector));
  }

  /** Returns the leaf's slot in `leafValue`, not a node index. */
  walk(root: number, x: Float32Array): number {
    let node = root;
    while (this.feature[node] !== -1) {
      node = x[this.feature[node]] <= this.threshold[node] ? this.left[node] : this.right[node];
    }
    return this.left[node];
  }
}

class ForestModel implements Model {
  private readonly trees: TreeTable;

  constructor(artifact: ForestArtifact) {
    this.trees = new TreeTable(artifact);
  }

  get nClasses(): number {
    return this.trees.nOutputs;
  }

  predict(vector: FeatureVector): PredictionResult {
    const { nOutputs, roots, leafValue } = this.trees;
    const x = this.trees.narrow(vector);
    const totals = new Array<number>(nOutputs).fill(0);
    for (let t = 0; t < roots.length; t += 1) {
      const slot = this.trees.walk(roots[t], x) * nOutputs;
      for (let c = 0; c < nOutputs; c += 1) totals[c] += leafValue[slot + c];
    }
    for (let c = 0; c < nOutputs; c += 1) totals[c] /= roots.length;
    return { predictedIndex: argmax(totals), scores: totals, probabilities: totals };
  }

  contributions(): null {
    return null;
  }
}

class GbdtModel implements Model {
  private readonly trees: TreeTable;
  private readonly init: number;

  constructor(artifact: GbdtArtifact) {
    this.trees = new TreeTable(artifact);
    this.init = artifact.init;
  }

  get nClasses(): number {
    return 2;
  }

  predict(vector: FeatureVector): PredictionResult {
    const x = this.trees.narrow(vector);
    let raw = this.init;
    for (let t = 0; t < this.trees.roots.length; t += 1) {
      raw += this.trees.leafValue[this.trees.walk(this.trees.roots[t], x)];
    }
    const probability = sigmoid(raw);
    return {
      predictedIndex: probability >= 0.5 ? 1 : 0,
      scores: [raw],
      probabilities: [1 - probability, probability],
    };
  }

  contributions(): null {
    return null;
  }
}

export function loadModel(artifact: ModelArtifact): Model {
  switch (artifact.type) {
    case "linear":
      return new LinearModel(artifact);
    case "multinomial_nb":
      return new NaiveBayesModel(artifact);
    case "forest":
      return new ForestModel(artifact);
    case "gbdt":
      return new GbdtModel(artifact);
    default: {
      const unknown = artifact as { type: string };
      throw new Error(`unknown artifact type: ${unknown.type}`);
    }
  }
}
