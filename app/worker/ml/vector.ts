/**
 * Feature vectors, in the two shapes this project produces.
 *
 * Tabular tasks encode a handful of columns and are dense. Text tasks produce
 * a TF-IDF vector over thousands of terms of which a few dozen are non-zero,
 * so they stay sparse -- scoring 27 movie genres against a sparse vector is
 * ~1,400 multiplications instead of 135,000.
 */

import type { FloatArray } from "./pack";

export interface FeatureVector {
  /** Positions of `values`, or null when the vector is dense. */
  indices: Int32Array | null;
  values: Float64Array;
  /** Full dimensionality, which for a sparse vector exceeds `values.length`. */
  length: number;
}

export function dense(values: Float64Array): FeatureVector {
  return { indices: null, values, length: values.length };
}

export function sparse(indices: Int32Array, values: Float64Array, length: number): FeatureVector {
  return { indices, values, length };
}

/** Dot product of row `row` of a flat matrix with `vector`. */
export function dotRow(data: FloatArray, row: number, cols: number, vector: FeatureVector): number {
  const offset = row * cols;
  const { indices, values } = vector;
  let total = 0;
  if (indices === null) {
    for (let i = 0; i < values.length; i += 1) total += data[offset + i] * values[i];
  } else {
    for (let i = 0; i < indices.length; i += 1) total += data[offset + indices[i]] * values[i];
  }
  return total;
}

export function toDenseArray(vector: FeatureVector): Float64Array {
  if (vector.indices === null) return vector.values;
  const out = new Float64Array(vector.length);
  for (let i = 0; i < vector.indices.length; i += 1) out[vector.indices[i]] = vector.values[i];
  return out;
}

export function softmax(scores: number[]): number[] {
  let max = -Infinity;
  for (const score of scores) if (score > max) max = score;
  let total = 0;
  const exponentials = scores.map((score) => {
    const value = Math.exp(score - max);
    total += value;
    return value;
  });
  return exponentials.map((value) => value / total);
}

export function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}
