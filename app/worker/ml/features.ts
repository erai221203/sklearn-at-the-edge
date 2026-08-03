/**
 * The two feature pipelines, ported from scikit-learn.
 */

import type { TabularInputSpec, VectorizerSpec } from "../../shared/types";
import { readFloats, type FloatArray } from "./pack";
import { dense, sparse, type FeatureVector } from "./vector";

/**
 * scikit-learn's default `token_pattern` is `(?u)\b\w\w+\b`, which selects
 * maximal runs of Unicode word characters at least two long. Python's `\w` is
 * "alphanumeric per `str.isalnum()`, plus underscore"; JavaScript's `\w` is
 * ASCII-only, so the class is spelled out to keep accented and non-Latin
 * tokens -- which the SMS corpus contains -- matching on both sides.
 */
const TOKEN_PATTERN = /[\p{L}\p{N}_]{2,}/gu;
const SKLEARN_TOKEN_PATTERN = "(?u)\\b\\w\\w+\\b";

export function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_PATTERN) ?? [];
}

export class TfidfVectorizer {
  private readonly vocabulary: Map<string, number>;
  private readonly stopWords: Set<string>;
  private readonly idf: FloatArray;
  readonly terms: string[];

  constructor(spec: VectorizerSpec) {
    this.stopWords = new Set(spec.stopWords ?? []);
    const { analyzer } = spec;
    if (analyzer.tokenPattern !== SKLEARN_TOKEN_PATTERN || !analyzer.lowercase) {
      throw new Error("artifact uses an analyzer this implementation does not replicate");
    }
    if (analyzer.sublinearTf || analyzer.norm !== "l2" || !analyzer.smoothIdf) {
      throw new Error("artifact uses tf-idf options this implementation does not replicate");
    }

    this.idf = readFloats(spec.idf);
    this.vocabulary = new Map<string, number>();
    this.terms = new Array<string>(this.idf.length).fill("");
    for (const [term, index] of Object.entries(spec.vocabulary)) {
      this.vocabulary.set(term, index);
      this.terms[index] = term;
    }
  }

  /**
   * Counts in-vocabulary tokens, weights them by idf and L2-normalises.
   *
   * Stop words, `min_df` and `max_features` all act by keeping terms out of
   * the vocabulary, and `transform` ignores anything it has not seen, so no
   * separate filtering step is needed to match scikit-learn here.
   */
  transform(text: string): FeatureVector {
    const counts = new Map<number, number>();
    for (const token of tokenize(text)) {
      const index = this.vocabulary.get(token);
      if (index !== undefined) counts.set(index, (counts.get(index) ?? 0) + 1);
    }

    const indices = Int32Array.from(counts.keys()).sort();
    const values = new Float64Array(indices.length);
    let squared = 0;
    for (let i = 0; i < indices.length; i += 1) {
      const value = (counts.get(indices[i]) as number) * this.idf[indices[i]];
      values[i] = value;
      squared += value * value;
    }
    if (squared > 0) {
      const norm = Math.sqrt(squared);
      for (let i = 0; i < values.length; i += 1) values[i] /= norm;
    }
    return sparse(indices, values, this.idf.length);
  }

  /**
   * Splits the input into the words that carried the decision, the everyday
   * words dropped during training, and the ones never seen before. Only the
   * first group moves the answer at all.
   */
  analyse(text: string): { recognised: string[]; ignored: string[]; unknown: string[]; total: number } {
    const recognised: string[] = [];
    const ignored: string[] = [];
    const unknown: string[] = [];
    const tokens = tokenize(text);
    for (const token of tokens) {
      if (this.vocabulary.has(token)) {
        if (!recognised.includes(token)) recognised.push(token);
      } else if (this.stopWords.has(token)) {
        if (!ignored.includes(token)) ignored.push(token);
      } else if (!unknown.includes(token)) {
        unknown.push(token);
      }
    }
    return { recognised, ignored, unknown, total: tokens.length };
  }
}

export class TabularEncoder {
  constructor(private readonly spec: TabularInputSpec) {}

  /** Applies the declared column encoding, then the StandardScaler. */
  encode(record: Record<string, string | number>): FeatureVector {
    const { encode, scaler } = this.spec;
    const values = new Float64Array(encode.length);
    for (let i = 0; i < encode.length; i += 1) {
      const column = encode[i];
      const raw = record[column.from];
      if (column.equals !== undefined) {
        values[i] = String(raw) === column.equals ? 1 : 0;
      } else {
        values[i] = Number(raw);
      }
    }
    if (scaler) {
      for (let i = 0; i < values.length; i += 1) {
        values[i] = (values[i] - scaler.mean[i]) / scaler.scale[i];
      }
    }
    return dense(values);
  }
}
