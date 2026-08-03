/**
 * Readers for the two array encodings `ml/artifact.py` emits.
 *
 * Small arrays arrive as plain JSON numbers. Large ones -- the forest node
 * tables and the 27 x 5000 movie-genre matrices -- arrive base64-encoded,
 * because parsing them as JSON costs tens of milliseconds on the first request
 * an isolate serves and that is most of the Workers free-tier CPU budget.
 */

import type { FloatArrayJson, IntArrayJson, MatrixJson, PackedArray } from "../../shared/types";

export type FloatArray = Float32Array | Float64Array;

/** Matrices are held flat; row `r` starts at `r * cols`. */
export interface Matrix {
  rows: number;
  cols: number;
  data: FloatArray;
}

function isPacked(value: unknown): value is PackedArray {
  return typeof value === "object" && value !== null && "__pack" in value;
}

function base64ToBytes(b64: string): Uint8Array {
  // Available in recent V8; roughly an order of magnitude faster than the
  // char-by-char fallback on the multi-megabyte movie-genre weights.
  const native = (Uint8Array as unknown as { fromBase64?: (s: string) => Uint8Array }).fromBase64;
  if (typeof native === "function") return native(b64);

  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function decode(packed: PackedArray): FloatArray | Int32Array {
  const bytes = base64ToBytes(packed.b64);
  // Typed-array views require the byte offset to be a multiple of the element
  // size, which a decoder is not obliged to guarantee.
  const width = packed.__pack === "f64" ? 8 : 4;
  const aligned =
    bytes.byteOffset % width === 0
      ? bytes
      : new Uint8Array(bytes.slice().buffer);
  const { buffer, byteOffset, byteLength } = aligned;
  const count = byteLength / width;

  switch (packed.__pack) {
    case "f32":
      return new Float32Array(buffer, byteOffset, count);
    case "f64":
      return new Float64Array(buffer, byteOffset, count);
    case "i32":
      return new Int32Array(buffer, byteOffset, count);
  }
}

export function readFloats(value: FloatArrayJson): FloatArray {
  if (isPacked(value)) return decode(value) as FloatArray;
  return Float64Array.from(value);
}

export function readInts(value: IntArrayJson): Int32Array {
  if (isPacked(value)) return decode(value) as Int32Array;
  return Int32Array.from(value);
}

export function readMatrix(value: MatrixJson): Matrix {
  if (isPacked(value)) {
    const [rows, cols] = value.shape;
    return { rows, cols, data: decode(value) as FloatArray };
  }
  const rows = value.length;
  const cols = rows === 0 ? 0 : value[0].length;
  const data = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r += 1) data.set(value[r], r * cols);
  return { rows, cols, data };
}
