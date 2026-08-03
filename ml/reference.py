"""A NumPy reading of the exported artifacts.

This module exists to answer one question before anything is deployed: does
the artifact still describe the model scikit-learn actually fitted? Each
trainer runs its estimator and this reader over the same held-out rows and
refuses to write the bundle unless they agree.

It is also the executable specification the TypeScript port in
`app/worker/ml/` follows -- the two implementations are deliberately written
to the same shapes, so a disagreement shows up as a failing parity test rather
than as a wrong answer in production.
"""

from __future__ import annotations

import base64
import re
from typing import Any, Sequence

import numpy as np

# The default TfidfVectorizer token pattern. Kept verbatim so a change in the
# trainers is caught by the assertion in `LoadedVectorizer`.
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
_TOKEN_RE = re.compile(TOKEN_PATTERN)

_PACK_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4"}


def unpack(value: Any) -> np.ndarray:
    """Read either a plain JSON array or a base64 typed array."""
    if isinstance(value, dict) and "__pack" in value:
        buffer = base64.b64decode(value["b64"])
        array = np.frombuffer(buffer, dtype=_PACK_DTYPES[value["__pack"]])
        shape = value["shape"]
        if len(shape) == 2:
            array = array.reshape(shape[0], shape[1])
        return array
    return np.asarray(value)


def unpack_floats(value: Any) -> np.ndarray:
    return unpack(value).astype(np.float64)


def unpack_ints(value: Any) -> np.ndarray:
    return unpack(value).astype(np.int64)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LoadedVectorizer:
    """TF-IDF transform rebuilt from the exported vocabulary and idf table."""

    def __init__(self, spec: dict[str, Any]) -> None:
        analyzer = spec["analyzer"]
        if analyzer["tokenPattern"] != TOKEN_PATTERN or not analyzer["lowercase"]:
            raise ValueError("artifact uses an analyzer this reader does not implement")
        if analyzer["sublinearTf"] or analyzer["norm"] != "l2" or not analyzer["smoothIdf"]:
            raise ValueError("artifact uses tf-idf options this reader does not implement")
        self.vocabulary = spec["vocabulary"]
        self.idf = unpack_floats(spec["idf"])

    def transform(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.idf), dtype=np.float64)
        for token in tokenize(text):
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] += 1.0
        vector *= self.idf
        norm = np.sqrt((vector * vector).sum())
        if norm > 0:
            vector /= norm
        return vector


def encode_tabular(spec: dict[str, Any], record: dict[str, Any]) -> np.ndarray:
    """Build the model's feature vector from a raw record.

    `encode` is a declarative column list: an entry with `equals` emits a
    one-hot indicator, otherwise the field is passed through as a number.
    """
    values: list[float] = []
    for column in spec["encode"]:
        raw = record[column["from"]]
        if "equals" in column:
            values.append(1.0 if str(raw) == column["equals"] else 0.0)
        else:
            values.append(float(raw))
    vector = np.asarray(values, dtype=np.float64)
    scaler = spec.get("scaler")
    if scaler is not None:
        vector = (vector - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    return vector


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - scores.max())
    return exp / exp.sum()


def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


class LoadedModel:
    """Decodes an artifact once, then answers predictions from it."""

    def __init__(self, art: dict[str, Any]) -> None:
        self.type = art["type"]
        self.output = art.get("output")
        if self.type == "linear":
            self.coef = unpack_floats(art["coef"])
            self.intercept = unpack_floats(art["intercept"])
        elif self.type == "multinomial_nb":
            self.feature_log_prob = unpack_floats(art["featureLogProb"])
            self.class_log_prior = unpack_floats(art["classLogPrior"])
        elif self.type in ("forest", "gbdt"):
            self.n_outputs = art["nOutputs"]
            self.roots = unpack_ints(art["roots"])
            self.feature = unpack_ints(art["feature"])
            self.threshold = unpack_floats(art["threshold"])
            self.left = unpack_ints(art["left"])
            self.right = unpack_ints(art["right"])
            self.leaf_value = unpack_floats(art["leafValue"])
            self.init = float(art.get("init", 0.0))
        else:
            raise ValueError(f"unknown artifact type: {self.type}")

    def _walk(self, root: int, x: np.ndarray) -> int:
        """`x` must already be float32-rounded -- see `featureDtype`."""
        node = int(root)
        while self.feature[node] != -1:
            if x[self.feature[node]] <= self.threshold[node]:
                node = int(self.left[node])
            else:
                node = int(self.right[node])
        return int(self.left[node])  # leaf slot, see `_pack_trees`

    def predict(self, x: Sequence[float]) -> dict[str, Any]:
        vector = np.asarray(x, dtype=np.float64)
        if self.type in ("forest", "gbdt"):
            # Match scikit-learn's tree code, which narrows X to float32 first.
            vector = vector.astype(np.float32).astype(np.float64)
        if self.type == "linear":
            return self._predict_linear(vector)
        if self.type == "multinomial_nb":
            scores = self.feature_log_prob @ vector + self.class_log_prior
            probabilities = _softmax(scores)
            return {
                "scores": scores.tolist(),
                "probabilities": probabilities.tolist(),
                "predictedIndex": int(np.argmax(scores)),
            }
        if self.type == "forest":
            totals = np.zeros(self.n_outputs, dtype=np.float64)
            for root in self.roots:
                slot = self._walk(root, vector) * self.n_outputs
                totals += self.leaf_value[slot : slot + self.n_outputs]
            totals /= len(self.roots)
            return {
                "scores": totals.tolist(),
                "probabilities": totals.tolist(),
                "predictedIndex": int(np.argmax(totals)),
            }
        raw = self.init + sum(self.leaf_value[self._walk(root, vector)] for root in self.roots)
        probability = _sigmoid(raw)
        return {
            "scores": [float(raw)],
            "probabilities": [1.0 - probability, probability],
            "predictedIndex": 1 if probability >= 0.5 else 0,
        }

    def _predict_linear(self, vector: np.ndarray) -> dict[str, Any]:
        scores = self.coef @ vector + self.intercept
        if self.output == "sigmoid":
            probability = _sigmoid(scores[0])
            return {
                "scores": [float(scores[0])],
                "probabilities": [1.0 - probability, probability],
                "predictedIndex": 1 if probability >= 0.5 else 0,
            }
        if self.output == "margin":
            return {
                "scores": [float(scores[0])],
                "probabilities": None,
                "predictedIndex": 1 if scores[0] > 0 else 0,
            }
        if self.output == "softmax":
            probabilities = _softmax(scores)
            return {
                "scores": scores.tolist(),
                "probabilities": probabilities.tolist(),
                "predictedIndex": int(np.argmax(probabilities)),
            }
        if self.output == "argmax":
            return {
                "scores": scores.tolist(),
                "probabilities": None,
                "predictedIndex": int(np.argmax(scores)),
            }
        raise ValueError(f"unknown linear output mode: {self.output}")
