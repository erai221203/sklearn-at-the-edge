"""Turn fitted scikit-learn estimators into portable JSON artifacts.

Cloudflare Workers run JavaScript on V8 isolates, so scikit-learn cannot be
shipped to the edge. Instead every estimator is reduced to the small set of
numbers its inference path actually needs (coefficients, log-probabilities,
tree tables) and re-implemented in TypeScript in `app/worker/ml/`.

The formats defined here are the contract between the two languages. Anything
added on this side must gain a matching reader in `app/worker/ml/` and a
fixture in the parity suite, otherwise `make_parity_fixtures.py` will not be
able to prove the two implementations agree.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import sklearn

SCHEMA_VERSION = 1

# Weights are rounded before serialisation: full float64 repr roughly doubles
# the artifact size for accuracy differences far below the class-decision gap.
WEIGHT_DECIMALS = 6

# Above this many elements an array is base64-encoded rather than written as
# JSON numbers. The 27 x 5000 movie-genre matrices and the churn forest's node
# tables are the reason: as JSON they are several megabytes each and cost tens
# of milliseconds of `JSON.parse` on the first request an isolate serves,
# which is most of the Workers free-tier CPU budget. Decoding base64 into a
# typed array is a linear byte copy instead. Smaller arrays stay as plain JSON
# so the artifacts remain readable and diffable.
PACK_MIN_ELEMENTS = 4096
_PACK_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_OUT_DIR = os.path.join(REPO_ROOT, "app", "public", "models")
FIXTURE_OUT_DIR = os.path.join(REPO_ROOT, "app", "tests", "fixtures")


def _round(value: float, decimals: int) -> float:
    rounded = round(float(value), decimals)
    # Normalise -0.0 so the JSON stays byte-stable between runs.
    return rounded + 0.0


def round_list(values: Iterable[float], decimals: int = WEIGHT_DECIMALS) -> list[float]:
    return [_round(v, decimals) for v in values]


def round_matrix(rows: Iterable[Sequence[float]], decimals: int = WEIGHT_DECIMALS) -> list[list[float]]:
    return [round_list(row, decimals) for row in rows]


def _encode(values: np.ndarray, dtype: str, shape: list[int]) -> dict[str, Any]:
    little_endian = values.astype(_PACK_DTYPES[dtype], copy=False)
    return {
        "__pack": dtype,
        "shape": shape,
        "b64": base64.b64encode(little_endian.tobytes(order="C")).decode("ascii"),
    }


def pack_vector(values: Iterable[float], dtype: str = "f32", decimals: int | None = WEIGHT_DECIMALS):
    """Plain JSON list when small, base64 typed array when large.

    `decimals=None` keeps full float64 precision, which tree split points need.
    """
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    if array.size < PACK_MIN_ELEMENTS:
        if dtype == "i32":
            return [int(v) for v in array]
        return [float(v) for v in array] if decimals is None else round_list(array, decimals)
    return _encode(array.ravel(), dtype, [int(array.size)])


def pack_matrix(rows, dtype: str = "f32", decimals: int = WEIGHT_DECIMALS):
    array = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    if array.size < PACK_MIN_ELEMENTS:
        return round_matrix(array, decimals)
    return _encode(array.ravel(order="C"), dtype, [int(array.shape[0]), int(array.shape[1])])


# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------

def export_scaler(scaler) -> dict[str, Any]:
    """StandardScaler -> (x - mean) / scale."""
    return {
        "mean": round_list(scaler.mean_, 8),
        "scale": round_list(scaler.scale_, 8),
    }


def export_tfidf(vectorizer) -> dict[str, Any]:
    """TfidfVectorizer -> vocabulary + idf table.

    Only the vocabulary and idf weights are needed at inference time. Stop
    words, `max_features` and `min_df` all act by *excluding* terms from the
    vocabulary, and `transform` ignores any token that is not in the
    vocabulary -- so the TypeScript tokenizer can skip those steps entirely
    and still produce an identical vector.
    """
    vocabulary = {str(term): int(index) for term, index in vectorizer.vocabulary_.items()}
    # Shipped so the app can tell a word it deliberately ignores ("this",
    # "from") apart from one it has simply never seen ("erai"). Both are
    # absent from the vocabulary, but they mean very different things to
    # someone asking why a word was not used.
    stop_words = vectorizer.get_stop_words()
    return {
        "vocabulary": vocabulary,
        "stopWords": sorted(str(word) for word in stop_words) if stop_words else [],
        "idf": pack_vector(vectorizer.idf_, "f32"),
        # Recorded so the TypeScript tokenizer can assert it implements the
        # same analyzer rather than silently drifting from it.
        "analyzer": {
            "lowercase": bool(vectorizer.lowercase),
            "tokenPattern": vectorizer.token_pattern,
            "sublinearTf": bool(vectorizer.sublinear_tf),
            "norm": vectorizer.norm,
            "smoothIdf": bool(vectorizer.smooth_idf),
        },
    }


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def export_linear(clf, output: str, decimals: int = WEIGHT_DECIMALS) -> dict[str, Any]:
    """Any estimator exposing `coef_` / `intercept_`.

    `output` selects how the raw scores become a user-facing answer:
      sigmoid -> binary probability from a single row of coefficients
      softmax -> multinomial probabilities over one row per class
      margin  -> signed distance only (SVC without `probability=True` has no
                 calibrated probabilities, and inventing one would be a lie)
      argmax  -> one-vs-rest scores, highest wins
    """
    raw_coef = clf.coef_
    if hasattr(raw_coef, "todense"):  # SVC(kernel="linear") returns a sparse matrix
        raw_coef = np.asarray(raw_coef.todense())
    coef = np.atleast_2d(np.asarray(raw_coef, dtype=np.float64))
    intercept = np.atleast_1d(np.asarray(clf.intercept_, dtype=np.float64))
    return {
        "type": "linear",
        "output": output,
        "coef": pack_matrix(coef, "f32", decimals),
        "intercept": round_list(intercept, 8),
    }


def export_multinomial_nb(clf, decimals: int = WEIGHT_DECIMALS) -> dict[str, Any]:
    return {
        "type": "multinomial_nb",
        "classLogPrior": round_list(np.asarray(clf.class_log_prior_, dtype=np.float64), 8),
        "featureLogProb": pack_matrix(
            np.asarray(clf.feature_log_prob_, dtype=np.float64), "f32", decimals
        ),
    }


def _pack_trees(estimators, n_outputs: int, scale: float, normalize: bool) -> dict[str, Any]:
    """Flatten a list of sklearn trees into parallel arrays.

    Every tree is appended to one shared node table and `roots` records where
    each starts, which keeps the JSON free of per-tree object overhead. A node
    is a leaf when `feature[i] == -1`; for those nodes `left[i]` is reused to
    point at the leaf's slot in `leafValue` instead of a child node.
    """
    feature: list[int] = []
    threshold: list[float] = []
    left: list[int] = []
    right: list[int] = []
    leaf_value: list[float] = []
    roots: list[int] = []

    for est in estimators:
        tree = est.tree_
        offset = len(feature)
        roots.append(offset)
        for node in range(tree.node_count):
            child_left = int(tree.children_left[node])
            if child_left == -1:  # leaf
                values = np.asarray(tree.value[node][0], dtype=np.float64)
                if normalize:
                    # sklearn >= 1.4 already stores class fractions for
                    # classifiers, older versions store weighted counts.
                    total = values.sum()
                    if total > 0:
                        values = values / total
                feature.append(-1)
                threshold.append(0.0)
                left.append(len(leaf_value) // n_outputs)
                right.append(-1)
                leaf_value.extend((values * scale).tolist())
            else:
                feature.append(int(tree.feature[node]))
                threshold.append(float(tree.threshold[node]))
                left.append(offset + child_left)
                right.append(offset + int(tree.children_right[node]))

    return {
        "nOutputs": n_outputs,
        "nNodes": len(feature),
        "roots": pack_vector(roots, "i32"),
        "feature": pack_vector(feature, "i32"),
        # Split points are stored exactly, never rounded. scikit-learn picks a
        # threshold as the midpoint of two adjacent *float32* feature values,
        # so a split can sit ~2e-9 away from a real data point -- rounding to
        # eight decimals is enough to send that sample down the wrong branch.
        "threshold": pack_vector(threshold, "f64", None),
        "left": pack_vector(left, "i32"),
        "right": pack_vector(right, "i32"),
        "leafValue": pack_vector(leaf_value, "f64", None),
        # scikit-learn casts X to float32 before walking a tree, so a faithful
        # reader has to compare against the float32 rounding of each feature.
        "featureDtype": "f32",
    }


def export_random_forest(clf) -> dict[str, Any]:
    packed = _pack_trees(clf.estimators_, n_outputs=len(clf.classes_), scale=1.0, normalize=True)
    return {"type": "forest", **packed}


def export_gradient_boosting(clf, X_reference: np.ndarray) -> dict[str, Any]:
    """Binary GradientBoostingClassifier -> constant + sum of scaled trees.

    The starting log-odds live behind a private attribute that has moved
    between scikit-learn releases, so it is recovered arithmetically instead:
    `decision_function - learning_rate * sum(trees)` is the same constant for
    every sample. That value is asserted to be constant here, which also
    catches the case where a future release folds the learning rate into the
    stored leaf values.
    """
    if clf.estimators_.shape[1] != 1:
        raise ValueError("only binary GradientBoostingClassifier is supported")

    estimators = [stage[0] for stage in clf.estimators_]
    packed = _pack_trees(estimators, n_outputs=1, scale=float(clf.learning_rate), normalize=False)

    tree_sum = np.zeros(len(X_reference), dtype=np.float64)
    for est in estimators:
        tree_sum += est.predict(X_reference) * clf.learning_rate
    init_candidates = clf.decision_function(X_reference).ravel() - tree_sum
    spread = float(init_candidates.max() - init_candidates.min())
    if spread > 1e-6:
        raise ValueError(
            f"gradient boosting init term is not constant (spread={spread:.3g}); "
            "the scikit-learn prediction path has changed and the exporter needs updating"
        )

    return {
        "type": "gbdt",
        "output": "sigmoid",
        "init": _round(float(init_candidates.mean()), 10),
        **packed,
    }


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def classification_metrics(y_true, y_pred, y_score=None, average: str = "binary") -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, Any] = {
        "accuracy": _round(accuracy_score(y_true, y_pred), 6),
        "precision": _round(precision_score(y_true, y_pred, average=average, zero_division=0), 6),
        "recall": _round(recall_score(y_true, y_pred, average=average, zero_division=0), 6),
        "f1": _round(f1_score(y_true, y_pred, average=average, zero_division=0), 6),
    }
    if average != "binary":
        metrics["f1Weighted"] = _round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 6
        )
    if y_score is not None and average == "binary":
        metrics["rocAuc"] = _round(roc_auc_score(y_true, y_score), 6)
    return metrics


def confusion(y_true, y_pred, n_classes: int) -> list[list[int]]:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return [[int(v) for v in row] for row in matrix]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def base_bundle(task: str, title: str, subtitle: str, classes: Sequence[str]) -> dict[str, Any]:
    return {
        "task": task,
        "title": title,
        "subtitle": subtitle,
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearnVersion": sklearn.__version__,
        "classes": list(classes),
        "models": [],
    }


def write_bundle(bundle: dict[str, Any]) -> tuple[str, str]:
    """Split a task into a light metadata file and a heavy weights file.

    The catalogue and metrics screens only ever need the metadata, so the
    worker can answer `/api/models` without touching megabytes of
    coefficients. Weights are pulled in on the first prediction for a task and
    then cached for the life of the isolate.
    """
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    weights: dict[str, Any] = {
        "task": bundle["task"],
        "schemaVersion": SCHEMA_VERSION,
        "models": {},
    }
    for model in bundle["models"]:
        weights["models"][model["id"]] = model.pop("artifact")
    vectorizer = bundle.get("input", {}).pop("vectorizer", None)
    if vectorizer is not None:
        weights["vectorizer"] = vectorizer
    bundle["modelIds"] = [model["id"] for model in bundle["models"]]

    paths = []
    for suffix, payload in (("meta", bundle), ("weights", weights)):
        path = os.path.join(MODEL_OUT_DIR, f"{bundle['task']}.{suffix}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), ensure_ascii=False)
        print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({os.path.getsize(path) / 1024:,.0f} KB)")
        paths.append(path)
    return paths[0], paths[1]


def write_fixtures(task: str, fixtures: dict[str, Any]) -> str:
    os.makedirs(FIXTURE_OUT_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_OUT_DIR, f"{task}.parity.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(fixtures, handle, indent=1, ensure_ascii=False)
    print(f"  wrote {os.path.relpath(path, REPO_ROOT)} ({len(fixtures['cases'])} parity cases)")
    return path
