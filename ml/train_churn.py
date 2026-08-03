"""Customer churn prediction -- Logistic Regression vs Random Forest vs Gradient Boosting.

Grown from the original `churn.py`, with three changes worth calling out:

* `Geography` and `Gender` were dropped by the original script. Both carry real
  signal (German customers churn at roughly twice the French rate), so they are
  one-hot encoded and kept.
* The tree ensembles are depth-limited. Fully grown forests memorise the
  training set and, more practically here, serialise to tens of megabytes.
* Threshold sweeps and ROC points are exported alongside the headline metrics.
  Accuracy alone flatters a model on a dataset that is ~80% non-churners.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import artifact as art
import reference as ref

RANDOM_STATE = 42
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
FEATURE_LABELS = [
    "Credit score",
    "Age",
    "Tenure (years)",
    "Account balance",
    "Number of products",
    "Has credit card",
    "Is active member",
    "Estimated salary",
    "Geography: Germany",
    "Geography: Spain",
    "Gender: Male",
]


def build_features(frame: pd.DataFrame) -> np.ndarray:
    columns = [frame[name].astype(float).to_numpy() for name in NUMERIC_FEATURES]
    columns.append((frame["Geography"] == "Germany").astype(float).to_numpy())
    columns.append((frame["Geography"] == "Spain").astype(float).to_numpy())
    columns.append((frame["Gender"] == "Male").astype(float).to_numpy())
    return np.column_stack(columns)


def input_spec(frame: pd.DataFrame, scaler: StandardScaler) -> dict:
    """Form definition for the frontend plus the encoding the worker replays."""

    def numeric_field(column: str, name: str, label: str, help_text: str, step: float = 1) -> dict:
        series = frame[column].astype(float)
        return {
            "name": name,
            "label": label,
            "help": help_text,
            "type": "number",
            "min": float(np.floor(series.min())),
            "max": float(np.ceil(series.max())),
            "step": step,
            "default": float(round(series.median(), 2)),
        }

    fields = [
        numeric_field("CreditScore", "creditScore", "Credit score", "Bureau score, 350-850."),
        numeric_field("Age", "age", "Age", "Customer age in years."),
        numeric_field("Tenure", "tenure", "Tenure", "Years as a customer."),
        numeric_field("Balance", "balance", "Account balance", "Current balance in EUR.", step=100),
        numeric_field("NumOfProducts", "numOfProducts", "Products held", "Bank products held, 1-4."),
        numeric_field(
            "EstimatedSalary", "estimatedSalary", "Estimated salary", "Annual salary in EUR.", step=100
        ),
        {
            "name": "hasCrCard",
            "label": "Holds a credit card",
            "help": "Whether the customer holds a card with the bank.",
            "type": "boolean",
            "default": 1,
        },
        {
            "name": "isActiveMember",
            "label": "Active member",
            "help": "Recent account activity. The single strongest lever in this dataset.",
            "type": "boolean",
            "default": 1,
        },
        {
            "name": "geography",
            "label": "Geography",
            "help": "Branch country.",
            "type": "category",
            "options": ["France", "Germany", "Spain"],
            "default": "France",
        },
        {
            "name": "gender",
            "label": "Gender",
            "help": "As recorded by the bank.",
            "type": "category",
            "options": ["Female", "Male"],
            "default": "Female",
        },
    ]

    encode = [{"from": name} for name in
              ["creditScore", "age", "tenure", "balance", "numOfProducts", "hasCrCard",
               "isActiveMember", "estimatedSalary"]]
    encode += [
        {"from": "geography", "equals": "Germany"},
        {"from": "geography", "equals": "Spain"},
        {"from": "gender", "equals": "Male"},
    ]

    return {
        "kind": "tabular",
        "fields": fields,
        "encode": encode,
        "featureLabels": FEATURE_LABELS,
        "scaler": art.export_scaler(scaler),
    }


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, points: int = 41) -> list[dict]:
    """Precision / recall / F1 across the operating range of a score."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    grid = np.linspace(lo, hi, points)[:-1]
    sweep = []
    for threshold in grid:
        predicted = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predicted, average="binary", zero_division=0
        )
        sweep.append(
            {
                "threshold": round(float(threshold), 6),
                "precision": round(float(precision), 5),
                "recall": round(float(recall), 5),
                "f1": round(float(f1), 5),
            }
        )
    return sweep


def roc_points(y_true: np.ndarray, scores: np.ndarray, points: int = 60) -> list[dict]:
    fpr, tpr, _ = roc_curve(y_true, scores)
    if len(fpr) > points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return [{"fpr": round(float(a), 5), "tpr": round(float(b), 5)} for a, b in zip(fpr, tpr)]


def main() -> None:
    print("Customer churn prediction")
    csv_path = os.path.join(art.REPO_ROOT, "Churn_Modelling.csv")
    frame = pd.read_csv(csv_path)

    X = build_features(frame)
    y = frame["Exited"].astype(int).to_numpy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    bundle = art.base_bundle(
        task="churn",
        title="Customer Churn Prediction",
        subtitle="Which retail-banking customers are about to leave, and why.",
        classes=["Retained", "Churned"],
    )
    bundle["kind"] = "binary"
    bundle["positiveClass"] = 1
    bundle["input"] = input_spec(frame, scaler)
    bundle["dataset"] = {
        "name": "Churn_Modelling.csv",
        "rows": int(len(frame)),
        "trainRows": int(len(y_train)),
        "testRows": int(len(y_test)),
        "features": len(FEATURE_LABELS),
        "classBalance": {
            "Retained": int((y == 0).sum()),
            "Churned": int((y == 1).sum()),
        },
    }

    specs = [
        (
            "logistic_regression",
            "Logistic Regression",
            "A single weighted sum of the inputs. Fast, and every coefficient is readable.",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        (
            "random_forest",
            "Random Forest",
            "200 depth-limited trees voting independently. Captures interactions the linear model cannot.",
            # Depth and leaf size are capped deliberately. A fully grown forest
            # memorises the training set and serialises to tens of megabytes of
            # node tables; this one is both better regularised and ~10x smaller.
            RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "gradient_boosting",
            "Gradient Boosting",
            "Shallow trees fitted in sequence, each correcting the last. Usually the strongest here.",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
    ]

    fixture_cases: list[dict] = []
    for model_id, name, description, estimator in specs:
        print(f"  training {name} ...")
        estimator.fit(X_train_scaled, y_train)
        predictions = estimator.predict(X_test_scaled)
        probabilities = estimator.predict_proba(X_test_scaled)[:, 1]

        if model_id == "logistic_regression":
            exported = art.export_linear(estimator, output="sigmoid")
            explain = {
                "kind": "coefficients",
                "values": art.round_list(estimator.coef_[0], 6),
            }
        elif model_id == "random_forest":
            exported = art.export_random_forest(estimator)
            explain = {
                "kind": "importances",
                "values": art.round_list(estimator.feature_importances_, 6),
            }
        else:
            exported = art.export_gradient_boosting(estimator, X_test_scaled[:200])
            explain = {
                "kind": "importances",
                "values": art.round_list(estimator.feature_importances_, 6),
            }

        # The artifact must reproduce the estimator it was derived from.
        loaded = ref.LoadedModel(exported)
        checked = min(400, len(X_test_scaled))
        for row in range(checked):
            got = loaded.predict(X_test_scaled[row])
            assert got["predictedIndex"] == int(predictions[row]), (
                f"{model_id}: exported artifact disagrees with scikit-learn on test row {row}"
            )
            assert abs(got["probabilities"][1] - probabilities[row]) < 1e-6, (
                f"{model_id}: exported probability drifted on test row {row}"
            )

        bundle["models"].append(
            {
                "id": model_id,
                "name": name,
                "description": description,
                "metrics": art.classification_metrics(y_test, predictions, probabilities),
                "confusion": art.confusion(y_test, predictions, 2),
                "rocCurve": roc_points(y_test, probabilities),
                "thresholdSweep": threshold_sweep(y_test, probabilities),
                "explain": explain,
                "artifact": exported,
            }
        )
        print(
            f"    accuracy={bundle['models'][-1]['metrics']['accuracy']:.4f} "
            f"roc_auc={bundle['models'][-1]['metrics']['rocAuc']:.4f} "
            f"recall={bundle['models'][-1]['metrics']['recall']:.4f}"
        )

    # Parity fixtures: raw records in, scikit-learn's own answers out.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_rows = rng.choice(len(idx_test), size=60, replace=False)
    for row in sample_rows:
        source = frame.iloc[int(idx_test[row])]
        record = {
            "creditScore": float(source["CreditScore"]),
            "age": float(source["Age"]),
            "tenure": float(source["Tenure"]),
            "balance": float(source["Balance"]),
            "numOfProducts": float(source["NumOfProducts"]),
            "hasCrCard": float(source["HasCrCard"]),
            "isActiveMember": float(source["IsActiveMember"]),
            "estimatedSalary": float(source["EstimatedSalary"]),
            "geography": str(source["Geography"]),
            "gender": str(source["Gender"]),
        }
        expected = {}
        for (model_id, _, _, estimator) in specs:
            vector = X_test_scaled[row : row + 1]
            expected[model_id] = {
                "predictedIndex": int(estimator.predict(vector)[0]),
                "probabilities": [float(p) for p in estimator.predict_proba(vector)[0]],
            }
        fixture_cases.append({"input": record, "expected": expected})

    art.write_bundle(bundle)
    art.write_fixtures("churn", {"task": "churn", "cases": fixture_cases})


if __name__ == "__main__":
    main()
