"""SMS spam detection -- Naive Bayes vs Logistic Regression vs linear SVM.

Follows the original `spam.py` (TF-IDF over the SMS Spam Collection, three
classifiers compared) and adds a stratified split, held-out ranking metrics,
and the token-level weights the frontend uses to show *why* a message was
flagged.

The SVM is deliberately left uncalibrated: `SVC` without `probability=True`
has no probability to report, so the artifact carries `output: "margin"` and
the UI shows a signed distance from the decision boundary instead of inventing
a confidence number.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

import artifact as art
import reference as ref
from train_churn import roc_points, threshold_sweep

RANDOM_STATE = 42

SAMPLE_MESSAGES = [
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jawa. Text WIN to 80086 now",
    "Hey, are we still on for lunch tomorrow? Let me know what time works.",
    "Congratulations! You've been selected to receive a free iPhone. Click here to claim your prize.",
    "Can you pick up milk on your way home please",
    "FreeMsg: Claim your 500 free text messages. Reply YES to 87121 to activate. Std rates apply.",
    "I'll be there in 10 minutes, just leaving the office now",
]


def main() -> None:
    print("SMS spam detection")
    csv_path = os.path.join(art.REPO_ROOT, "spam.csv")
    frame = pd.read_csv(csv_path, encoding="latin-1")[["v1", "v2"]]
    frame.columns = ["label", "message"]
    frame = frame.dropna(subset=["message"])
    y = frame["label"].map({"ham": 0, "spam": 1}).astype(int).to_numpy()
    messages = frame["message"].astype(str).to_numpy()

    msg_train, msg_test, y_train, y_test = train_test_split(
        messages, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(msg_train)
    X_test = vectorizer.transform(msg_test)
    print(f"  vocabulary: {len(vectorizer.vocabulary_):,} terms")

    bundle = art.base_bundle(
        task="spam",
        title="SMS Spam Detection",
        subtitle="Separating scam texts from real ones, and showing which words gave it away.",
        classes=["Ham", "Spam"],
    )
    bundle["kind"] = "binary"
    bundle["positiveClass"] = 1
    bundle["input"] = {
        "kind": "text",
        "label": "SMS message",
        "placeholder": "Paste an SMS message...",
        "samples": SAMPLE_MESSAGES,
        "vectorizer": art.export_tfidf(vectorizer),
    }
    bundle["dataset"] = {
        "name": "spam.csv (SMS Spam Collection)",
        "rows": int(len(frame)),
        "trainRows": int(len(y_train)),
        "testRows": int(len(y_test)),
        "features": len(vectorizer.vocabulary_),
        "classBalance": {"Ham": int((y == 0).sum()), "Spam": int((y == 1).sum())},
    }

    specs = [
        (
            "naive_bayes",
            "Multinomial Naive Bayes",
            "Scores each word independently. The classic spam-filter baseline, and still hard to beat.",
            MultinomialNB(),
        ),
        (
            "logistic_regression",
            "Logistic Regression",
            "Learns one weight per word and adds them up. Gives calibrated probabilities.",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        (
            "linear_svm",
            "Linear SVM",
            "Finds the widest possible margin between the two classes. Reports distance, not probability.",
            SVC(kernel="linear", random_state=RANDOM_STATE),
        ),
    ]

    for model_id, name, description, estimator in specs:
        print(f"  training {name} ...")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        has_proba = hasattr(estimator, "predict_proba")
        probabilities = estimator.predict_proba(X_test)[:, 1] if has_proba else None
        ranking_score = probabilities if has_proba else estimator.decision_function(X_test)

        if model_id == "naive_bayes":
            exported = art.export_multinomial_nb(estimator)
            weights = estimator.feature_log_prob_[1] - estimator.feature_log_prob_[0]
        elif model_id == "logistic_regression":
            exported = art.export_linear(estimator, output="sigmoid")
            weights = estimator.coef_[0]
        else:
            exported = art.export_linear(estimator, output="margin")
            coef = estimator.coef_
            weights = np.asarray(coef.todense() if hasattr(coef, "todense") else coef).ravel()

        loaded = ref.LoadedModel(exported)
        dense_test = np.asarray(X_test.todense())
        checked = min(300, dense_test.shape[0])
        for row in range(checked):
            got = loaded.predict(dense_test[row])
            assert got["predictedIndex"] == int(predictions[row]), (
                f"{model_id}: exported artifact disagrees with scikit-learn on test row {row}"
            )
            if has_proba:
                assert abs(got["probabilities"][1] - probabilities[row]) < 1e-6, (
                    f"{model_id}: exported probability drifted on test row {row}"
                )

        # The re-implemented tokenizer must also survive the round trip from raw text.
        reader = ref.LoadedVectorizer(bundle["input"]["vectorizer"])
        for message in msg_test[:400]:
            expected = int(estimator.predict(vectorizer.transform([str(message)]))[0])
            assert loaded.predict(reader.transform(str(message)))["predictedIndex"] == expected, (
                f"{model_id}: re-implemented TF-IDF changed the prediction for {message!r}"
            )

        bundle["models"].append(
            {
                "id": model_id,
                "name": name,
                "description": description,
                "metrics": art.classification_metrics(y_test, predictions, ranking_score),
                "confusion": art.confusion(y_test, predictions, 2),
                "rocCurve": roc_points(y_test, ranking_score),
                "thresholdSweep": threshold_sweep(y_test, ranking_score),
                "explain": {"kind": "tokenWeights"},
                "artifact": exported,
            }
        )
        metrics = bundle["models"][-1]["metrics"]
        print(
            f"    accuracy={metrics['accuracy']:.4f} roc_auc={metrics['rocAuc']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f}"
        )

        # Most indicative vocabulary, taken from this model's own weights.
        terms = np.array(vectorizer.get_feature_names_out())
        order = np.argsort(weights)
        bundle["models"][-1]["topTokens"] = {
            "spam": [
                {"term": str(terms[i]), "weight": round(float(weights[i]), 4)}
                for i in order[::-1][:18]
            ],
            "ham": [
                {"term": str(terms[i]), "weight": round(float(weights[i]), 4)}
                for i in order[:18]
            ],
        }

    rng = np.random.default_rng(RANDOM_STATE)
    sample_rows = rng.choice(len(msg_test), size=60, replace=False)
    texts = [str(msg_test[i]) for i in sample_rows] + SAMPLE_MESSAGES
    fixture_cases = []
    for text in texts:
        vector = vectorizer.transform([text])
        expected = {}
        for model_id, _, _, estimator in specs:
            entry = {"predictedIndex": int(estimator.predict(vector)[0])}
            if hasattr(estimator, "predict_proba"):
                entry["probabilities"] = [float(p) for p in estimator.predict_proba(vector)[0]]
            else:
                entry["scores"] = [float(estimator.decision_function(vector)[0])]
            expected[model_id] = entry
        fixture_cases.append({"input": {"text": text}, "expected": expected})

    art.write_bundle(bundle)
    art.write_fixtures("spam", {"task": "spam", "cases": fixture_cases})


if __name__ == "__main__":
    main()
