"""Movie genre classification from plot summaries -- 27 genres, three classifiers.

Two corrections to the original `Movie_genere_classification.py`:

* It wrapped a `MultiLabelBinarizer` + `MultiOutputClassifier` around a dataset
  where every row carries exactly one genre. Twenty-seven independent binary
  classifiers then had to agree unanimously to produce a correct label set, and
  most rows came back empty -- which is where the reported 27.7% came from.
  Treating it as single-label multiclass is both correct and far stronger.
* It scored the model on the data it was trained on. The published number was
  training accuracy. Here the model is scored on `test_data_solution.txt`,
  which it never sees during fitting.
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import artifact as art
import reference as ref

RANDOM_STATE = 42
MAX_FEATURES = 5000

SAMPLE_PLOTS = [
    "A retired hitman is pulled back into the criminal underworld when a gang murders "
    "the dog left to him by his late wife, and he sets out to hunt down every last one of them.",
    "Two sisters open a bakery in a small coastal town and slowly rebuild their relationship "
    "over one summer, learning to forgive their mother along the way.",
    "A team of astronauts travels through a newly discovered wormhole in a desperate search "
    "for a habitable planet as crops fail and dust storms swallow the Earth.",
    "This film follows three families in rural Kenya over four years as they fight a mining "
    "company for the right to remain on their ancestral land.",
    "A young wizard discovers on his eleventh birthday that he has been accepted to a school "
    "of magic, where he must confront the dark sorcerer who killed his parents.",
]


def load_dataset(path: str, with_genre: bool) -> tuple[list[str], list[str], list[str]]:
    """Read the ` ::: ` delimited format: ID ::: TITLE ::: [GENRE :::] PLOT."""
    titles, genres, plots = [], [], []
    expected = 4 if with_genre else 3
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(":::", expected - 1)]
            if len(parts) != expected:
                raise ValueError(f"{os.path.basename(path)}:{line_number}: expected {expected} fields")
            titles.append(parts[1])
            if with_genre:
                genres.append(parts[2].lower())
                plots.append(parts[3])
            else:
                plots.append(parts[2])
    return titles, genres, plots


def per_class_report(y_true, y_pred, classes) -> list[dict]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )
    rows = [
        {
            "genre": classes[i],
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    ]
    return sorted(rows, key=lambda row: row["support"], reverse=True)


def top_terms_per_class(weights: np.ndarray, terms: np.ndarray, classes, count: int = 8) -> list[dict]:
    result = []
    for index, genre in enumerate(classes):
        order = np.argsort(weights[index])[::-1][:count]
        result.append(
            {
                "genre": genre,
                "terms": [
                    {"term": str(terms[i]), "weight": round(float(weights[index][i]), 4)}
                    for i in order
                ],
            }
        )
    return result


def main() -> None:
    print("Movie genre classification")
    train_path = os.path.join(art.REPO_ROOT, "data", "train_data.txt")
    test_path = os.path.join(art.REPO_ROOT, "data", "test_data_solution.txt")
    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {os.path.relpath(path, art.REPO_ROOT)}.\n"
                "Run ml/fetch_movie_data.py first (see ml/README.md)."
            )

    _, train_genres, train_plots = load_dataset(train_path, with_genre=True)
    test_titles, test_genres, test_plots = load_dataset(test_path, with_genre=True)

    classes = sorted(set(train_genres))
    class_index = {genre: i for i, genre in enumerate(classes)}
    y_train = np.array([class_index[g] for g in train_genres])
    # A genre that only appears in the test split cannot be predicted; drop
    # those rows rather than scoring the models against an impossible target.
    keep = [i for i, g in enumerate(test_genres) if g in class_index]
    if len(keep) != len(test_genres):
        print(f"  dropped {len(test_genres) - len(keep)} test rows with unseen genres")
    test_plots = [test_plots[i] for i in keep]
    test_titles = [test_titles[i] for i in keep]
    y_test = np.array([class_index[test_genres[i]] for i in keep])
    print(f"  {len(y_train):,} train / {len(y_test):,} test rows across {len(classes)} genres")

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english", min_df=2)
    X_train = vectorizer.fit_transform(train_plots)
    X_test = vectorizer.transform(test_plots)
    print(f"  vocabulary: {len(vectorizer.vocabulary_):,} terms")

    bundle = art.base_bundle(
        task="movie-genre",
        title="Movie Genre Classification",
        subtitle="Reading a plot summary and picking one of 27 genres.",
        classes=classes,
    )
    bundle["kind"] = "multiclass"
    bundle["positiveClass"] = None
    bundle["input"] = {
        "kind": "text",
        "label": "Plot summary",
        "placeholder": "Describe the plot of a film...",
        "samples": SAMPLE_PLOTS,
        "vectorizer": art.export_tfidf(vectorizer),
    }
    counts = np.bincount(y_train, minlength=len(classes))
    bundle["dataset"] = {
        "name": "IMDb genre classification (train_data.txt)",
        "rows": int(len(y_train) + len(y_test)),
        "trainRows": int(len(y_train)),
        "testRows": int(len(y_test)),
        "features": len(vectorizer.vocabulary_),
        "classBalance": {classes[i]: int(counts[i]) for i in np.argsort(counts)[::-1]},
    }

    specs = [
        (
            "naive_bayes",
            "Multinomial Naive Bayes",
            "Treats a plot as a bag of independent words. Fast to fit, and biased toward the common genres.",
            MultinomialNB(alpha=0.3),
            "softmax_nb",
        ),
        (
            "logistic_regression",
            "Logistic Regression",
            "One weight per word per genre, trained jointly with a softmax. Gives comparable probabilities.",
            LogisticRegression(max_iter=1000, C=5.0, random_state=RANDOM_STATE),
            "softmax",
        ),
        (
            "linear_svm",
            "Linear SVM",
            "One-vs-rest margins, highest score wins. Usually the strongest on sparse text like this.",
            LinearSVC(C=0.5, random_state=RANDOM_STATE),
            "argmax",
        ),
    ]

    terms = np.array(vectorizer.get_feature_names_out())
    for model_id, name, description, estimator, output in specs:
        print(f"  training {name} ...")
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)

        if output == "softmax_nb":
            exported = art.export_multinomial_nb(estimator)
            # Raw log-probabilities are all negative and rank the *commonest*
            # words in a genre, which is the same boring list every time.
            # Centring across genres turns them into "how much more this genre
            # uses this word than the others do", which is what the UI shows.
            log_prob = estimator.feature_log_prob_
            weights = log_prob - log_prob.mean(axis=0)
        else:
            exported = art.export_linear(estimator, output=output)
            weights = np.asarray(estimator.coef_)

        loaded = ref.LoadedModel(exported)
        reader = ref.LoadedVectorizer(bundle["input"]["vectorizer"])
        for row in range(min(400, len(test_plots))):
            got = loaded.predict(reader.transform(test_plots[row]))
            assert got["predictedIndex"] == int(predictions[row]), (
                f"{model_id}: exported artifact disagrees with scikit-learn on test row {row}"
            )

        bundle["models"].append(
            {
                "id": model_id,
                "name": name,
                "description": description,
                "metrics": art.classification_metrics(y_test, predictions, average="macro"),
                "confusion": art.confusion(y_test, predictions, len(classes)),
                "perClass": per_class_report(y_test, predictions, classes),
                "topTermsPerClass": top_terms_per_class(weights, terms, classes),
                "explain": {"kind": "tokenWeights"},
                "artifact": exported,
            }
        )
        metrics = bundle["models"][-1]["metrics"]
        print(
            f"    accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1']:.4f} "
            f"weighted_f1={metrics['f1Weighted']:.4f}"
        )

    rng = np.random.default_rng(RANDOM_STATE)
    sample_rows = rng.choice(len(test_plots), size=50, replace=False)
    fixture_cases = []
    for text in [test_plots[int(i)] for i in sample_rows] + SAMPLE_PLOTS:
        vector = vectorizer.transform([text])
        expected = {}
        for model_id, _, _, estimator, output in specs:
            entry = {"predictedIndex": int(estimator.predict(vector)[0])}
            if hasattr(estimator, "predict_proba"):
                entry["probabilities"] = [float(p) for p in estimator.predict_proba(vector)[0]]
            expected[model_id] = entry
        fixture_cases.append({"input": {"text": text}, "expected": expected})

    art.write_bundle(bundle)
    art.write_fixtures("movie-genre", {"task": "movie-genre", "cases": fixture_cases})


if __name__ == "__main__":
    main()
