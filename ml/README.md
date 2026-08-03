# Training pipeline

Fits the models and exports them in a form the Cloudflare Worker can execute. See the
[root README](../README.md) for why the models are ported rather than served from Python.

## Running

```bash
pip install -r requirements.txt
python fetch_movie_data.py      # ~70 MB into ../data/, needed only for movie genre
python train_all.py             # or: python train_all.py churn spam
```

Outputs:

- `../app/public/models/<task>.meta.json` — metrics, curves, the input spec the frontend
  renders, and the per-class breakdowns. Small enough to serve on every page load.
- `../app/public/models/<task>.weights.json` — coefficients, log-probabilities and tree
  node tables. Loaded only when a prediction is requested.
- `../app/tests/fixtures/<task>.parity.json` — raw inputs paired with scikit-learn's own
  answers, which the TypeScript test suite is checked against.

Commit all three. The app deploys from the artifacts, so a fresh clone can build and ship
without Python installed.

## Files

| File | |
|---|---|
| `artifact.py` | Export formats — the contract with `app/worker/ml/`. Anything added here needs a matching reader there and a fixture in the parity suite. |
| `reference.py` | A NumPy reader for the exported format. Every trainer runs it against its own estimator before writing, so a broken export fails the training run rather than production. |
| `train_churn.py` | Tabular, binary. Logistic Regression · Random Forest · Gradient Boosting. |
| `train_spam.py` | TF-IDF, binary. Naive Bayes · Logistic Regression · SVC(linear). |
| `train_movie_genre.py` | TF-IDF, 27-class. Naive Bayes · Logistic Regression · LinearSVC. |
| `fetch_movie_data.py` | Downloads the IMDb corpus, which is too large to commit. |

## Adding an estimator

1. Write an `export_*` function in `artifact.py` producing a JSON-serialisable dict with a
   `type` field.
2. Teach `reference.LoadedModel` to read it.
3. Assert in the trainer that the reader reproduces the estimator's own predictions — copy
   the verification loop from any existing trainer.
4. Port it to `app/worker/ml/models.ts` and add it to `loadModel`.
5. Emit fixtures for it. `app/tests/parity.test.ts` picks them up automatically.

Steps 3 and 5 are the point. Without them a subtle numerical difference — a rounded
threshold, a float32 cast, a tokenizer that drops accented words — reaches users as a wrong
answer with no error anywhere.

## Determinism

Every split and estimator is seeded with `random_state=42`, so retraining reproduces the
published metrics exactly. Weights are rounded before serialisation (six decimals, or
float32 for large arrays); the tolerance this costs is documented and asserted in
`app/tests/parity.test.ts`. Tree split points and leaf values are never rounded — see the
comment in `_pack_trees`.
