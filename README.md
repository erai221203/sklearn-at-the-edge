# ML Studio — three scikit-learn models, served from Cloudflare

Three classical machine-learning tasks, each compared across three algorithms, wrapped in
a web application that runs entirely on Cloudflare's edge.

| Task | Data | Algorithms compared | Best held-out accuracy |
|---|---|---|---|
| Customer churn | `Churn_Modelling.csv`, 10,000 customers | Logistic Regression · Random Forest · Gradient Boosting | **87.0%** (Gradient Boosting) |
| SMS spam | `spam.csv`, 5,623 messages | Multinomial Naive Bayes · Logistic Regression · Linear SVM | **98.4%** (Linear SVM) |
| Movie genre | IMDb plot summaries, 108,414 films, 27 genres | Multinomial Naive Bayes · Logistic Regression · Linear SVM | **57.9%** (Linear SVM) |

Each task gets its own section. You put in your own example — move the sliders on a customer
profile, paste an SMS, describe a film's plot — and get an answer back straight away.

## The answer comes in two layers

**By default the page answers in plain English**, for a reader who does not care which
algorithm produced it:

> ✓ **Likely to stay** — the models put this at only about 1 in 10 leaving.
> *All 3 methods agree on this answer.*
>
> **What is driving this**
> ▼ They are an active member, which is the strongest single sign a customer will stay.
> ▲ They are female, and women leave more often than men in this data.
> ▼ They do not bank in Germany, which has the highest churn of the three countries.
>
> **What you could do** — no action needed. This customer looks settled.

For the text tasks the giveaway words are highlighted in the message itself, so it is obvious
at a glance what the model reacted to.

**"The technical detail" expands** to the model-by-model breakdown, accuracy/precision/recall,
ROC curves, confusion matrices, threshold sweeps and coefficient charts. Nothing is hidden —
it just is not what greets you.

Some deliberate choices in that plain layer:

- **Frequencies, not just percentages.** "About 1 in 10" is read correctly far more often
  than "10.3%". The exact figure is still shown next to it.
- **Confidence in words.** "Fairly confident" alongside the number, so a percentage never has
  to be interpreted on its own.
- **Disagreement is surfaced, not smoothed over.** When the three methods split, the page says
  "only 2 of the 3 methods gave this answer, so treat it as a maybe rather than a verdict."
- **Bands anchored to the data.** "Worth watching" for churn starts a quarter above the
  dataset's actual churn rate rather than at a round number, so a genuinely above-average
  profile is never filed under "likely to stay".
- **No invented certainty.** The SVM has no calibrated probability, so it is excluded from the
  averaged figure rather than having one guessed for it.
- **"I didn't actually read that."** A bag-of-words model cannot abstain: given a message with
  no words it recognises, it returns whichever class was commonest in training with a
  perfectly confident-looking number. So the response carries how many words were recognised,
  how many were everyday words dropped during training, and how many it had never seen. Type
  `qqzzxx wubbleflarn` and the page says the answer means nothing rather than reporting 89%
  confidence in "Ham".

---

## The interesting problem: scikit-learn does not run on Cloudflare

Cloudflare Workers execute JavaScript on V8 isolates. There is no Python, so a `.pkl` of a
fitted estimator is not deployable. The usual workaround is to host a separate Python API,
but that gives up the thing that makes edge hosting worth having.

This project takes the other route:

```
   ml/*.py                    app/public/models/*.json          app/worker/ml/*.ts
┌──────────────┐             ┌──────────────────────┐        ┌───────────────────────┐
│ scikit-learn │  export ──> │ coefficients, log-   │  read  │ inference re-written  │
│ fits models  │             │ probabilities, tree  │ ─────> │ in TypeScript, runs   │
│              │             │ node tables, idf     │        │ inside the Worker     │
└──────────────┘             └──────────────────────┘        └───────────────────────┘
        │                                                                │
        └──────────── parity fixtures: scikit-learn's own answers ───────┘
                          `app/tests/parity.test.ts`
```

Each estimator is reduced to the numbers its prediction path actually needs, and that maths
is re-implemented in TypeScript. **Nothing is trusted to be equivalent — it is checked.**
The trainers verify their own exports against scikit-learn before writing anything, and the
test suite replays 181 fixtures of scikit-learn's real predictions through the TypeScript
port on every build.

Measured agreement across all nine models:

| Estimator family | Worst probability difference vs scikit-learn |
|---|---|
| Random Forest | 2.2 × 10⁻¹⁶ |
| Gradient Boosting | 8.6 × 10⁻¹² |
| Logistic Regression / Naive Bayes / SVM | 4.5 × 10⁻⁷ |

Tree ensembles are exact. The linear families differ only because their weights are stored
as float32 to halve the artifact size — far below any decision boundary, and every predicted
label matches.

### Two details that are easy to get wrong

Both of these produced real, reproducible disagreements during development, and both are
caught by the test suite if they ever regress:

- **scikit-learn narrows features to `float32` before walking a tree**, and its split
  thresholds are midpoints of adjacent float32 values — some sit only ~2 × 10⁻⁹ away from
  real data points. Comparing in float64, or rounding thresholds to 8 decimals, sends
  samples down the wrong branch. Split points are therefore stored exactly, and the
  TypeScript walk narrows its input the same way scikit-learn does.
- **`TfidfVectorizer`'s default `token_pattern` is `(?u)\b\w\w+\b`**, where Python's `\w`
  covers all Unicode alphanumerics. JavaScript's `\w` is ASCII-only, so the tokenizer spells
  the class out as `[\p{L}\p{N}_]{2,}` — otherwise every accented token in the SMS corpus
  silently disappears.

---

## Running it

```bash
# 1. Train the models (writes app/public/models/*.json)
pip install -r ml/requirements.txt
python ml/fetch_movie_data.py     # downloads the IMDb corpus into data/
python ml/train_all.py

# 2. Run the app
cd app
npm install
npm run dev                       # http://localhost:5173, Worker runs in real workerd
```

The trained artifacts are committed, so if you only want to run the site you can skip
step 1 entirely.

```bash
npm run check      # typecheck both projects + 57 tests
npm run build      # client bundle + Worker
```

---

## Deploying

This is **not a static site**, so it goes to Workers rather than Pages. One Worker serves
both halves:

| Path | Served as |
|---|---|
| `/`, `/churn`, `/spam`, `/movie-genre` | static assets (the SPA shell) |
| `/assets/*`, `/models/*.json` | static assets |
| `/api/*` | Worker code — this is where inference runs |

`run_worker_first: ["/api/*"]` in `wrangler.jsonc` sends API calls to the Worker and lets
everything else be answered straight from the edge, and `not_found_handling:
"single-page-application"` makes `/churn` deep-link instead of 404ing.

The model weights are static files, but the **Worker** reads them through its `ASSETS`
binding and caches them per isolate — the browser downloads ~74 KB of JavaScript, never the
3.6 MB of weights.

### By hand

```bash
cd app
npx wrangler login
npm run deploy          # runs the checks, builds, then deploys
```

That publishes to `https://sklearn-at-the-edge.<your-subdomain>.workers.dev`. To change the
name — and the URL — edit `name` in `wrangler.jsonc`.

Nothing else to configure: no database, no environment variables, no separate backend, and
no Python at request time. The trained artifacts are committed, so a fresh clone deploys
without ever running the trainers.

### From the Cloudflare dashboard (Workers Builds)

If you connect the repository to Cloudflare's own Git integration, set these under
**Workers & Pages → your Worker → Settings → Build**:

| Field | Value |
|---|---|
| **Root directory** | `app` |
| **Build command** | `npm ci && npm run build` |
| **Deploy command** | `npx wrangler deploy` |

**Root directory is the one that matters.** The deployable project is `app/`, not the
repository root. Left at the default, `wrangler deploy` finds no `wrangler.jsonc`, silently
falls back to its non-interactive first-run wizard, invents a config with an `assets`
directory and *no* `main`, and reports success — publishing the raw source tree as static
files with no Worker and no API behind it. A correct run reports roughly **74 KiB uploaded
and 13 asset files**; the broken one reports `0.33 KiB` and 53 files.

A quick check after any deploy:

```bash
curl -s https://<your-worker>.workers.dev/api/health          # {"status":"ok",...}
curl -s -o /dev/null -w '%{http_code}\n' https://<...>/src/main.tsx   # want 404, not 200
```

If `/api/health` 404s, or `/src/main.tsx` returns 200, the build settings are wrong.

### From GitHub

`.github/workflows/deploy.yml` typechecks, tests and builds every push and pull request, and
deploys on a push to `main`. It needs two repository secrets
(**Settings → Secrets and variables → Actions**):

| Secret | Where it comes from |
|---|---|
| `CLOUDFLARE_API_TOKEN` | Cloudflare dashboard → My Profile → API Tokens → *Edit Cloudflare Workers* template |
| `CLOUDFLARE_ACCOUNT_ID` | Workers & Pages overview, right-hand sidebar |

The deploy job only runs if the parity suite passes, so a model export that stops matching
scikit-learn cannot reach production.

### Limits this fits inside

| | This project | Free plan |
|---|---|---|
| Worker script | 74 KB (20 KB gzipped) | 3 MB gzipped |
| Static assets | 13 files, 3.8 MB, largest 2.2 MB | 20,000 files, 25 MB each |
| CPU per request | ~1 ms warm | 10 ms |

---

## How it is put together

```
ml/                        Python: training and artifact export
  artifact.py                serialisation formats (the Python/TypeScript contract)
  reference.py               NumPy reader used to verify every export
  train_churn.py             ┐
  train_spam.py              ├ one per task; each verifies before it writes
  train_movie_genre.py       ┘
  train_all.py               retrain everything

app/
  shared/types.ts          the API + artifact contract, used by Worker and browser
  worker/
    index.ts                 Hono routes
    artifacts.ts             loads artifacts from static assets, cached per isolate
    predict.ts               request validation, response shape
    ml/                      the inference port: tf-idf, linear, naive bayes, trees
  src/                     React frontend
  public/models/           the deployed model artifacts
  tests/                   parity + API suites

legacy/                  the original task scripts, kept for reference
Churn_Modelling.csv      ┐ source data, read by the trainers
spam.csv                 ┘
data/                    the IMDb corpus, downloaded rather than committed
```

### Artifacts are split, and large arrays are packed

Each task ships as `<task>.meta.json` (metrics, input spec, charts — tens of KB) and
`<task>.weights.json` (the actual numbers). The catalogue and metrics screens never touch
the weights, and a task's weights load once per isolate on its first prediction.

Arrays above 4,096 elements are base64-encoded typed arrays rather than JSON numbers. The
movie-genre model is a 27 × 5,000 matrix per algorithm; parsing that as JSON costs tens of
milliseconds on an isolate's first request, which is most of the Workers free-tier CPU
budget. Decoding base64 into a `Float32Array` is a linear byte copy instead. Warm requests
answer in about 9 ms end to end.

### Routing

`wrangler.jsonc` sets `run_worker_first: ["/api/*"]` so API calls reach the Worker, and
`not_found_handling: "single-page-application"` so `/churn` and `/spam` deep-link correctly
instead of 404ing.

| Route | |
|---|---|
| `GET /api/health` | liveness + task list |
| `GET /api/models` | all tasks with metrics, no weights |
| `GET /api/models/:task` | full metadata: input spec, curves, confusion matrices |
| `POST /api/predict/:task` | all three models score one input |

---

## What changed from the original scripts

The three original scripts are kept in `legacy/` for reference. The trainers in `ml/` grew
out of them, with these corrections:

**Movie genre** — the original wrapped a `MultiLabelBinarizer` and `MultiOutputClassifier`
around a dataset where every film carries exactly one genre. Twenty-seven independent binary
classifiers then had to agree unanimously to produce a correct label set, and most rows came
back empty. It also scored the model on its own training data. Treating the problem as
single-label multiclass and evaluating on the held-out `test_data_solution.txt` takes it
from a reported 27.7% (training accuracy) to **57.9% on 54,200 unseen films** — against a
25.1% majority-class baseline.

**Churn** — `Geography` and `Gender` were dropped by the original; German customers churn at
roughly twice the French rate, so both are one-hot encoded and kept. The tree ensembles are
depth-limited, which regularises them and shrinks the serialised forest from 5.2 MB to 1 MB.
Accuracy goes from 0.8095 / 0.8585 / 0.8625 to **0.808 / 0.865 / 0.870**, and the split is
now stratified with ROC-AUC and threshold sweeps reported alongside accuracy — on a dataset
that is 80% non-churners, accuracy alone flatters every model.

**Spam** — same three algorithms, plus a stratified split, ranking metrics, and the
token-level weights the UI uses to show which words triggered a decision.

### On the SVM's "confidence"

`SVC` fitted without `probability=True` has no calibrated probability. Rather than invent
one, the artifact carries `output: "margin"` and the UI shows the signed distance from the
decision boundary, labelled as such. It is the honest answer to a question the model cannot
be asked.
