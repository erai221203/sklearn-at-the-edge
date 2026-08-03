import { useCallback, useEffect, useRef, useState } from "react";

import type { PredictResponse, TaskMeta } from "../../shared/types";
import { fetchTask, predict } from "../api";
import { BarChart, Figure, Legend, Stat } from "../components/charts";
import { defaultsFor, InputPanel, type InputValues } from "../components/InputPanel";
import { ModelReport } from "../components/ModelReport";
import { PlainAnswer } from "../components/PlainAnswer";
import { Predictions } from "../components/Predictions";
import { count, decimal, percent } from "../format";

/** Says what the page does, without naming an algorithm. */
const PLAIN_INTRO: Record<string, string> = {
  churn:
    "Describe a bank customer and find out whether they are likely to close their account — and which parts of their profile are driving that.",
  spam: "Paste a text message and find out whether it looks like a scam, and which words gave it away.",
  "movie-genre":
    "Describe what happens in a film and find out which of 27 genres it sounds like, and which words point that way.",
};

/** Contributions are signed, so they take the diverging pair rather than a series hue. */
function WhyFigures({ result, meta }: { result: PredictResponse; meta: TaskMeta }) {
  const explained = result.models.filter((model) => model.contributions.length > 0);
  if (explained.length === 0) return null;

  const towards = meta.kind === "binary" ? meta.classes[1] : "the predicted genre";
  const away = meta.kind === "binary" ? meta.classes[0] : "other genres";

  return (
    <section className="section">
      <div className="section__head">
        <h2>Why this answer</h2>
        <span className="small muted">
          Contribution of each input to the score the model actually computed
        </span>
      </div>
      <div className="grid grid--2">
        {explained.map((model) => (
          <div className="card" key={model.modelId}>
            <div className="card__body">
              <Figure
                title={model.modelName}
                note={`Said ${model.predictedLabel}${
                  model.confidence !== null ? ` at ${percent(model.confidence, 1)} confidence` : ""
                }.`}
                legend={
                  <Legend
                    square
                    items={[
                      { label: `towards ${towards}`, color: "var(--diverge-pos)" },
                      { label: `towards ${away}`, color: "var(--diverge-neg)" },
                    ]}
                  />
                }
                table={{
                  head: ["Input", "Contribution"],
                  rows: model.contributions.map((entry) => [entry.label, decimal(entry.value, 4)]),
                }}
              >
                <BarChart
                  rows={model.contributions.map((entry) => ({
                    label: entry.label,
                    value: entry.value,
                  }))}
                  diverging
                  format={(value) => decimal(value, 3)}
                  labelWidth={meta.input.kind === "text" ? 110 : 170}
                />
              </Figure>
            </div>
          </div>
        ))}
      </div>
      {meta.input.kind === "text" && (
        <p className="small muted" style={{ marginTop: 10 }}>
          Only words that appear in the training vocabulary carry weight. Anything else — names,
          typos, rare words — is invisible to the model.
        </p>
      )}
    </section>
  );
}

function DatasetCard({ meta }: { meta: TaskMeta }) {
  const balance = Object.entries(meta.dataset.classBalance);
  const total = balance.reduce((sum, [, value]) => sum + value, 0);
  const majority = balance.reduce((best, entry) => (entry[1] > best[1] ? entry : best));

  return (
    <div className="card">
      <div className="card__body stack">
        <div className="stat-row">
          <Stat label="Training rows" value={count(meta.dataset.trainRows)} />
          <Stat label="Test rows" value={count(meta.dataset.testRows)} foot="never seen in fitting" />
          <Stat label="Features" value={count(meta.dataset.features)} />
          <Stat
            label="Majority baseline"
            value={percent(majority[1] / total, 1)}
            foot={`always guess "${majority[0]}"`}
          />
        </div>
        <p className="small muted">
          Source: {meta.dataset.name}. Trained with scikit-learn {meta.sklearnVersion} on{" "}
          {new Date(meta.generatedAt).toLocaleDateString("en-GB", {
            day: "numeric",
            month: "short",
            year: "numeric",
          })}
          .
        </p>
      </div>
    </div>
  );
}

export function TaskView({ task }: { task: string }) {
  const [meta, setMeta] = useState<TaskMeta | null>(null);
  const [values, setValues] = useState<InputValues>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  useEffect(() => {
    let cancelled = false;
    fetchTask(task)
      .then((loaded) => {
        if (cancelled) return;
        setMeta(loaded);
        setValues(defaultsFor(loaded.input));
      })
      .catch((cause: Error) => !cancelled && setError(cause.message));
    return () => {
      cancelled = true;
    };
  }, [task]);

  const run = useCallback(
    async (payload: InputValues) => {
      const id = requestId.current + 1;
      requestId.current = id;
      setBusy(true);
      try {
        const response = await predict(task, payload);
        // Ignore a slow response that a newer request has already superseded.
        if (requestId.current === id) {
          setResult(response);
          setError(null);
        }
      } catch (cause) {
        if (requestId.current === id) setError((cause as Error).message);
      } finally {
        if (requestId.current === id) setBusy(false);
      }
    },
    [task],
  );

  // Tabular inputs rescore as the sliders move; free text waits for a submit,
  // and only gets one automatic run so the page is not empty on arrival.
  const live = meta?.input.kind === "tabular";
  const primed = useRef(false);
  useEffect(() => {
    if (!meta || Object.keys(values).length === 0) return;
    if (!live) {
      if (!primed.current) {
        primed.current = true;
        void run(values);
      }
      return;
    }
    const timer = setTimeout(() => void run(values), 220);
    return () => clearTimeout(timer);
  }, [meta, values, live, run]);

  if (error && !meta) {
    return (
      <div className="notice" data-tone="error">
        Could not load this task: {error}
      </div>
    );
  }

  if (!meta) {
    return (
      <div className="stack">
        <div className="skeleton" style={{ height: 120 }} />
        <div className="skeleton" style={{ height: 380 }} />
      </div>
    );
  }

  return (
    <>
      <div className="page-head">
        <h1>{meta.title}</h1>
        <p>{PLAIN_INTRO[meta.task] ?? meta.subtitle}</p>
      </div>

      <section className="section" style={{ marginTop: 0 }}>
        <div className="split">
          <InputPanel
            spec={meta.input}
            values={values}
            onChange={setValues}
            onSubmit={(payload) => void run(payload ?? values)}
            busy={busy}
          />
          <div style={{ opacity: busy && result ? 0.6 : 1, transition: "opacity 0.15s ease" }}>
            {error && (
              <div className="notice" data-tone="error" style={{ marginBottom: 12 }}>
                {error}
              </div>
            )}
            {result ? (
              <PlainAnswer
                meta={meta}
                result={result}
                values={values}
                inputText={typeof values.text === "string" ? values.text : undefined}
              />
            ) : (
              <div className="skeleton" style={{ height: 300 }} />
            )}
          </div>
        </div>
      </section>

      <details className="disclosure">
        <summary>
          The technical detail
          <span className="disclosure__note">
            how the three methods were built, and how well each one scores
          </span>
        </summary>

        <p className="small muted" style={{ margin: "10px 0 4px", maxWidth: "72ch" }}>
          Three different algorithms are trained on the same data and asked the same question.
          They usually agree; where they do not, the answer above says so. Everything below is
          measured on {count(meta.dataset.testRows)} examples the models never saw while learning.
        </p>

        <DatasetCard meta={meta} />

        {result && (
          <section className="section">
            <div className="section__head">
              <h2>What each method said</h2>
            </div>
            <Predictions result={result} meta={meta} />
          </section>
        )}

        {result && <WhyFigures result={result} meta={meta} />}

        <ModelReport meta={meta} />
      </details>
    </>
  );
}
