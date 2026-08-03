import { useMemo, useState } from "react";

import type { ModelMeta, TabularInputSpec, TaskMeta } from "../../shared/types";
import { count, decimal, titleCase } from "../format";
import { BarChart, ConfusionMatrix, Figure, Legend, SERIES, UnitLineChart } from "./charts";

function MetricsTable({ meta }: { meta: TaskMeta }) {
  const columns: { key: keyof ModelMeta["metrics"]; label: string }[] = [
    { key: "accuracy", label: "Accuracy" },
    { key: "precision", label: "Precision" },
    { key: "recall", label: "Recall" },
    { key: "f1", label: "F1" },
  ];
  if (meta.models.some((model) => model.metrics.rocAuc !== undefined)) {
    columns.push({ key: "rocAuc", label: "ROC AUC" });
  }
  if (meta.models.some((model) => model.metrics.f1Weighted !== undefined)) {
    columns.push({ key: "f1Weighted", label: "Weighted F1" });
  }

  const best = new Map<string, number>();
  for (const column of columns) {
    best.set(
      column.key,
      Math.max(...meta.models.map((model) => model.metrics[column.key] ?? -Infinity)),
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {meta.models.map((model, index) => (
            <tr key={model.id}>
              <td>
                <span
                  className="legend__key legend__key--square"
                  style={{
                    background: SERIES[index % SERIES.length],
                    display: "inline-block",
                    marginRight: 8,
                  }}
                />
                {model.name}
              </td>
              {columns.map((column) => {
                const value = model.metrics[column.key];
                if (value === undefined) return <td key={column.key} className="muted">—</td>;
                return (
                  <td key={column.key} className={value === best.get(column.key) ? "is-best" : undefined}>
                    {decimal(value, 3)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RocFigure({ meta }: { meta: TaskMeta }) {
  const series = meta.models
    .map((model, index) => ({
      name: model.name,
      color: SERIES[index % SERIES.length],
      points: (model.rocCurve ?? []).map((point) => ({ x: point.fpr, y: point.tpr })),
      auc: model.metrics.rocAuc,
    }))
    .filter((entry) => entry.points.length > 0);

  if (series.length === 0) return null;

  return (
    <Figure
      title="ROC curves"
      note="How well each model ranks positives above negatives, across every possible threshold. Further from the diagonal is better."
      legend={
        <Legend
          items={series.map((entry) => ({
            label: `${entry.name} · AUC ${decimal(entry.auc ?? 0, 3)}`,
            color: entry.color,
          }))}
        />
      }
      table={{
        head: ["Model", "ROC AUC"],
        rows: series.map((entry) => [entry.name, decimal(entry.auc ?? 0, 4)]),
      }}
    >
      <UnitLineChart
        series={series}
        xLabel="False positive rate"
        yLabel="True positive rate"
        reference="random guessing"
      />
    </Figure>
  );
}

function ThresholdFigure({ model }: { model: ModelMeta }) {
  const sweep = model.thresholdSweep ?? [];
  if (sweep.length === 0) return null;
  const series = [
    { name: "Precision", color: SERIES[0], points: sweep.map((p) => ({ x: p.threshold, y: p.precision })) },
    { name: "Recall", color: SERIES[1], points: sweep.map((p) => ({ x: p.threshold, y: p.recall })) },
    { name: "F1", color: SERIES[2], points: sweep.map((p) => ({ x: p.threshold, y: p.f1 })) },
  ];
  const bestF1 = sweep.reduce((best, point) => (point.f1 > best.f1 ? point : best));

  return (
    <Figure
      title="Precision, recall and F1 by threshold"
      note={`The default cut-off is 0.50. F1 peaks at ${decimal(bestF1.threshold, 2)} on this model, where precision is ${decimal(bestF1.precision, 2)} and recall ${decimal(bestF1.recall, 2)}.`}
      legend={<Legend items={series.map((s) => ({ label: s.name, color: s.color }))} />}
      table={{
        head: ["Threshold", "Precision", "Recall", "F1"],
        rows: sweep
          .filter((_, index) => index % 4 === 0)
          .map((point) => [
            decimal(point.threshold, 2),
            decimal(point.precision, 3),
            decimal(point.recall, 3),
            decimal(point.f1, 3),
          ]),
      }}
    >
      <UnitLineChart series={series} xLabel="Decision threshold" yLabel="Score" />
    </Figure>
  );
}

function ExplainCard({ meta, model }: { meta: TaskMeta; model: ModelMeta }) {
  const figure = explainFigure(meta, model);
  if (!figure) return null;
  return (
    <div className="card" style={{ marginTop: 18 }}>
      <div className="card__body">{figure}</div>
    </div>
  );
}

function explainFigure(meta: TaskMeta, model: ModelMeta) {
  if (meta.input.kind === "tabular" && model.explain.values) {
    const labels = (meta.input as TabularInputSpec).featureLabels;
    const diverging = model.explain.kind === "coefficients";
    const rows = model.explain.values
      .map((value, index) => ({ label: labels[index] ?? `#${index}`, value }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    return (
      <Figure
        title={diverging ? "What the model weighs, and which way" : "Which inputs the model relies on"}
        note={
          diverging
            ? "Logistic-regression coefficients on standardised inputs. Right pushes towards churn, left towards staying."
            : "Impurity-based feature importance: how much each input contributes to the splits. Magnitude only — it does not say which direction."
        }
        legend={
          diverging ? (
            <Legend
              square
              items={[
                { label: "pushes towards churn", color: "var(--diverge-pos)" },
                { label: "pushes towards staying", color: "var(--diverge-neg)" },
              ]}
            />
          ) : undefined
        }
        table={{
          head: ["Feature", diverging ? "Coefficient" : "Importance"],
          rows: rows.map((row) => [row.label, decimal(row.value, 4)]),
        }}
      >
        <BarChart rows={rows} diverging={diverging} format={(value) => decimal(value, 3)} labelWidth={170} />
      </Figure>
    );
  }

  if (model.topTokens) {
    const rows = [
      ...model.topTokens.spam.slice(0, 12),
      ...model.topTokens.ham.slice(0, 12),
    ]
      .map((token) => ({ label: token.term, value: token.weight }))
      .sort((a, b) => b.value - a.value);

    return (
      <Figure
        title="The words that decide it"
        note={`The vocabulary this model weighs most heavily in each direction, learned from ${count(meta.dataset.trainRows)} training messages.`}
        legend={
          <Legend
            square
            items={[
              { label: "evidence for spam", color: "var(--diverge-pos)" },
              { label: "evidence for ham", color: "var(--diverge-neg)" },
            ]}
          />
        }
        table={{
          head: ["Term", "Weight"],
          rows: rows.map((row) => [row.label, decimal(row.value, 3)]),
        }}
      >
        <BarChart rows={rows} diverging format={(value) => decimal(value, 2)} labelWidth={110} />
      </Figure>
    );
  }

  if (model.topTermsPerClass) return <TopTermsPerGenre model={model} />;

  return null;
}

/** One genre at a time: with 27 of them, all at once is a wall of words. */
function TopTermsPerGenre({ model }: { model: ModelMeta }) {
  const groups = model.topTermsPerClass ?? [];
  const ordered = useMemo(() => {
    const support = new Map((model.perClass ?? []).map((row) => [row.genre, row.support]));
    return [...groups].sort((a, b) => (support.get(b.genre) ?? 0) - (support.get(a.genre) ?? 0));
  }, [groups, model.perClass]);
  const [genre, setGenre] = useState(ordered[0]?.genre ?? "");
  const selected = ordered.find((group) => group.genre === genre) ?? ordered[0];
  if (!selected) return null;

  return (
    <Figure
      title="The vocabulary behind each genre"
      note="The terms this model weighs most heavily when deciding a plot belongs to the selected genre."
      table={{
        head: ["Term", "Weight"],
        rows: selected.terms.map((term) => [term.term, decimal(term.weight, 4)]),
      }}
    >
      <div className="row" style={{ marginBottom: 4 }}>
        <label className="small muted" htmlFor="genre-terms">
          Genre
        </label>
        <select
          id="genre-terms"
          value={selected.genre}
          onChange={(event) => setGenre(event.target.value)}
          style={{ width: "auto", minWidth: 180 }}
        >
          {ordered.map((group) => (
            <option key={group.genre} value={group.genre}>
              {titleCase(group.genre)}
            </option>
          ))}
        </select>
      </div>
      <BarChart
        rows={selected.terms.map((term) => ({ label: term.term, value: term.weight }))}
        format={(value) => decimal(value, 2)}
        labelWidth={120}
      />
    </Figure>
  );
}

function PerClassFigure({ model }: { model: ModelMeta }) {
  const rows = model.perClass ?? [];
  if (rows.length === 0) return null;
  return (
    <Figure
      title="F1 by genre"
      note="Ordered by how many test films carry each genre. The long tail is where a bag-of-words model runs out of signal — rare genres have too few examples to learn a distinctive vocabulary."
      table={{
        head: ["Genre", "Precision", "Recall", "F1", "Test films"],
        rows: rows.map((row) => [
          titleCase(row.genre),
          decimal(row.precision, 3),
          decimal(row.recall, 3),
          decimal(row.f1, 3),
          count(row.support),
        ]),
      }}
    >
      <BarChart
        rows={rows.map((row) => ({
          label: titleCase(row.genre),
          value: row.f1,
          detail: `${count(row.support)} test films · precision ${decimal(row.precision, 2)} · recall ${decimal(row.recall, 2)}`,
        }))}
        format={(value) => decimal(value, 2)}
        labelWidth={110}
      />
    </Figure>
  );
}

function ConfusionFigure({ meta, model }: { meta: TaskMeta; model: ModelMeta }) {
  const confusions = useMemo(() => {
    const entries: { label: string; value: number }[] = [];
    model.confusion.forEach((row, actual) => {
      row.forEach((value, predicted) => {
        if (actual === predicted || value === 0) return;
        entries.push({
          label: `${titleCase(meta.classes[actual])} → ${titleCase(meta.classes[predicted])}`,
          value,
        });
      });
    });
    return entries.sort((a, b) => b.value - a.value).slice(0, 12);
  }, [meta.classes, model.confusion]);

  if (meta.kind === "binary") {
    return (
      <Figure
        title="Confusion matrix"
        note="Rows are what actually happened, columns are what the model said. The off-diagonal cells are the mistakes."
        table={{
          head: ["Actual", ...meta.classes.map((c) => `Predicted ${c}`)],
          rows: model.confusion.map((row, index) => [meta.classes[index], ...row.map(count)]),
        }}
      >
        <ConfusionMatrix matrix={model.confusion} labels={meta.classes} />
      </Figure>
    );
  }

  return (
    <Figure
      title="Most common mistakes"
      note="With 27 genres a full matrix is unreadable, so these are the twelve largest off-diagonal cells: what the film was, and what the model called it."
      table={{
        head: ["Mistake", "Films"],
        rows: confusions.map((row) => [row.label, count(row.value)]),
      }}
    >
      <BarChart
        rows={confusions}
        format={(value) => count(value)}
        labelWidth={190}
      />
    </Figure>
  );
}

export function ModelReport({ meta }: { meta: TaskMeta }) {
  const [selectedId, setSelectedId] = useState(meta.models[0].id);
  const selected = meta.models.find((model) => model.id === selectedId) ?? meta.models[0];

  return (
    <>
      <section className="section">
        <div className="section__head">
          <h2>How the three models compare</h2>
          <span className="small muted">
            {count(meta.dataset.testRows)} held-out rows · best value in each column is bold
          </span>
        </div>
        <div className="card">
          <div className="card__body stack">
            <MetricsTable meta={meta} />
            <p className="small muted">
              {meta.kind === "binary"
                ? "Precision and recall are reported for the positive class. On an imbalanced dataset accuracy alone is misleading — a model that never predicts the positive class still scores well."
                : "Precision, recall and F1 are macro-averaged, so every genre counts equally regardless of how many films it has. Weighted F1 counts films instead."}
            </p>
          </div>
        </div>
      </section>

      {meta.kind === "binary" && (
        <section className="section">
          <div className="card">
            <div className="card__body">
              <RocFigure meta={meta} />
            </div>
          </div>
        </section>
      )}

      <section className="section">
        <div className="section__head">
          <h2>Inside one model</h2>
          <div className="switch" role="group" aria-label="Select a model to inspect">
            {meta.models.map((model) => (
              <button
                key={model.id}
                aria-pressed={model.id === selectedId}
                onClick={() => setSelectedId(model.id)}
              >
                {model.name}
              </button>
            ))}
          </div>
        </div>

        <div className="card" style={{ marginBottom: 18 }}>
          <div className="card__body">
            <p className="small">{selected.description}</p>
          </div>
        </div>

        <div className="grid grid--2">
          <div className="card">
            <div className="card__body">
              <ConfusionFigure meta={meta} model={selected} />
            </div>
          </div>
          <div className="card">
            <div className="card__body">
              {meta.kind === "binary" ? (
                <ThresholdFigure model={selected} />
              ) : (
                <PerClassFigure model={selected} />
              )}
            </div>
          </div>
        </div>

        <ExplainCard meta={meta} model={selected} />
      </section>
    </>
  );
}
