import type { ModelPrediction, PredictResponse, TaskMeta } from "../../shared/types";
import { decimal, percent, titleCase } from "../format";
import { Meter, SERIES } from "./charts";

/**
 * Churn and spam have a "bad" outcome, so the verdict uses the reserved status
 * colours -- always with an icon and a written label, never colour alone.
 * Genre has no good or bad, so it stays on the neutral accent.
 */
function toneFor(meta: TaskMeta, label: string): "good" | "critical" | "neutral" {
  if (meta.kind !== "binary") return "neutral";
  return label === meta.classes[1] ? "critical" : "good";
}

const TONE_GLYPH = { good: "✓", critical: "!", neutral: "◆" } as const;

function ModelCard({
  prediction,
  meta,
  color,
}: {
  prediction: ModelPrediction;
  meta: TaskMeta;
  color: string;
}) {
  const tone = toneFor(meta, prediction.predictedLabel);
  const multiclass = meta.kind === "multiclass";

  return (
    <div className="model-result">
      <div className="model-result__head">
        <span className="model-result__name">
          <span
            className="legend__key legend__key--square"
            style={{ background: color, display: "inline-block", marginRight: 7 }}
          />
          {prediction.modelName}
        </span>
        <span className="verdict__icon" data-tone={tone} style={{ width: 20, height: 20, fontSize: 11 }}>
          {TONE_GLYPH[tone]}
        </span>
      </div>

      <div>
        <div className="model-result__answer">{titleCase(prediction.predictedLabel)}</div>
        {prediction.confidence !== null ? (
          <>
            <div className="kv" style={{ marginBottom: 6 }}>
              <span>confidence</span>
              <span>{percent(prediction.confidence, 1)}</span>
            </div>
            <Meter value={prediction.confidence} color={color} />
          </>
        ) : (
          <div className="kv">
            <span>margin from boundary</span>
            {/* A binary SVM stores one score describing class 1; a one-vs-rest
                multiclass SVM stores one per class, so pick the winner's. */}
            <span>
              {decimal(
                prediction.scores.length === meta.classes.length
                  ? prediction.scores[prediction.predictedIndex]
                  : prediction.scores[0],
                3,
              )}
            </span>
          </div>
        )}
      </div>

      {multiclass && (
        <div className="stack" style={{ gap: 5 }}>
          {prediction.topClasses.slice(0, 4).map((entry) => (
            <div key={entry.label}>
              <div className="kv">
                <span>{titleCase(entry.label)}</span>
                <span>
                  {entry.probability !== null ? percent(entry.probability, 1) : decimal(entry.score, 2)}
                </span>
              </div>
              {entry.probability !== null && <Meter value={entry.probability} color={color} />}
            </div>
          ))}
        </div>
      )}

      {prediction.confidence === null && (
        <p className="small muted">
          An SVM fitted without probability calibration has no probability to report — the signed
          distance from the decision boundary is the honest equivalent.
        </p>
      )}
    </div>
  );
}

export function Predictions({
  result,
  meta,
}: {
  result: PredictResponse;
  meta: TaskMeta;
}) {
  const unanimous = result.consensus.agreeing === result.consensus.total;
  const tone = toneFor(meta, result.consensus.label);

  return (
    <div className="stack">
      <div className="verdict">
        <span className="verdict__icon" data-tone={tone}>
          {TONE_GLYPH[tone]}
        </span>
        <div>
          <div className="verdict__label">{titleCase(result.consensus.label)}</div>
          <div className="verdict__note">
            {unanimous
              ? `All ${result.consensus.total} models agree`
              : `${result.consensus.agreeing} of ${result.consensus.total} models agree — the rest disagree, so treat this one as uncertain`}
            {" · "}
            scored in {result.elapsedMs}ms
          </div>
        </div>
      </div>

      <div className="grid grid--3">
        {result.models.map((prediction, index) => (
          <ModelCard
            key={prediction.modelId}
            prediction={prediction}
            meta={meta}
            color={SERIES[index % SERIES.length]}
          />
        ))}
      </div>
    </div>
  );
}
