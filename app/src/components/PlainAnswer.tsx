/**
 * The answer, for someone who does not care which algorithm produced it.
 *
 * This is the default view of every task. It says what the model concluded,
 * how sure it is in words, what drove that conclusion in sentences, and — for
 * churn — what to do about it. The model-by-model breakdown, the metrics and
 * the charts all live behind "the technical detail" below.
 */

import type { PredictResponse, TaskMeta } from "../../shared/types";
import { percent } from "../format";
import {
  agreementSentence,
  capitalise,
  churnAdvice,
  churnReasons,
  confidenceWord,
  evidenceLevel,
  evidenceNote,
  explainingModel,
  listWords,
  representativeModel,
  verdictFor,
  type InputValues,
  type Tone,
} from "../plain";

const TONE_GLYPH: Record<Tone, string> = {
  good: "✓",
  warning: "!",
  serious: "!",
  critical: "!",
  neutral: "★",
};

function HighlightedMessage({
  text,
  towards,
  against,
}: {
  text: string;
  towards: string[];
  against: string[];
}) {
  const up = new Set(towards);
  const down = new Set(against);
  // Splitting on a capturing group keeps the punctuation and spacing intact,
  // so the message reads exactly as it was typed.
  const parts = text.split(/([\p{L}\p{N}_]+)/gu);

  return (
    <p className="quoted">
      {parts.map((part, index) => {
        const word = part.toLowerCase();
        if (up.has(word)) {
          return (
            <mark key={index} className="mark mark--up">
              {part}
            </mark>
          );
        }
        if (down.has(word)) {
          return (
            <mark key={index} className="mark mark--down">
              {part}
            </mark>
          );
        }
        return <span key={index}>{part}</span>;
      })}
    </p>
  );
}

export function PlainAnswer({
  meta,
  result,
  values,
  inputText,
}: {
  meta: TaskMeta;
  result: PredictResponse;
  values: InputValues;
  inputText?: string;
}) {
  const verdict = verdictFor(meta, result);
  const agreement = agreementSentence(result);

  // Headline, ranking and highlights all have to speak for the same model as
  // the answer above them -- see `representativeModel`.
  const speaker = representativeModel(result);
  const contributions = explainingModel(result)?.contributions ?? [];

  // A bag-of-words model given nothing it knows still answers confidently.
  const level = evidenceLevel(result.evidence);
  const commonest = Object.entries(meta.dataset.classBalance).reduce((best, entry) =>
    entry[1] > best[1] ? entry : best,
  )[0];
  const note = result.evidence ? evidenceNote(result.evidence, level, commonest) : null;

  const reasons = meta.task === "churn" ? churnReasons(contributions, values) : [];
  const towards = contributions.filter((c) => c.value > 0).map((c) => c.label);
  const against = contributions.filter((c) => c.value < 0).map((c) => c.label);

  return (
    <div className="stack">
      <section className={`answer answer--${verdict.tone}`}>
        <div className="answer__top">
          <span className="answer__icon" data-tone={verdict.tone}>
            {TONE_GLYPH[verdict.tone]}
          </span>
          <div>
            <h2 className="answer__headline">{verdict.headline}</h2>
            {verdict.detail && level !== "none" && (
              <p className="answer__detail">{verdict.detail}</p>
            )}
          </div>
        </div>

        {verdict.likelihood !== null && (
          <div className="answer__gauge">
            <div className="answer__scale" aria-hidden="true">
              <div
                className="answer__scale-fill"
                data-tone={verdict.tone}
                style={{ width: `${Math.round(verdict.likelihood * 100)}%` }}
              />
            </div>
            <div className="answer__gauge-legend">
              <span>
                {meta.task === "churn"
                  ? "Chance of leaving"
                  : meta.task === "spam"
                    ? "Chance it is spam"
                    : "Score for this genre"}
                : <strong>{percent(verdict.likelihood, 0)}</strong>
              </span>
              <span className="muted">{confidenceWord(verdict.likelihood, level)}</span>
            </div>
          </div>
        )}

        {note && <p className="answer__caveat">{note}</p>}

        <p
          className={`answer__agreement answer__agreement--${
            level === "none" ? "warning" : agreement.tone
          }`}
        >
          {level === "none"
            ? "All three methods fall back to the same default when there is nothing to read, so their agreement here means nothing."
            : agreement.text}
        </p>
      </section>

      {meta.task === "movie-genre" && (
        <section className="card">
          <div className="card__body">
            <h3 className="card__title">Other genres it could be</h3>
            <p className="card__note">
              There are 27 genres to choose from, so even a good guess rarely scores above
              a third. What matters is which genres rise to the top, not the size of the number.
            </p>
            <div className="ranked">
              {(speaker?.topClasses ?? [])
                .slice(0, 4)
                .map((entry, index) => (
                  <div className="ranked__row" key={entry.label}>
                    <span className="ranked__rank">{index + 1}</span>
                    <span className="ranked__label">{capitalise(entry.label)}</span>
                    <span className="ranked__bar" aria-hidden="true">
                      <span
                        className="ranked__fill"
                        style={{ width: `${Math.round((entry.probability ?? 0) * 100)}%` }}
                      />
                    </span>
                    <span className="ranked__value">
                      {entry.probability !== null ? percent(entry.probability, 0) : "—"}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </section>
      )}

      {reasons.length > 0 && (
        <section className="card">
          <div className="card__body">
            <h3 className="card__title">What is driving this</h3>
            <p className="card__note">Strongest factors first.</p>
            <ul className="reasons">
              {reasons.map((reason, index) => (
                <li key={index} className={reason.raises ? "reason reason--up" : "reason reason--down"}>
                  <span className="reason__mark" aria-hidden="true">
                    {reason.raises ? "▲" : "▼"}
                  </span>
                  <span>
                    {reason.text}
                    <span className="reason__tag">
                      {reason.raises ? "raises risk" : "lowers risk"}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {meta.input.kind === "text" && inputText && result.evidence && (
        <section className="card">
          <div className="card__body">
            <h3 className="card__title">
              {towards.length > 0 || against.length > 0
                ? "Which words gave it away"
                : "What the models could read"}
            </h3>
            <p className="card__note">
              {towards.length > 0 || against.length > 0 ? (
                <>
                  Highlighted in the message below.{" "}
                  {meta.task === "spam"
                    ? "Red words are common in scam texts; blue words are common in ordinary ones."
                    : `Red words point to ${result.consensus.label}; blue words point elsewhere.`}
                </>
              ) : (
                "Nothing in this message is highlighted, because none of it was recognised."
              )}
            </p>
            <HighlightedMessage text={inputText} towards={towards} against={against} />
            {towards.length > 0 && (
              <p className="small" style={{ marginTop: 12 }}>
                {listWords(towards.slice(0, 5))}{" "}
                {towards.length === 1 ? "is the word" : "are the words"} pushing hardest towards{" "}
                <strong>{result.consensus.label}</strong>.
              </p>
            )}
            {result.evidence && (
              <p className="small muted" style={{ marginTop: 8 }}>
                {result.evidence.recognised.length} of {result.evidence.total} words counted.
                {result.evidence.ignored.length > 0 &&
                  ` ${listWords(result.evidence.ignored.slice(0, 6))} ${
                    result.evidence.ignored.length === 1 ? "is an" : "are"
                  } everyday ${result.evidence.ignored.length === 1 ? "word" : "words"} the models ignore on purpose.`}
                {result.evidence.unknown.length > 0 &&
                  ` ${listWords(result.evidence.unknown.slice(0, 6))} never appeared in the training data, so ${
                    result.evidence.unknown.length === 1 ? "it is" : "they are"
                  } invisible to them.`}
              </p>
            )}
          </div>
        </section>
      )}

      {meta.task === "churn" && verdict.likelihood !== null && (
        <section className="card">
          <div className="card__body">
            <h3 className="card__title">What you could do</h3>
            <p style={{ marginTop: 6 }}>{churnAdvice(verdict.likelihood)}</p>
          </div>
        </section>
      )}
    </div>
  );
}
