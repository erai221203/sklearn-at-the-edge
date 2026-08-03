import { useState } from "react";

import type { FieldSpec, InputSpec } from "../../shared/types";
import { count } from "../format";

export type InputValues = Record<string, string | number>;

export function defaultsFor(spec: InputSpec): InputValues {
  if (spec.kind === "text") return { text: spec.samples[0] ?? "" };
  const values: InputValues = {};
  for (const field of spec.fields) values[field.name] = field.default;
  return values;
}

function formatFieldValue(field: FieldSpec, value: string | number): string {
  if (field.type !== "number") return String(value);
  const numeric = Number(value);
  return numeric >= 1000 ? count(Math.round(numeric)) : String(numeric);
}

function Field({
  field,
  value,
  onChange,
}: {
  field: FieldSpec;
  value: string | number;
  onChange: (next: string | number) => void;
}) {
  const id = `field-${field.name}`;
  const labelId = `${id}-label`;
  const helpId = field.help ? `${id}-help` : undefined;
  // A boolean is a pair of buttons rather than one control, so there is no
  // element for `htmlFor` to point at; the group is labelled by id instead.
  const grouped = field.type === "boolean";

  return (
    <div className="field">
      <label className="field__label" id={labelId} htmlFor={grouped ? undefined : id}>
        <span>{field.label}</span>
        {field.type === "number" && (
          <span className="field__value">{formatFieldValue(field, value)}</span>
        )}
      </label>

      {field.type === "number" && (
        <input
          id={id}
          type="range"
          min={field.min}
          max={field.max}
          step={field.step}
          value={Number(value)}
          aria-describedby={helpId}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      )}

      {field.type === "boolean" && (
        <div className="switch" role="group" aria-labelledby={labelId} aria-describedby={helpId}>
          <button type="button" aria-pressed={Number(value) === 0} onClick={() => onChange(0)}>
            No
          </button>
          <button type="button" aria-pressed={Number(value) === 1} onClick={() => onChange(1)}>
            Yes
          </button>
        </div>
      )}

      {field.type === "category" && (
        <select
          id={id}
          value={String(value)}
          aria-describedby={helpId}
          onChange={(event) => onChange(event.target.value)}
        >
          {field.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      )}

      {field.help && (
        <span className="field__help" id={helpId}>
          {field.help}
        </span>
      )}
    </div>
  );
}

export function InputPanel({
  spec,
  values,
  onChange,
  onSubmit,
  busy,
}: {
  spec: InputSpec;
  values: InputValues;
  onChange: (next: InputValues) => void;
  /** Scores `payload` if given, otherwise whatever is currently in the form. */
  onSubmit: (payload?: InputValues) => void;
  busy: boolean;
}) {
  const [dirty, setDirty] = useState(false);

  if (spec.kind === "text") {
    return (
      <div className="card">
        <div className="card__head">
          <div className="card__title">{spec.label}</div>
          <div className="card__note">
            Type your own, or start from an example. The models only see words they met during
            training — anything else is ignored.
          </div>
        </div>
        <div className="card__body stack">
          <textarea
            value={String(values.text ?? "")}
            placeholder={spec.placeholder}
            aria-label={spec.label}
            onChange={(event) => {
              setDirty(true);
              onChange({ text: event.target.value });
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                setDirty(false);
                onSubmit();
              }
            }}
          />
          <div className="row">
            <button
              className="btn"
              onClick={() => {
                setDirty(false);
                onSubmit();
              }}
              disabled={busy || String(values.text ?? "").trim().length === 0}
            >
              {busy ? "Scoring…" : dirty ? "Run all three models" : "Run again"}
            </button>
            <span className="small muted">or press ⌘/Ctrl + Enter</span>
          </div>

          <div>
            <div className="eyebrow" style={{ marginBottom: 8 }}>
              Examples
            </div>
            <div className="chip-row">
              {spec.samples.map((sample, index) => (
                <button
                  key={index}
                  className="chip"
                  title={sample}
                  onClick={() => {
                    setDirty(false);
                    onChange({ text: sample });
                    // Score it straight away. Filling the box and then waiting
                    // for a second click reads as if the chip did nothing.
                    onSubmit({ text: sample });
                  }}
                >
                  {sample.length > 46 ? `${sample.slice(0, 46)}…` : sample}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card__head">
        <div className="card__title">Customer profile</div>
        <div className="card__note">
          Predictions update as you move a control — all three models rescore on every change.
        </div>
      </div>
      <div className="card__body">
        {spec.fields.map((field) => (
          <Field
            key={field.name}
            field={field}
            value={values[field.name]}
            onChange={(next) => onChange({ ...values, [field.name]: next })}
          />
        ))}
      </div>
    </div>
  );
}
