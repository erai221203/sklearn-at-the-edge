/**
 * Chart primitives, built as plain SVG.
 *
 * Conventions that hold across all of them: 2px lines and <=24px bars with a
 * 4px rounded data-end squared at the baseline, hairline solid gridlines one
 * step off the surface, text in ink tokens rather than the series colour, a
 * legend whenever there are two or more series, and a table view underneath
 * every figure so no value is reachable only by hovering.
 */

import { useEffect, useId, useRef, useState, type ReactNode, type RefObject } from "react";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"] as const;
const HEAT_STEPS = 6;

/**
 * Charts are drawn at their true pixel size rather than stretched from a fixed
 * viewBox. A scaled viewBox scales the type with it, so the same component
 * rendered in a full-width card and a half-width one ends up with labels at
 * two different sizes -- and neither matches the surrounding text.
 */
function useMeasure(): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width ?? 0;
      setWidth((previous) => (Math.abs(previous - next) > 0.5 ? next : previous));
    });
    observer.observe(node);
    setWidth(node.getBoundingClientRect().width);
    return () => observer.disconnect();
  }, []);

  return [ref, width];
}

interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

function useTooltip() {
  const [tip, setTip] = useState<TooltipState | null>(null);
  const show = (event: { clientX: number; clientY: number }, content: ReactNode) =>
    setTip({ x: event.clientX, y: event.clientY, content });
  const hide = () => setTip(null);
  const node = tip ? (
    <div
      className="tooltip"
      // Hidden from assistive tech on purpose: it follows the pointer, and
      // every value it shows is also in the figure's table view.
      aria-hidden="true"
      style={{
        left: Math.min(tip.x + 14, (globalThis.innerWidth ?? 1200) - 275),
        top: Math.max(tip.y - 12, 8),
      }}
    >
      {tip.content}
    </div>
  ) : null;
  return { show, hide, node };
}

/** Wraps a chart with its title, legend and the table-view twin. */
export function Figure({
  title,
  note,
  legend,
  children,
  table,
}: {
  title?: string;
  note?: string;
  legend?: ReactNode;
  children: ReactNode;
  table?: { head: string[]; rows: (string | number)[][] };
}) {
  return (
    <div className="figure">
      {title && (
        <div>
          <div className="card__title">{title}</div>
          {note && <div className="card__note">{note}</div>}
        </div>
      )}
      {legend}
      {children}
      {table && (
        <details className="figure__table">
          <summary>Show data table</summary>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {table.head.map((cell) => (
                    <th key={cell}>{cell}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, index) => (
                  <tr key={index}>
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

export function Legend({
  items,
  square = false,
}: {
  items: { label: string; color: string }[];
  square?: boolean;
}) {
  return (
    <div className="legend">
      {items.map((item) => (
        <span className="legend__item" key={item.label}>
          <span
            className={square ? "legend__key legend__key--square" : "legend__key"}
            style={{ background: item.color }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

// --------------------------------------------------------------- line chart

export interface LineSeries {
  name: string;
  color: string;
  points: { x: number; y: number }[];
}

/**
 * Unit-square line chart, used for ROC curves.
 *
 * ROC curves all terminate at (1,1), so direct end-labels would collide;
 * identity lives in the legend and exact values in the crosshair tooltip.
 */
export function UnitLineChart({
  series,
  xLabel,
  yLabel,
  reference,
}: {
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  reference?: string;
}) {
  const [ref, measured] = useMeasure();
  // Both axes are 0-1, so the plot is capped and centred to stay near-square.
  // Stretched across a full-width card a ROC curve flattens into a shape that
  // reads much better than it is.
  const width = Math.min(Math.max(measured, 260), 620);
  const height = Math.round(width * 0.74);
  const pad = { top: 14, right: 16, bottom: 42, left: 46 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const sx = (value: number) => pad.left + value * plotW;
  const sy = (value: number) => pad.top + (1 - value) * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const { show, hide, node } = useTooltip();
  const [cursor, setCursor] = useState<number | null>(null);

  const nearest = (points: { x: number; y: number }[], target: number) => {
    let best = points[0];
    for (const point of points) {
      if (Math.abs(point.x - target) < Math.abs(best.x - target)) best = point;
    }
    return best;
  };

  const handleMove = (event: React.MouseEvent<SVGRectElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - box.left) / box.width;
    const value = Math.min(1, Math.max(0, ratio));
    setCursor(value);
    show(
      event,
      <>
        <strong>
          {xLabel} {value.toFixed(2)}
        </strong>
        {series.map((line) => (
          <div className="tooltip__row" key={line.name}>
            <span className="legend__key" style={{ background: line.color }} />
            {line.name}: {nearest(line.points, value).y.toFixed(3)}
          </div>
        ))}
      </>,
    );
  };

  return (
    <div ref={ref}>
      {measured > 0 && (
      <svg
        className="figure__chart"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        style={{ marginInline: "auto" }}
        role="img"
        aria-label={`${yLabel} against ${xLabel} for ${series.map((s) => s.name).join(", ")}`}
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              x1={pad.left}
              x2={pad.left + plotW}
              y1={sy(tick)}
              y2={sy(tick)}
              stroke="var(--grid)"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
            <text x={pad.left - 10} y={sy(tick) + 4} textAnchor="end" fontSize="11" fill="var(--ink-muted)">
              {tick.toFixed(2)}
            </text>
            <text
              x={sx(tick)}
              y={pad.top + plotH + 20}
              textAnchor="middle"
              fontSize="11"
              fill="var(--ink-muted)"
            >
              {tick.toFixed(2)}
            </text>
          </g>
        ))}

        <line
          x1={pad.left}
          x2={pad.left + plotW}
          y1={pad.top + plotH}
          y2={pad.top + plotH}
          stroke="var(--axis)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />

        {reference && (
          <>
            <line
              x1={sx(0)}
              y1={sy(0)}
              x2={sx(1)}
              y2={sy(1)}
              stroke="var(--axis)"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={sx(0.68)}
              y={sy(0.68) + 15}
              fontSize="11"
              fill="var(--ink-muted)"
              transform={`rotate(${-(Math.atan2(plotH, plotW) * 180) / Math.PI} ${sx(0.68)} ${
                sy(0.68) + 15
              })`}
            >
              {reference}
            </text>
          </>
        )}

        {series.map((line) => (
          <path
            key={line.name}
            d={line.points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x)},${sy(p.y)}`).join(" ")}
            fill="none"
            stroke={line.color}
            strokeWidth="2"
            strokeLinejoin="round"
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {cursor !== null && (
          <g>
            <line
              x1={sx(cursor)}
              x2={sx(cursor)}
              y1={pad.top}
              y2={pad.top + plotH}
              stroke="var(--axis)"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
            {series.map((line) => {
              const point = nearest(line.points, cursor);
              return (
                <circle
                  key={line.name}
                  cx={sx(point.x)}
                  cy={sy(point.y)}
                  r="4.5"
                  fill={line.color}
                  stroke="var(--surface)"
                  strokeWidth="2"
                />
              );
            })}
          </g>
        )}

        <text
          x={pad.left + plotW / 2}
          y={height - 6}
          textAnchor="middle"
          fontSize="11.5"
          fill="var(--ink-2)"
        >
          {xLabel}
        </text>
        <text
          x={14}
          y={pad.top + plotH / 2}
          textAnchor="middle"
          fontSize="11.5"
          fill="var(--ink-2)"
          transform={`rotate(-90 14 ${pad.top + plotH / 2})`}
        >
          {yLabel}
        </text>

        <rect
          x={pad.left}
          y={pad.top}
          width={plotW}
          height={plotH}
          fill="transparent"
          onMouseMove={handleMove}
          onMouseLeave={() => {
            setCursor(null);
            hide();
          }}
        />
      </svg>
      )}
      {node}
    </div>
  );
}

// ---------------------------------------------------------------- bar chart

/** Rounded on the data-end only; square where it meets the baseline. */
function barPath(x: number, y: number, w: number, h: number, r: number, roundRight: boolean): string {
  const radius = Math.max(0, Math.min(r, Math.abs(w)));
  if (w <= 0.5) return `M${x},${y}h${w}v${h}h${-w}Z`;
  return roundRight
    ? `M${x},${y}h${w - radius}a${radius},${radius} 0 0 1 ${radius},${radius}v${h - 2 * radius}a${radius},${radius} 0 0 1 ${-radius},${radius}h${-(w - radius)}Z`
    : `M${x + radius},${y}h${w - radius}v${h}h${-(w - radius)}a${radius},${radius} 0 0 1 ${-radius},${-radius}v${-(h - 2 * radius)}a${radius},${radius} 0 0 1 ${radius},${-radius}Z`;
}

export interface BarRow {
  label: string;
  value: number;
  /** Optional detail line for the tooltip. */
  detail?: string;
}

/**
 * Horizontal bars.
 *
 * `diverging` centres on zero and colours by sign (the blue/red pole pair);
 * otherwise every bar takes slot 1, because bar length already encodes
 * magnitude and a value-ramp would double-encode it.
 */
export function BarChart({
  rows,
  diverging = false,
  format = (value: number) => value.toFixed(3),
  maxRows,
  labelWidth = 150,
}: {
  rows: BarRow[];
  diverging?: boolean;
  format?: (value: number) => string;
  maxRows?: number;
  labelWidth?: number;
}) {
  const visible = maxRows ? rows.slice(0, maxRows) : rows;
  const rowHeight = 26;
  const barHeight = Math.min(18, rowHeight - 8);
  const [ref, measured] = useMeasure();
  const width = Math.max(measured, 240);
  const height = visible.length * rowHeight + 14;
  const right = 56;
  // Long category names must not squeeze the plot out of a narrow card.
  const gutter = Math.min(labelWidth, Math.max(70, width * 0.38));
  const plotW = Math.max(width - gutter - right, 40);
  const extent = Math.max(...visible.map((row) => Math.abs(row.value)), 1e-9);
  const zeroX = diverging ? gutter + plotW / 2 : gutter;
  const scale = diverging ? Math.max(plotW / 2 - 46, 10) / extent : plotW / extent;
  const { show, hide, node } = useTooltip();
  const clipId = useId();

  return (
    <div ref={ref}>
      {measured > 0 && (
      <svg
        className="figure__chart"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Bar chart of ${visible.length} values`}
      >
        <line
          x1={zeroX}
          x2={zeroX}
          y1={4}
          y2={visible.length * rowHeight + 4}
          stroke="var(--axis)"
          strokeWidth="1"
          vectorEffect="non-scaling-stroke"
        />
        {visible.map((row, index) => {
          const y = index * rowHeight + 4;
          const length = Math.abs(row.value) * scale;
          const negative = row.value < 0;
          const x = diverging && negative ? zeroX - length : zeroX;
          const color = diverging
            ? negative
              ? "var(--diverge-neg)"
              : "var(--diverge-pos)"
            : SERIES[0];
          const valueX = diverging && negative ? x - 6 : x + length + 6;
          return (
            <g
              key={`${row.label}-${index}`}
              onMouseMove={(event) =>
                show(
                  event,
                  <>
                    <strong>{row.label}</strong>
                    <div className="tooltip__row">{format(row.value)}</div>
                    {row.detail && <div className="tooltip__row">{row.detail}</div>}
                  </>,
                )
              }
              onMouseLeave={hide}
            >
              <rect x={0} y={y - 4} width={width} height={rowHeight} fill="transparent" />
              <clipPath id={`${clipId}-${index}`}>
                <rect x={0} y={y - 4} width={gutter - 10} height={rowHeight} />
              </clipPath>
              <text
                x={gutter - 10}
                y={y + barHeight / 2 + 4}
                textAnchor="end"
                fontSize="11.5"
                fill="var(--ink-2)"
                clipPath={`url(#${clipId}-${index})`}
              >
                {row.label}
              </text>
              <path
                d={barPath(x, y, length, barHeight, 4, !(diverging && negative))}
                fill={color}
              />
              <text
                x={valueX}
                y={y + barHeight / 2 + 4}
                textAnchor={diverging && negative ? "end" : "start"}
                fontSize="11"
                fill="var(--ink-muted)"
                fontVariant="tabular-nums"
              >
                {format(row.value)}
              </text>
            </g>
          );
        })}
      </svg>
      )}
      {node}
    </div>
  );
}

// -------------------------------------------------------- confusion matrix

export function ConfusionMatrix({
  matrix,
  labels,
}: {
  matrix: number[][];
  labels: string[];
}) {
  const total = matrix.flat().reduce((a, b) => a + b, 0);
  const max = Math.max(...matrix.flat(), 1);
  const { show, hide, node } = useTooltip();
  const [ref, measured] = useMeasure();
  const width = Math.max(measured, 260);
  const labelPad = Math.min(96, width * 0.22);
  const cell = Math.max(48, (width - labelPad - 6) / labels.length);
  const headRoom = 34;
  const height = headRoom + labels.length * cell;

  const stepFor = (value: number) => Math.min(HEAT_STEPS - 1, Math.round((value / max) * (HEAT_STEPS - 1)));

  return (
    <div ref={ref}>
      {measured > 0 && (
      <svg
        className="figure__chart"
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Confusion matrix of predicted against actual classes"
      >
        {labels.map((label, column) => (
          <text
            key={`head-${label}`}
            x={labelPad + column * cell + cell / 2}
            y={headRoom - 12}
            textAnchor="middle"
            fontSize="13"
            fill="var(--ink-2)"
          >
            {label}
          </text>
        ))}
        <text x={labelPad - 10} y={headRoom - 12} textAnchor="end" fontSize="11" fill="var(--ink-muted)">
          predicted →
        </text>

        {matrix.map((row, rowIndex) =>
          row.map((value, columnIndex) => {
            const step = stepFor(value);
            const x = labelPad + columnIndex * cell;
            const y = headRoom + rowIndex * cell;
            return (
              <g
                key={`${rowIndex}-${columnIndex}`}
                onMouseMove={(event) =>
                  show(
                    event,
                    <>
                      <strong>
                        {labels[rowIndex]} predicted as {labels[columnIndex]}
                      </strong>
                      <div className="tooltip__row">
                        {value.toLocaleString()} of {total.toLocaleString()} (
                        {((value / total) * 100).toFixed(1)}%)
                      </div>
                    </>,
                  )
                }
                onMouseLeave={hide}
              >
                {/* 2px surface gap does the separating -- no borders on marks. */}
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={cell - 2}
                  height={cell - 2}
                  rx="6"
                  fill={`var(--heat-${step})`}
                />
                <text
                  x={x + cell / 2}
                  y={y + cell / 2 + 6}
                  textAnchor="middle"
                  fontSize={Math.min(20, Math.max(13, cell * 0.16))}
                  fontWeight="600"
                  fill={step >= 3 ? "#fff" : "var(--ink)"}
                >
                  {value.toLocaleString()}
                </text>
              </g>
            );
          }),
        )}

        {labels.map((label, row) => (
          <text
            key={`side-${label}`}
            x={labelPad - 12}
            y={headRoom + row * cell + cell / 2 + 5}
            textAnchor="end"
            fontSize="13"
            fill="var(--ink-2)"
          >
            {label}
          </text>
        ))}
        <text
          x={20}
          y={headRoom + (labels.length * cell) / 2}
          textAnchor="middle"
          fontSize="11"
          fill="var(--ink-muted)"
          transform={`rotate(-90 20 ${headRoom + (labels.length * cell) / 2})`}
        >
          actual →
        </text>
      </svg>
      )}
      {node}
    </div>
  );
}

// ---------------------------------------------------------------- fragments

export function Meter({ value, color = SERIES[0] }: { value: number; color?: string }) {
  return (
    <div className="meter">
      <div
        className="meter__fill"
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }}
      />
    </div>
  );
}

export function Stat({
  label,
  value,
  foot,
  hero = false,
}: {
  label: string;
  value: string;
  foot?: string;
  hero?: boolean;
}) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className={hero ? "stat__value stat__value--hero" : "stat__value"}>{value}</span>
      {foot && <span className="stat__foot">{foot}</span>}
    </div>
  );
}
