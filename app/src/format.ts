export const percent = (value: number, digits = 1): string =>
  `${(value * 100).toFixed(digits)}%`;

export const decimal = (value: number, digits = 3): string => value.toFixed(digits);

export const count = (value: number): string => value.toLocaleString("en-US");

export const compact = (value: number): string =>
  value >= 1000 ? new Intl.NumberFormat("en-US", { notation: "compact" }).format(value) : String(value);

export const signed = (value: number, digits = 3): string =>
  `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;

export const titleCase = (value: string): string =>
  value.charAt(0).toUpperCase() + value.slice(1);
