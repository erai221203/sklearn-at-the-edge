/**
 * Turns model output into sentences a non-technical reader can act on.
 *
 * Everything here is presentation only — no number is invented and none is
 * hidden. A probability becomes a band and a frequency ("about 7 in 10")
 * because people read frequencies far more reliably than percentages, and a
 * feature contribution becomes the sentence that contribution actually means.
 * The raw figures stay one click away under "the technical detail".
 */

import type {
  Contribution,
  Evidence,
  ModelPrediction,
  PredictResponse,
  TaskMeta,
} from "../shared/types";

export type InputValues = Record<string, string | number>;

/**
 * The model the plain-language view speaks for.
 *
 * The headline is the majority answer, so every number and every highlighted
 * word beside it has to come from a model that actually gave that answer.
 * Taking the first model in the list instead lets the page announce
 * "thriller" and then rank drama top, explained by words chosen for drama.
 */
export function representativeModel(result: PredictResponse): ModelPrediction | undefined {
  const withProbabilities = result.models.filter((model) => model.probabilities !== null);
  const agreeing = withProbabilities.find(
    (model) => model.predictedLabel === result.consensus.label,
  );
  return (
    agreeing ??
    result.models.find((model) => model.predictedLabel === result.consensus.label) ??
    withProbabilities[0] ??
    result.models[0]
  );
}

/** The model that can attribute a score to individual inputs, if any can. */
export function explainingModel(result: PredictResponse): ModelPrediction | undefined {
  const preferred = representativeModel(result);
  if (preferred && preferred.contributions.length > 0) return preferred;
  return result.models.find(
    (model) =>
      model.contributions.length > 0 && model.predictedLabel === result.consensus.label,
  );
}

// ---------------------------------------------------------------------------
// How much the models actually had to work with
// ---------------------------------------------------------------------------

export type EvidenceLevel = "none" | "thin" | "enough";

export function evidenceLevel(evidence: Evidence | undefined): EvidenceLevel {
  if (!evidence) return "enough";
  if (evidence.recognised.length === 0) return "none";
  return evidence.recognised.length <= 2 ? "thin" : "enough";
}

/**
 * What to say when the input barely overlaps the training vocabulary.
 *
 * A bag-of-words model cannot abstain. Given nothing it recognises it returns
 * the class that was commonest in training, with a probability that looks
 * exactly as trustworthy as a real one. Saying so is the difference between a
 * tool someone can rely on and one that quietly misleads them.
 */
export function evidenceNote(
  evidence: Evidence,
  level: EvidenceLevel,
  commonestClass: string,
): string | null {
  if (level === "none") {
    return `None of these words appear in the models' vocabulary, so nothing here was actually read. The answer below is just the commonest outcome in the training data (“${commonestClass}”) — treat it as no answer at all.`;
  }
  if (level === "thin") {
    const only = evidence.recognised.length === 1 ? "only one word" : "only two words";
    return `The models recognised ${only} of the ${evidence.total} in this message, so there is very little to go on. A longer message gives a far more reliable answer.`;
  }
  return null;
}

export type Tone = "good" | "warning" | "serious" | "critical" | "neutral";

export interface Verdict {
  headline: string;
  detail: string;
  tone: Tone;
  /** 0-1, used for the meter. Null when no model reports a probability. */
  likelihood: number | null;
}

/**
 * Mean probability of the positive class across the models that report one.
 * The SVM is uncalibrated and deliberately excluded rather than guessed at.
 */
export function averagePositiveProbability(result: PredictResponse): number | null {
  const values = result.models
    .map((model) => model.probabilities?.[1])
    .filter((value): value is number => typeof value === "number");
  return values.length === 0
    ? null
    : values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** "about 7 in 10" — rounded to a denominator people actually picture. */
export function frequency(probability: number): string {
  const outOfTen = Math.round(probability * 10);
  if (outOfTen >= 1 && outOfTen <= 9) return `about ${outOfTen} in 10`;
  return `about ${Math.round(probability * 100)} in 100`;
}

export function confidenceWord(probability: number, level: EvidenceLevel = "enough"): string {
  if (level === "none") return "No real evidence";
  const strength = Math.max(probability, 1 - probability);
  if (level === "thin") return strength >= 0.9 ? "Weak evidence" : "Barely a guess";
  if (strength >= 0.95) return "Very confident";
  if (strength >= 0.8) return "Fairly confident";
  if (strength >= 0.65) return "Leaning that way";
  return "Not at all sure";
}

/**
 * `baseRate` is how often customers churn in the dataset overall (about 1 in
 * 5). "Worth watching" is pegged to a quarter above that rather than a round
 * number, so a profile that is genuinely riskier than average is never filed
 * under "likely to stay" just because it sits below an arbitrary cutoff.
 */
export function churnVerdict(probability: number, baseRate: number): Verdict {
  const chance = frequency(probability);
  const watchFrom = Math.min(0.5, baseRate * 1.25);
  if (probability >= 0.7) {
    return {
      headline: "High risk of leaving",
      detail: `The models put this at ${chance} — of customers with this profile, that many left.`,
      tone: "critical",
      likelihood: probability,
    };
  }
  if (probability >= 0.5) {
    return {
      headline: "At risk of leaving",
      detail: `The models put this at ${chance} — of customers with this profile, that many left.`,
      tone: "serious",
      likelihood: probability,
    };
  }
  if (probability >= watchFrom) {
    return {
      headline: "Probably staying, but worth watching",
      detail: `The models put this at ${chance} leaving — low, but above the average customer.`,
      tone: "warning",
      likelihood: probability,
    };
  }
  return {
    headline: "Likely to stay",
    detail: `The models put this at only ${chance} leaving.`,
    tone: "good",
    likelihood: probability,
  };
}

export function spamVerdict(probability: number): Verdict {
  if (probability >= 0.5) {
    return {
      headline: "This looks like a scam message",
      detail:
        probability >= 0.9
          ? "It has the hallmarks of the spam texts the models were trained on."
          : "It leans spam, but not strongly — read it before acting.",
      tone: probability >= 0.9 ? "critical" : "serious",
      likelihood: probability,
    };
  }
  return {
    headline: "This looks like a normal message",
    detail:
      probability <= 0.1
        ? "Nothing in it resembles the scam texts the models were trained on."
        : "Probably fine, though a few words in it do show up in scam texts.",
    tone: probability <= 0.1 ? "good" : "warning",
    likelihood: probability,
  };
}

export function genreVerdict(result: PredictResponse): Verdict {
  const speaker = representativeModel(result);
  const ranked = speaker?.topClasses ?? [];
  const alternatives = ranked
    .filter((entry) => entry.label !== result.consensus.label)
    .slice(0, 2)
    .map((entry) => entry.label)
    .join(" or ");
  const top = ranked.find((entry) => entry.label === result.consensus.label);

  return {
    headline: `This sounds like a ${result.consensus.label} film`,
    detail: alternatives
      ? `${capitalise(alternatives)} would be the next best guess.`
      : "Nothing else came close.",
    tone: "neutral",
    likelihood: top?.probability ?? null,
  };
}

export function agreementSentence(result: PredictResponse): { text: string; tone: Tone } {
  const { agreeing, total } = result.consensus;
  if (agreeing === total) {
    return {
      text: `All ${total} methods agree on this answer.`,
      tone: "good",
    };
  }
  return {
    text: `Only ${agreeing} of the ${total} methods gave this answer, so treat it as a maybe rather than a verdict.`,
    tone: "warning",
  };
}

// ---------------------------------------------------------------------------
// Turning feature contributions into sentences
// ---------------------------------------------------------------------------

export interface Reason {
  text: string;
  /** True when this factor pushes towards the "bad" outcome. */
  raises: boolean;
  strength: number;
}

/**
 * What each churn input means in plain terms.
 *
 * `raises` says which way this factor pushed; `values` is the profile the
 * reader typed in. Both are needed. The inputs are standardised before the
 * model sees them, so a one-hot column like "Gender: Male" is *negative* for a
 * woman — reading direction off the sign alone produces "men churn more often"
 * on a female customer. Categorical and yes/no factors therefore phrase
 * themselves from the actual value, and only continuous ones lean on the sign.
 */
type Phrase = (raises: boolean, values: InputValues) => string;

const CHURN_PHRASES: Record<string, Phrase> = {
  "Is active member": (_, v) =>
    Number(v.isActiveMember) === 1
      ? "They are an active member, which is the strongest single sign a customer will stay."
      : "They have not been active on the account recently — the strongest warning sign in this data.",
  "Has credit card": (_, v) =>
    Number(v.hasCrCard) === 1
      ? "They hold a card with the bank, which strengthens the relationship."
      : "They hold no card with the bank, which weakens the relationship.",
  "Geography: Germany": (_, v) =>
    v.geography === "Germany"
      ? "They bank in Germany, where customers leave about twice as often as in France."
      : "They do not bank in Germany, which has the highest churn of the three countries.",
  "Geography: Spain": (_, v) =>
    v.geography === "Spain"
      ? "They bank in Spain, which sits between France and Germany for churn."
      : "They do not bank in Spain.",
  "Gender: Male": (_, v) =>
    v.gender === "Male"
      ? "They are male, and men leave less often than women in this data."
      : "They are female, and women leave more often than men in this data.",
  Age: (raises) =>
    raises
      ? "Customers in their age group leave noticeably more often."
      : "Customers in their age group tend to stay.",
  "Number of products": (raises) =>
    raises
      ? "The number of products they hold is one that tends to go with leaving."
      : "The number of products they hold is one that tends to go with staying.",
  "Account balance": (raises) =>
    raises
      ? "Their balance sits in a range that often comes before a move."
      : "Their balance is in a range that usually stays put.",
  "Credit score": (raises) =>
    raises
      ? "Their credit score is on the low side for this group."
      : "Their credit score is healthy for this group.",
  "Tenure (years)": (raises) =>
    raises
      ? "They have not been with the bank very long."
      : "They have been with the bank a good while.",
  "Estimated salary": (raises) =>
    raises
      ? "Their salary level nudges the risk up a little."
      : "Their salary level nudges the risk down a little.",
};

export function churnReasons(
  contributions: Contribution[],
  values: InputValues,
  limit = 5,
): Reason[] {
  return contributions
    .map((entry) => {
      const phrase = CHURN_PHRASES[entry.label];
      if (!phrase) return null;
      return {
        text: phrase(entry.value > 0, values),
        raises: entry.value > 0,
        strength: Math.abs(entry.value),
      };
    })
    .filter((reason): reason is Reason => reason !== null)
    .slice(0, limit);
}

/** Advice that follows from the risk band, not from anything the model said. */
export function churnAdvice(probability: number): string {
  if (probability >= 0.7) {
    return "Worth a retention call. Customers this close to leaving usually respond to a direct offer rather than a mailshot.";
  }
  if (probability >= 0.5) {
    return "Add them to a retention campaign and check whether they have been contacted recently.";
  }
  if (probability >= 0.3) {
    return "No action needed today, but they are worth including in the next review.";
  }
  return "No action needed. This customer looks settled.";
}

export function wordReasons(contributions: Contribution[], positiveLabel: string, limit = 6) {
  const towards = contributions.filter((entry) => entry.value > 0).slice(0, limit);
  const against = contributions.filter((entry) => entry.value < 0).slice(0, limit);
  return {
    towards: towards.map((entry) => entry.label),
    against: against.map((entry) => entry.label),
    positiveLabel,
  };
}

export function listWords(words: string[]): string {
  const quoted = words.map((word) => `“${word}”`);
  if (quoted.length <= 1) return quoted.join("");
  return `${quoted.slice(0, -1).join(", ")} and ${quoted[quoted.length - 1]}`;
}

export function capitalise(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function verdictFor(meta: TaskMeta, result: PredictResponse): Verdict {
  if (meta.task === "movie-genre") return genreVerdict(result);
  const probability = averagePositiveProbability(result);
  if (probability === null) {
    return {
      headline: capitalise(result.consensus.label),
      detail: "",
      tone: "neutral",
      likelihood: null,
    };
  }
  if (meta.task !== "churn") return spamVerdict(probability);

  const counts = Object.values(meta.dataset.classBalance);
  const total = counts.reduce((sum, value) => sum + value, 0);
  const churned = meta.dataset.classBalance[meta.classes[1]] ?? 0;
  return churnVerdict(probability, total > 0 ? churned / total : 0.2);
}
