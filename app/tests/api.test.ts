/**
 * Exercises the Worker's routes against the real artifacts.
 *
 * The only thing stubbed is the ASSETS binding, which reads the same files
 * `vite build` would copy into the deployment.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import type { PredictResponse, TaskMeta } from "../shared/types";
import app from "../worker/index";

const assets = {
  async fetch(input: Request | string | URL): Promise<Response> {
    const url = new URL(typeof input === "string" ? input : input instanceof URL ? input : input.url);
    try {
      const body = readFileSync(
        fileURLToPath(new URL(`../public${url.pathname}`, import.meta.url).href),
      );
      return new Response(body, { headers: { "content-type": "application/json" } });
    } catch {
      return new Response("not found", { status: 404 });
    }
  },
};

const env = { ASSETS: assets } as unknown as Env;

function call(path: string, init?: RequestInit) {
  return app.request(path, init, env);
}

function postJson(path: string, body: unknown) {
  return call(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("GET /api/health", () => {
  it("reports the available tasks", async () => {
    const response = await call("/api/health");
    expect(response.status).toBe(200);
    const body = (await response.json()) as { status: string; tasks: string[] };
    expect(body.status).toBe("ok");
    expect(body.tasks).toEqual(["churn", "spam", "movie-genre"]);
  });
});

describe("GET /api/models", () => {
  it("lists every task with its metrics but without weights", async () => {
    const response = await call("/api/models");
    expect(response.status).toBe(200);
    const body = (await response.json()) as { tasks: TaskMeta[] };
    expect(body.tasks).toHaveLength(3);
    for (const task of body.tasks) {
      expect(task.models.length).toBe(3);
      for (const model of task.models) {
        expect(model.metrics.accuracy).toBeGreaterThan(0.2);
        expect(model).not.toHaveProperty("artifact");
      }
    }
    expect(JSON.stringify(body)).not.toContain("__pack");
  });
});

describe("GET /api/models/:task", () => {
  it("returns the input specification", async () => {
    const response = await call("/api/models/churn");
    const meta = (await response.json()) as TaskMeta;
    expect(meta.input.kind).toBe("tabular");
    expect(meta.classes).toEqual(["Retained", "Churned"]);
  });

  it("404s on an unknown task", async () => {
    expect((await call("/api/models/nope")).status).toBe(404);
  });
});

describe("POST /api/predict/churn", () => {
  const customer = {
    creditScore: 600,
    age: 45,
    tenure: 3,
    balance: 120000,
    numOfProducts: 1,
    hasCrCard: 1,
    isActiveMember: 0,
    estimatedSalary: 90000,
    geography: "Germany",
    gender: "Female",
  };

  it("answers with one prediction per model", async () => {
    const response = await postJson("/api/predict/churn", customer);
    expect(response.status).toBe(200);
    const body = (await response.json()) as PredictResponse;
    expect(body.models).toHaveLength(3);
    for (const model of body.models) {
      expect(body.classes).toContain(model.predictedLabel);
      expect(model.probabilities).not.toBeNull();
      const total = (model.probabilities as number[]).reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(1, 9);
    }
    expect(body.consensus.total).toBe(3);
    expect(body.consensus.agreeing).toBeGreaterThanOrEqual(2);
  });

  it("flags an inactive German customer as higher risk than an active French one", async () => {
    const risky = (await (await postJson("/api/predict/churn", customer)).json()) as PredictResponse;
    const safe = (await (
      await postJson("/api/predict/churn", {
        ...customer,
        geography: "France",
        isActiveMember: 1,
        age: 30,
        numOfProducts: 2,
      })
    ).json()) as PredictResponse;

    for (const [index, model] of risky.models.entries()) {
      const riskyProbability = (model.probabilities as number[])[1];
      const safeProbability = (safe.models[index].probabilities as number[])[1];
      expect(riskyProbability).toBeGreaterThan(safeProbability);
    }
  });

  it("explains which inputs drove the linear model", async () => {
    const body = (await (await postJson("/api/predict/churn", customer)).json()) as PredictResponse;
    const linear = body.models.find((m) => m.modelId === "logistic_regression");
    expect(linear?.contributions.length).toBeGreaterThan(0);
    expect(linear?.contributions[0].label).toBeTruthy();
  });

  /**
   * A positive contribution has to mean "more likely to churn" whichever way
   * the model voted. Explaining relative to the predicted class instead flips
   * every sign on a negative prediction, which silently inverts the reasons
   * shown to the reader.
   */
  it("keeps contribution signs pointing the same way for both verdicts", async () => {
    const activityOf = async (record: Record<string, unknown>) => {
      const body = (await (await postJson("/api/predict/churn", record)).json()) as PredictResponse;
      const linear = body.models.find((m) => m.modelId === "logistic_regression");
      const active = linear?.contributions.find((c) => c.label === "Is active member");
      return { label: linear?.predictedLabel, value: active?.value };
    };

    const risky = await activityOf({ ...customer, isActiveMember: 0, age: 55 });
    const safe = await activityOf({
      ...customer,
      isActiveMember: 1,
      age: 30,
      numOfProducts: 2,
      geography: "France",
    });

    expect(risky.label).toBe("Churned");
    expect(safe.label).toBe("Retained");

    // Being inactive raises churn risk; being active lowers it. The sign must
    // reflect that in both cases, not flip with the verdict.
    expect(risky.value).toBeGreaterThan(0);
    expect(safe.value).toBeLessThan(0);
  });

  it("rejects a missing field", async () => {
    const { age: _omitted, ...withoutAge } = customer;
    const response = await postJson("/api/predict/churn", withoutAge);
    expect(response.status).toBe(400);
    expect((await response.json()) as { detail: string }).toMatchObject({
      error: "invalid input",
    });
  });

  it("rejects an unknown category", async () => {
    const response = await postJson("/api/predict/churn", { ...customer, geography: "Atlantis" });
    expect(response.status).toBe(400);
  });

  it("rejects a non-numeric number", async () => {
    const response = await postJson("/api/predict/churn", { ...customer, age: "old" });
    expect(response.status).toBe(400);
  });
});

describe("POST /api/predict/spam", () => {
  it("labels an obvious scam as spam across all three models", async () => {
    const response = await postJson("/api/predict/spam", {
      text: "WINNER!! You have been specially selected to receive a £900 prize reward! To claim call 09061701461 now.",
    });
    const body = (await response.json()) as PredictResponse;
    expect(body.consensus.label).toBe("Spam");
    expect(body.consensus.agreeing).toBe(3);
  });

  it("labels ordinary conversation as ham", async () => {
    const body = (await (
      await postJson("/api/predict/spam", { text: "Running ten minutes late, see you at the cafe" })
    ).json()) as PredictResponse;
    expect(body.consensus.label).toBe("Ham");
  });

  it("names the words that triggered the decision", async () => {
    const body = (await (
      await postJson("/api/predict/spam", { text: "FREE entry to win a prize, text WIN now" })
    ).json()) as PredictResponse;
    const contributions = body.models[0].contributions;
    expect(contributions.length).toBeGreaterThan(0);
    expect(contributions.map((c) => c.label)).toContain("free");
  });

  it("reports a margin rather than a fake probability for the SVM", async () => {
    const body = (await (
      await postJson("/api/predict/spam", { text: "call now to claim your free prize" })
    ).json()) as PredictResponse;
    const svm = body.models.find((m) => m.modelId === "linear_svm");
    expect(svm?.probabilities).toBeNull();
    expect(svm?.confidence).toBeNull();
    expect(Number.isFinite(svm?.scores[0])).toBe(true);
  });

  /**
   * A bag-of-words model cannot abstain: given nothing it recognises it
   * returns the commonest training class with a confident-looking number.
   * The response has to carry enough for the UI to say so.
   */
  describe("evidence", () => {
    it("reports which words counted, which were ignored and which were unseen", async () => {
      const body = (await (
        await postJson("/api/predict/spam", { text: "This is erai from earth-555" })
      ).json()) as PredictResponse;

      expect(body.evidence).toBeDefined();
      // "this", "is" and "from" are removed as everyday words during training.
      expect(body.evidence?.ignored).toEqual(expect.arrayContaining(["this", "is", "from"]));
      // "erai" is a name the training data never contained.
      expect(body.evidence?.unknown).toContain("erai");
      expect(body.evidence?.recognised).toEqual(["earth"]);
      expect(body.evidence?.total).toBe(6);
    });

    it("recognises nothing in gibberish", async () => {
      const body = (await (
        await postJson("/api/predict/spam", { text: "qqzzxx wubbleflarn" })
      ).json()) as PredictResponse;
      expect(body.evidence?.recognised).toEqual([]);
      expect(body.evidence?.unknown.length).toBeGreaterThan(0);
    });

    it("finds plenty to read in a real scam text", async () => {
      const body = (await (
        await postJson("/api/predict/spam", {
          text: "URGENT! You have won a FREE prize, call 09061701461 now to claim",
        })
      ).json()) as PredictResponse;
      expect(body.evidence?.recognised).toEqual(
        expect.arrayContaining(["urgent", "won", "free", "prize", "claim"]),
      );
    });

    it("omits evidence for the tabular task", async () => {
      const body = (await (
        await postJson("/api/predict/churn", {
          creditScore: 600,
          age: 45,
          tenure: 3,
          balance: 120000,
          numOfProducts: 1,
          hasCrCard: 1,
          isActiveMember: 0,
          estimatedSalary: 90000,
          geography: "Germany",
          gender: "Female",
        })
      ).json()) as PredictResponse;
      expect(body.evidence).toBeUndefined();
    });
  });

  it("rejects empty text", async () => {
    expect((await postJson("/api/predict/spam", { text: "   " })).status).toBe(400);
  });

  it("rejects a malformed body", async () => {
    const response = await call("/api/predict/spam", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{not json",
    });
    expect(response.status).toBe(400);
  });
});

describe("POST /api/predict/movie-genre", () => {
  it("returns ranked genres for a plot summary", async () => {
    const body = (await (
      await postJson("/api/predict/movie-genre", {
        text: "A retired hitman hunts down the gang that killed his dog, cutting through the criminal underworld one shootout at a time.",
      })
    ).json()) as PredictResponse;

    expect(body.classes.length).toBe(27);
    for (const model of body.models) {
      expect(model.topClasses.length).toBeLessThanOrEqual(5);
      expect(model.topClasses[0].label).toBe(model.predictedLabel);
    }
  });

  it("classifies a documentary plot differently from an animated one", async () => {
    const documentary = (await (
      await postJson("/api/predict/movie-genre", {
        text: "This documentary follows three fishing families over four years as they testify before parliament about the collapse of their industry.",
      })
    ).json()) as PredictResponse;
    expect(documentary.consensus.label).toBe("documentary");
  });
});

describe("unknown routes", () => {
  it("404s unknown API paths as JSON", async () => {
    const response = await call("/api/does-not-exist");
    expect(response.status).toBe(404);
    expect(response.headers.get("content-type")).toContain("application/json");
  });
});
