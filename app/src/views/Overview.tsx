import type { CatalogTask } from "../api";
import { BarChart, Figure, Stat } from "../components/charts";
import { count, percent } from "../format";

function bestModel(task: CatalogTask) {
  return task.models.reduce((best, model) =>
    model.metrics.accuracy > best.metrics.accuracy ? model : best,
  );
}

/** What "always guess the commonest answer" would score. */
function majorityBaseline(task: CatalogTask): number {
  const counts = Object.values(task.dataset.classBalance);
  const total = counts.reduce((sum, value) => sum + value, 0);
  return total === 0 ? 0 : Math.max(...counts) / total;
}

/** Plain descriptions, so the landing page never has to say "classifier". */
const BLURB: Record<string, { question: string; does: string; scale: (t: CatalogTask) => string }> = {
  churn: {
    question: "Is this bank customer about to leave?",
    does: "Describe a customer — age, balance, how active they are — and get a risk level with the reasons behind it.",
    scale: (t) => `${count(t.dataset.rows)} real customers`,
  },
  spam: {
    question: "Is this text message a scam?",
    does: "Paste any SMS and see whether it looks like spam, with the giveaway words highlighted.",
    scale: (t) => `${count(t.dataset.rows)} real messages`,
  },
  "movie-genre": {
    question: "What genre is this film?",
    does: "Describe the plot in your own words and see which of 27 genres it sounds like.",
    scale: (t) => `${count(t.dataset.rows)} film summaries`,
  },
};

export function Overview({
  catalog,
  onOpen,
}: {
  catalog: CatalogTask[] | null;
  onOpen: (path: string) => void;
}) {
  if (!catalog) {
    return (
      <div className="grid grid--3">
        {[0, 1, 2].map((index) => (
          <div key={index} className="skeleton" style={{ height: 230 }} />
        ))}
      </div>
    );
  }

  const totalRows = catalog.reduce((sum, task) => sum + task.dataset.rows, 0);

  return (
    <>
      <div className="page-head">
        <h1>Three questions, answered by machine learning</h1>
        <p>
          Pick a section, put in your own example, and get an answer in plain English along with
          the reasons behind it. Each answer comes from a model that learned from{" "}
          {count(totalRows)} real records — nothing is guessed on the spot.
        </p>
      </div>

      <section className="section" style={{ marginTop: 0 }}>
        <div className="grid grid--3">
          {catalog.map((task) => {
            const best = bestModel(task);
            const blurb = BLURB[task.task];
            return (
              <button key={task.task} className="task-card" onClick={() => onOpen(`/${task.task}`)}>
                <div>
                  <h3>{blurb?.question ?? task.title}</h3>
                  <p style={{ marginTop: 8 }}>{blurb?.does ?? task.subtitle}</p>
                </div>
                <div className="task-card__foot">
                  <Stat
                    label="Gets it right"
                    value={percent(best.metrics.accuracy, 0)}
                    foot={`of the time · learned from ${blurb?.scale(task) ?? count(task.dataset.rows)}`}
                  />
                </div>
              </button>
            );
          })}
        </div>
      </section>

      <section className="section">
        <div className="card">
          <div className="card__body">
            <Figure
              title="How often each one is right"
              note="Measured on examples the models had never seen. Open the table below to compare each score against what you would get by always guessing the commonest answer — that is the bar worth beating."
              table={{
                head: ["Question", "Model is right", "Always-guess baseline"],
                rows: catalog.map((task) => [
                  BLURB[task.task]?.question ?? task.title,
                  percent(bestModel(task).metrics.accuracy, 1),
                  percent(majorityBaseline(task), 1),
                ]),
              }}
            >
              <BarChart
                rows={catalog.map((task) => ({
                  label: task.title.replace(" Prediction", "").replace(" Classification", ""),
                  value: bestModel(task).metrics.accuracy,
                  detail: `Beats the always-guess baseline of ${percent(majorityBaseline(task), 0)}`,
                }))}
                format={(value) => percent(value, 0)}
                labelWidth={130}
              />
            </Figure>
            <p className="small muted" style={{ marginTop: 12 }}>
              These numbers are not comparable with each other. Picking one of 27 film genres is a
              far harder task than sorting messages into two piles, which is why{" "}
              {(() => {
                const genre = catalog.find((task) => task.task === "movie-genre");
                return genre ? percent(bestModel(genre).metrics.accuracy, 0) : "—";
              })()}{" "}
              on genres — against a{" "}
              {(() => {
                const genre = catalog.find((task) => task.task === "movie-genre");
                return genre ? percent(majorityBaseline(genre), 0) : "—";
              })()}{" "}
              baseline — can be a better result than a higher score elsewhere.
            </p>
          </div>
        </div>
      </section>

      <p className="small muted" style={{ marginTop: 28, maxWidth: "72ch" }}>
        Under the hood: each model was trained in Python with scikit-learn, then converted so it
        can run inside a Cloudflare Worker, which has no Python. Every conversion is checked
        against the original model's own predictions before it ships. Open “the technical detail”
        on any section to see the algorithms, scores and charts.
      </p>
    </>
  );
}
