import { useCallback, useEffect, useState } from "react";

import { fetchCatalog, type CatalogTask } from "./api";
import { Overview } from "./views/Overview";
import { TaskView } from "./views/TaskView";

const TASK_ROUTES = ["churn", "spam", "movie-genre"];

type Theme = "light" | "dark" | "system";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem("theme") as Theme | null) ?? "system",
  );

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const cycle = useCallback(
    () => setTheme((current) => (current === "system" ? "light" : current === "light" ? "dark" : "system")),
    [],
  );
  return [theme, cycle];
}

function useRoute() {
  const [path, setPath] = useState(() => window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPath(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((to: string) => {
    if (to !== window.location.pathname) window.history.pushState({}, "", to);
    setPath(to);
    window.scrollTo({ top: 0 });
  }, []);

  return { path, navigate };
}

const THEME_ICON: Record<Theme, string> = { system: "◐", light: "☀", dark: "☾" };

export function App() {
  const { path, navigate } = useRoute();
  const [theme, cycleTheme] = useTheme();
  const [catalog, setCatalog] = useState<CatalogTask[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchCatalog()
      .then((response) => setCatalog(response.tasks))
      .catch((cause: Error) => setError(cause.message));
  }, []);

  const slug = path.replace(/^\/+|\/+$/g, "");
  const activeTask = TASK_ROUTES.includes(slug) ? slug : null;

  return (
    <div className="shell">
      <header className="masthead">
        <div className="masthead__inner">
          <button className="brand" onClick={() => navigate("/")}>
            <span className="brand__mark">ML Studio</span>
            <span className="brand__sub">scikit-learn at the edge</span>
          </button>
          <nav className="tabs" aria-label="Tasks">
            <button
              className="tab"
              aria-current={activeTask === null ? "page" : undefined}
              onClick={() => navigate("/")}
            >
              Overview
            </button>
            {(catalog ?? []).map((task) => (
              <button
                key={task.task}
                className="tab"
                aria-current={activeTask === task.task ? "page" : undefined}
                onClick={() => navigate(`/${task.task}`)}
              >
                {task.title.replace(" Prediction", "").replace(" Classification", "")}
              </button>
            ))}
          </nav>
          <button
            className="theme-toggle"
            onClick={cycleTheme}
            title={`Theme: ${theme}`}
            aria-label={`Switch theme (currently ${theme})`}
          >
            {THEME_ICON[theme]}
          </button>
        </div>
      </header>

      <main>
        {error && (
          <div className="notice" data-tone="error">
            Could not load the model catalogue: {error}
          </div>
        )}
        {!error && activeTask === null && <Overview catalog={catalog} onOpen={navigate} />}
        {!error && activeTask !== null && <TaskView key={activeTask} task={activeTask} />}
      </main>

      <footer className="site-foot">
        <div className="inner">
          <span>
            Models trained with scikit-learn in <code className="token">ml/</code>, exported to JSON,
            and scored in TypeScript on a Cloudflare Worker.
          </span>
          <span>Churn · SMS spam · Movie genre</span>
        </div>
      </footer>
    </div>
  );
}
