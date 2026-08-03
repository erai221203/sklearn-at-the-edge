"""Retrain every task and regenerate the artifacts the Worker serves.

    python ml/train_all.py

Writes `app/public/models/*.json` (deployed with the site) and
`app/tests/fixtures/*.parity.json` (scikit-learn's own answers, which the
TypeScript test suite is checked against).
"""

from __future__ import annotations

import sys
import time

import train_churn
import train_movie_genre
import train_spam

TASKS = [
    ("churn", train_churn.main),
    ("spam", train_spam.main),
    ("movie-genre", train_movie_genre.main),
]


def main() -> int:
    selected = sys.argv[1:]
    failures = []
    for name, run in TASKS:
        if selected and name not in selected:
            continue
        started = time.monotonic()
        try:
            run()
        except Exception as error:  # noqa: BLE001 - report and continue to the next task
            failures.append((name, error))
            print(f"  FAILED: {error}\n")
            continue
        print(f"  done in {time.monotonic() - started:.1f}s\n")

    if failures:
        print("Failed tasks:")
        for name, error in failures:
            print(f"  {name}: {error}")
        return 1
    print("All artifacts regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
