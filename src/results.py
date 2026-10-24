"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (metrics, classification report, fairness table) to a timestamped file in `results/` instead, so you can open it again later, or compare two runs side by side after changing something in config.yaml.

Week 7: every run also adds one row to the scoreboard (`results/runs.csv`), so all runs can be compared.
"""
import os

import pandas as pd


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to `results_dir/run_<run>.txt` (the folder is created
    automatically if it doesn't exist yet) and returns the path that was written.
    `run` is the run id (a timestamp) chosen by main.py, the same one as in the scoreboard.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Test size: {config['split']['test_size']}  "
        f"random_state: {config['split']['random_state']}\n"
        f"CV: {cv['n_splits']} folds  selection metric: {cv['selection_metric']}  "
        f"metrics: {', '.join(cv['metrics'])}\n"
    )
    tuning = config.get("tuning", {})
    if tuning.get("enabled", False):
        header += (
            f"Tuning: Optuna TPE, n_trials={tuning['n_trials']}  inner folds={tuning['n_splits']}  "
            f"random_state={tuning['random_state']}\n"
            f"Search space: {tuning['search_spaces'].get(config['model']['type'])}\n"
        )
    else:
        header += "Tuning: off\n"
    header += "=" * 60 + "\n\n"

    with open(path, "w") as f:
        f.write(header + report_text)

    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs
