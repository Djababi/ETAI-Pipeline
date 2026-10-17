"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so youcan open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime


def save_run(results_dir: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model']['params']}\n"
        f"Test size: {config['split']['test_size']}  "
        f"random_state: {config['split']['random_state']}\n"
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
