"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> split (development / locked test)
    -> cross-validate on development -> [tune: nested CV + Optuna study] -> refit final model -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import data_cleaning, preprocess, drop_duplicate_rows
from src.model import build_pipeline
from src.evaluate import build_cv, cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df = data_cleaning(df, config['diagnostics'])

    df = drop_duplicate_rows(df, config['diagnostics']['id_column'])

    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = preprocess(
        df,
        config['data'],
        config['split'],
        config['preprocessing']
    )

    # stratified k-fold on the development set -- the locked test set is never touched here
    cv_config = config["cv"]
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)
    cv = build_cv(cv_config)
    pipeline = build_pipeline(config['preprocessing'], config["model"])
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)

    print("=== Default hyperparameters ===")
    report = "=== Default hyperparameters ===\n" + cv_report(fold_scores, scoring)

    # week 5: hyperparameter tuning -- on the development set only, every candidate scored by
    # CV of the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    if tuning_config.get("enabled", False):
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning.enabled is true, but config.yaml has no search space for "
                             f"'{model_type}'. Add one under tuning.search_spaces, or set enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        # inner folds: score each candidate (the outer folds are the usual `cv` above)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # 1. the honest estimate (its out-of-fold predictions replace the untuned ones in the reports)
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        # 2. the hyperparameters we keep: the same procedure, once, on the whole development set
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )

        print("\n=== Tuned: nested cross-validation (honest score) ===")
        report += "\n\n=== Tuned: nested cross-validation (honest score) ===\n" + cv_report(nested_scores, scoring)
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # CV only estimates how good the recipe is; the model you'd actually use is refit on all development rows
    # (with tuning on, `pipeline` already carries the tuned hyperparameters)
    final_model = pipeline.fit(X_dev, y_dev)

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
