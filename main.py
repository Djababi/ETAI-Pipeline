"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""
import yaml

from src.data import load_data
from src.preprocessing import data_cleaning, preprocess, build_preprocessor, drop_duplicate_rows
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run
from sklearn.pipeline import Pipeline


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_pipeline( model_config:dict, preprocessing_config=None):
    # exactly what main.py builds: Pipeline([("prep", build_preprocessor(...)), ("model", build_model(...))])
    return Pipeline([("prep", build_preprocessor(preprocessing_config)),
                     ("model", build_model(model_config))])


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

    model = make_pipeline(config["model"],config['preprocessing'] )
    model.fit(X_dev, y_dev)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_dev_pred = model.predict(X_dev)
    y_test_pred = model.predict(X_test)

    report = evaluate(y_dev, y_dev_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
