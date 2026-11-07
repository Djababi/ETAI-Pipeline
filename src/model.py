"""Model construction (week 8: XGBoost and a feature-selection step between preprocessing and the model)."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectorMixin, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.random_projection import SparseRandomProjection
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor

def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    A forest of oblique trees: each tree is fit on a bootstrap sample, on `n_components`
    random sparse linear combinations of the features (a new set per tree), so its splits are
    hyperplanes. `tree_params` go to every tree (`max_features` defaults to "sqrt"). Tuned as
    `model__estimator__tree__<param>` and `model__estimator__project__<param>`.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)
def adaboost(max_depth: int = 1, **params):
    """
    AdaBoost over decision trees of depth `max_depth` (1 = stumps); `params` go to
    AdaBoostClassifier. The trees' depth is tuned as `model__estimator__max_depth`.
    """
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def adaboost_logistic(C: float = 1.0, max_iter: int = 2000, **params):
    """
    Week 6, Challenge 2: AdaBoost over logistic regressions instead of trees; `params` go to
    AdaBoostClassifier. A weighted sum of linear models is still linear, so this can only
    move the boundary, never bend it. `C` is tuned as `model__estimator__C`.
    """
    return AdaBoostClassifier(estimator=LogisticRegression(C=C, max_iter=max_iter), **params)


_MODELS = {
    # always predicts the most frequent class -- the floor every real model must beat
    "dummy": DummyClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,   # week 6
    "adaboost": adaboost,                             # week 6
    "adaboost_logistic": adaboost_logistic,           # week 6, Challenge 2
    "gradient_boosting": GradientBoostingClassifier,  # week 7
    "hist_gradient_boosting": HistGradientBoostingClassifier,  # week 7
    "naive_bayes": GaussianNB,                        # week 7: a fast probabilistic baseline
    "xgboost": XGBClassifier,                         # week 8: gradient boosting with brakes inside each split
}


def build_model(model_config: dict):
    """Builds the model named by `model_config["type"]` with `model_config["params"]`."""
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Week 8: keeps the features chosen by at least `min_votes` of three models, fitted on the
    training rows only: a random forest and gradient boosting vote for features with above-average
    importance, an L1-penalised logistic regression for features it doesn't shrink to zero.
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):
        return self.support_


def build_selector(selection_config: dict):
    """
    Week 8, Challenge B1: the selector named by `selection_config["method"]`. Each one is a
    learned step (it looks at the labels), so it lives inside the Pipeline and is re-fit per fold.
      - vote:       VoteSelector, keeps features with >= `min_votes` of 3 votes (combined)
      - kbest:      SelectKBest(f_classif), keeps the `k` features with the highest F-test (filter)
      - from_model: SelectFromModel(random forest), keeps above-average importances (embedded)
    """
    method = selection_config.get("method", "vote")
    seed = selection_config.get("random_state")
    if method == "vote":
        return VoteSelector(min_votes=selection_config.get("min_votes", 2), random_state=seed)
    if method == "kbest":
        return SelectKBest(f_classif, k=selection_config.get("k", 5))
    if method == "from_model":
        return SelectFromModel(RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=seed, n_jobs=-1))
    raise ValueError(f"Unknown feature_selection.method: {method}. Options: vote, kbest, from_model")


def describe_selection(selection_config: dict) -> str:
    """Short label of the selection setting, for the scoreboard and the report header."""
    if not (selection_config and selection_config.get("enabled", False)):
        return "off"
    method = selection_config.get("method", "vote")
    if method == "vote":
        return f"vote(min_votes={selection_config.get('min_votes', 2)})"
    if method == "kbest":
        return f"kbest(k={selection_config.get('k', 5)})"
    return method


def build_pipeline(preprocessing_config: dict, model_config: dict, selection_config: dict = None,
                   memory=None) -> Pipeline:
    """
    The estimator that is cross-validated, tuned and refit: preprocessing, feature selection
    (if `selection_config["enabled"]`) and model as one Pipeline, so every learned step is
    re-fit on the training part of each fold. Step names address hyperparameters from outside
    (`model__max_depth`, `select__min_votes`). `memory` (a folder) caches the fitted
    preprocessing and selection steps: a fit on the same rows with the same settings is
    reused instead of repeated, so tuning the model doesn't refit them in every trial.
    """
    steps = [("prep", build_preprocessor(preprocessing_config))]
    if selection_config and selection_config.get("enabled", False):
        steps.append(("select", build_selector(selection_config)))
    steps.append(("model", build_model(model_config)))
    return Pipeline(steps, memory=memory)
