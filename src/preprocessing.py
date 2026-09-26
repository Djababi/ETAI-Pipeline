"""
Preprocessing -- deliberately minimal for week 2.

This is intentionally the weakest part of the pipeline:
    - missing values are simply dropped (no imputation strategy)
    - categorical columns are one-hot encoded with no thought given to unseen categories or cardinality
    - a single train/test split is used (no cross-validation)

You will replace this with something better in the coming weeks.

One thing that is NOT naive, on purpose: `sensitive_attr` (race) is kept out of the model's input features entirely. It's split alongside the data so it's still available afterwards -- not to train on, but to check whether the model treats different groups differently. See src/evaluate.py:fairness_report.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(
    df: pd.DataFrame,
    target: str,
    sensitive_attr: str,
    drop_columns: list,
    test_size: float,
    random_state: int,
):
    # naive: just drop rows with any missing values
    df = df.dropna()

    y = df[target]

    # kept aside for fairness auditing after training -- never used as a model input
    extras = df[[sensitive_attr, "score_text"]].copy()

    columns_to_exclude = [target, sensitive_attr] + [
        c for c in drop_columns if c in df.columns
    ]
    X = df.drop(columns=columns_to_exclude)

    # naive: one-hot encode all non-numeric columns, no further thought
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, extras_test




PLACEHOLDER_TOKENS = {"-", "?", "n/a", "N/A", "na", "NA", ""}

CANONICAL_MAPS = {
    "sex": {"male": "Male", "female": "Female"},
    "race": {
        "african-american": "African-American", "african american": "African-American",
        "caucasian": "Caucasian", "hispanic": "Hispanic", "other": "Other",
        "asian": "Asian", "native american": "Native American",
    },
    "c_charge_degree": {"f": "F", "felony": "F", "m": "M", "misdemeanor": "M"},
    "score_text": {"low": "Low", "medium": "Medium", "high": "High"},
}

# redundant with priors_count / age / the three juv_*_count columns (see EDA notebook)
REDUNDANT_COLUMNS = ["prior_offenses", "age_in_months", "juvenile_total"]


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Fix structural data problems found in the EDA. No imputation here -- that has to be
    fitted on the training split only, so it belongs after the train/test split."""
    df = df.copy()

    # duplicates: exact rows first, then repeated ids (same case entered twice)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="id", keep="first")

    # placeholder tokens are missing values in disguise
    for col in df.select_dtypes(include="object").columns:
        df.loc[df[col].astype(str).str.strip().isin(PLACEHOLDER_TOKENS), col] = np.nan

    # one spelling per category
    for col, mapping in CANONICAL_MAPS.items():
        cleaned = df[col].str.strip()#clean spaces in the begining and in the ending of the gategory
        df[col] = cleaned.str.lower().map(mapping).fillna(cleaned)

    # columns that should be numeric but loaded as text because of the placeholders
    for col in ["age", "juv_fel_count", "priors_count", "decile_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # domain rules: impossible values become NaN ("impossible but not missing" is still missing)
    df.loc[~df["age"].between(18, 100), "age"] = np.nan
    df.loc[df["juv_fel_count"] < 0, "juv_fel_count"] = np.nan
    df.loc[~df["priors_count"].between(0, 60), "priors_count"] = np.nan
    df.loc[~df["decile_score"].between(1, 10), "decile_score"] = np.nan

    return df.drop(columns=[c for c in REDUNDANT_COLUMNS if c in df.columns])

