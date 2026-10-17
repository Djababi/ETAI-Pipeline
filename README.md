
## Conclusions 

### week 5
Best Model for now: RandomForest (tuned) -- but only by a small margin

#### Hyperparameter tuning
How it works: for each model, **Optuna** (TPE sampler, seed 42) tries different hyperparameter values and keeps the ones with the best mean accuracy over **5 inner folds** (30 trials for the tree and logistic regression, 15 for the slower random forest). The search spaces live in `config.yaml` (`tuning.search_spaces`): `max_depth`, `min_samples_leaf` and `criterion` for the tree, `C` (log scale) for logistic regression, and `n_estimators`, `max_depth`, `min_samples_leaf` and `max_features` for the forest.

Why it doesn't leak: tuning only uses the development set (the test set stays locked), every candidate is the **whole pipeline** (imputer, encoder, scaler and model are re-fit inside every fold), and all candidates are scored on the same folds. The best trial's score is a bit optimistic (it is the maximum of many noisy scores), so the number we report comes from **nested CV**: the inner folds choose the hyperparameters, the outer folds score them on rows that took no part in that choice.

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | `C=0.072` |
| Random forest | 0.647 ± 0.016 | 0.679 ± 0.013 | 0.682 | +0.001 | `n_estimators=133`, `max_depth=8`, `min_samples_leaf=24`, `max_features=log2` |

*Optimism = mean tuning score inside the outer folds − mean honest score on the outer folds.*

The **decision tree** gained the most from tuning: from 0.610 to 0.672 (+0.062, more than 3× the fold-to-fold std). This happened because tuning forced the tree to be smaller (depth 6, at least 101 people per leaf), so it stopped memorising: the train–validation gap dropped from 0.085 to 0.013. The **random forest** also improved clearly (+0.032, about 2× the std) for the same reason. **Logistic regression** barely changed (+0.001, much smaller than the std): it is a simple linear model that was already not overfitting (gap ≈ 0), so there was little left to fix, and stronger regularisation (`C=0.072`) only made a tiny difference. After tuning, the three models are very close (0.672 – 0.679), and these differences are **smaller than the std**, so the random forest is not clearly better. Logistic regression is still a strong choice because it is simpler and had the lowest false positive rates in the fairness check (0.25 African-American vs 0.12 Caucasian; the tuned forest is 0.30 vs 0.16). For our best model, we report the **nested CV score (0.679 ± 0.013)**, not the best trial score (0.682), because that one was used to choose the model and is therefore a bit optimistic.



### week 4
Best Model for now:  LogisticRegression

With imputation, encoding and scaling in place, no rows are lost to missing values anymore: the test set went from 1188 to 1443 people. All models clearly beat the Dummy baseline (0.550). Logistic Regression is still the best model (0.658 test accuracy, slightly up from 0.655) and barely overfits (gap of 0.018). The Decision Tree's gap was cut by more than half (0.194 → 0.088), but its test accuracy stayed the same (0.606). The improvement comes from the model memorising less of the training data, not from better predictions. Random Forest lands in between (0.642), also with some overfitting. On fairness, the gap remains: for Logistic Regression, the FPR is 0.28 for African-American defendants vs 0.14 for Caucasian defendants (previously 0.30 vs 0.16). Random Forest reaches 0.44 for African-American defendants, the same as COMPAS's own score.



### week 3
Best Model for now:  LogisticRegression

After the cleaning step, the Decision Tree still overfits. The gap between train and test went down a little (0.202 → 0.194), but mostly because train accuracy dropped, and test accuracy dropped too (0.627 → 0.605), so cleaning alone isn't enough to fix it. The Logistic Regression changed a bit more than expected: test accuracy went from 0.679 to 0.655. This is probably because more rows now get dropped as missing values (invalid values and placeholders became NaN), so the model has less data to work with. The biggest improvement is in the false positive rate report. Before, the same race showed up under several spellings (e.g. "African-American", "AFRICAN-AMERICAN", "african-american") and placeholders like "-" and "?" were counted as separate groups. Now each group appears only once.

### week 2
Best Model for now: LogisticRegression

Although it shows low performance (accuracy) on the test data. The decision tree model has an overfitting problem, performing better on the training data than on the test data.



# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   ├── tuning.py           # hyperparameter tuning (Optuna) + nested CV
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 5 | Hyperparameter tuning | `src/tuning.py` (new): Optuna tuning of the whole pipeline (`tune_pipeline()`), an honest score with nested cross-validation (`nested_cross_validate()`) and a tuning report with the optimism check (`tuning_report()`); `build_pipeline()` in `src/model.py`; a `tuning` section in `config.yaml` (on/off, number of trials, folds, one search space per model); `optuna` added to `requirements.txt` |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.

