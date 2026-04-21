from __future__ import annotations
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def get_models(seed: int):
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent", random_state=seed),
        "logreg": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=None,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        ),
        "hist_gb": HistGradientBoostingClassifier(
            random_state=seed,
            max_depth=None,
            learning_rate=0.08,
            max_iter=400,
        ),
        # Optional: add XGBoost here if installed
    }
