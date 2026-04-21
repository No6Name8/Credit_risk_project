from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score

from .config import CFG
from .utils import now_tag, ensure_dir, write_json
from .data_load import load_csv
from .preprocess import split_xy, build_preprocessor, train_test_split_stratified
from .models import get_models

def parse_args():
    p = argparse.ArgumentParser(description="Train multiple models for credit risk prediction.")
    p.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV.")
    p.add_argument("--target", type=str, required=True, help="Target column name (binary: 0/1).")
    p.add_argument("--id_col", type=str, default=None, help="Optional ID column to drop.")
    p.add_argument("--run_name", type=str, default=None, help="Optional run name.")
    return p.parse_args()

def main():
    args = parse_args()
    df = load_csv(args.data_path)

    X, y = split_xy(df, args.target, id_col=args.id_col)
    split = train_test_split_stratified(X, y, test_size=CFG.test_size, seed=CFG.seed)

    pre = build_preprocessor(split.X_train)
    models = get_models(CFG.seed)

    # CV config: use ROC-AUC when binary
    is_binary = y.nunique() == 2
    if not is_binary:
        raise ValueError("This template expects a binary target (0/1).")

    cv = StratifiedKFold(n_splits=CFG.cv_folds, shuffle=True, random_state=CFG.seed)

    run_dir = CFG.runs_dir / (args.run_name or now_tag())
    ensure_dir(run_dir)

    cv_results = []
    best_name, best_score, best_pipe = None, -1.0, None

    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        scores = cross_val_score(pipe, split.X_train, split.y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        mean_auc = float(scores.mean())
        std_auc = float(scores.std())
        cv_results.append({"model": name, "cv_roc_auc_mean": mean_auc, "cv_roc_auc_std": std_auc})

        if mean_auc > best_score:
            best_score = mean_auc
            best_name = name
            best_pipe = pipe

    # Fit the best model on full train split
    assert best_pipe is not None
    best_pipe.fit(split.X_train, split.y_train)

    # Save artifacts
    joblib.dump(best_pipe, run_dir / "model.joblib")
    (run_dir / "target.txt").write_text(args.target, encoding="utf-8")
    write_json(run_dir / "cv_results.json", {"cv_results": cv_results, "best_model": best_name, "best_cv_roc_auc": best_score})

    # Save split metadata (no raw data)
    meta = {
        "data_path": str(args.data_path),
        "target": args.target,
        "id_col": args.id_col,
        "n_rows": int(df.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "test_size": CFG.test_size,
        "cv_folds": CFG.cv_folds,
        "seed": CFG.seed,
        "best_model": best_name,
        "best_cv_roc_auc": best_score,
    }
    write_json(run_dir / "run_meta.json", meta)

    # Convenience pointer
    ensure_dir(CFG.runs_dir)
    (CFG.runs_dir / "latest").write_text(run_dir.name, encoding="utf-8")

    print(f"[OK] Trained. Best model: {best_name} (CV ROC-AUC={best_score:.4f})")
    print(f"[OK] Artifacts saved to: {run_dir}")

if __name__ == "__main__":
    main()
