from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

from .config import CFG
from .utils import ensure_dir, read_json, latest_run_dir
from .data_load import load_csv
from .preprocess import split_xy, train_test_split_stratified
from .evaluate_core import safe_proba, compute_metrics, confusion
from .plotting import plot_confusion_matrix, plot_roc_curve

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained run directory on the holdout test split.")
    p.add_argument("--run_dir", type=str, default="runs/latest", help="Run directory or 'runs/latest'.")
    p.add_argument("--data_path", type=str, default=None, help="Override data path (else taken from run_meta.json).")
    return p.parse_args()

def resolve_run_dir(run_dir_str: str) -> Path:
    run_dir = Path(run_dir_str)
    if run_dir_str.endswith("runs/latest"):
        # runs/latest contains run name text
        latest_tag_file = Path("runs/latest")
        if latest_tag_file.exists() and latest_tag_file.is_file():
            run_name = latest_tag_file.read_text(encoding="utf-8").strip()
            return Path("runs") / run_name
        # fallback: actual latest dir
        d = latest_run_dir(Path("runs"))
        if d is None:
            raise FileNotFoundError("No runs found in 'runs/'. Train first.")
        return d
    return run_dir

def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    meta = read_json(run_dir / "run_meta.json")
    target = (run_dir / "target.txt").read_text(encoding="utf-8").strip()

    data_path = args.data_path or meta["data_path"]
    df = load_csv(data_path)

    X, y = split_xy(df, target, id_col=meta.get("id_col"))
    split = train_test_split_stratified(X, y, test_size=float(meta["test_size"]), seed=int(meta["seed"]))

    model = joblib.load(run_dir / "model.joblib")

    y_pred = model.predict(split.X_test)
    y_score = safe_proba(model, split.X_test)

    m = compute_metrics(split.y_test, y_pred, y_score)
    m["model"] = meta["best_model"]
    m["n_test"] = int(len(split.y_test))

    # Save tables
    ensure_dir(CFG.tables_dir)
    pd.DataFrame([m]).to_csv(CFG.tables_dir / "metrics_test.csv", index=False)

    cv = read_json(run_dir / "cv_results.json")["cv_results"]
    pd.DataFrame(cv).sort_values("cv_roc_auc_mean", ascending=False).to_csv(CFG.tables_dir / "metrics_cv.csv", index=False)

    # Save figures
    ensure_dir(CFG.figures_dir)
    cm = confusion(split.y_test, y_pred)
    fig_cm = plot_confusion_matrix(cm)
    fig_cm.savefig(CFG.figures_dir / "confusion_matrix.png", dpi=180)
    fig_cm.clf()

    if np.isfinite(m.get("roc_auc", float("nan"))):
        fpr, tpr, _ = roc_curve(split.y_test, y_score)
        fig_roc = plot_roc_curve(fpr, tpr, m["roc_auc"])
        fig_roc.savefig(CFG.figures_dir / "roc_curve.png", dpi=180)
        fig_roc.clf()

    print("[OK] Saved:")
    print(f"- {CFG.tables_dir / 'metrics_test.csv'}")
    print(f"- {CFG.tables_dir / 'metrics_cv.csv'}")
    print(f"- {CFG.figures_dir / 'confusion_matrix.png'}")
    print(f"- {CFG.figures_dir / 'roc_curve.png'} (if applicable)")

if __name__ == "__main__":
    main()
