from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from .config import CFG
from .utils import ensure_dir, read_json, latest_run_dir
from .data_load import load_csv
from .preprocess import split_xy, train_test_split_stratified

def parse_args():
    p = argparse.ArgumentParser(description="Explainability for a trained model.")
    p.add_argument("--run_dir", type=str, default="runs/latest", help="Run directory or 'runs/latest'.")
    p.add_argument("--data_path", type=str, default=None, help="Override data path (else taken from run_meta.json).")
    p.add_argument("--top_k", type=int, default=20, help="Top features to display (after one-hot).")
    return p.parse_args()

def resolve_run_dir(run_dir_str: str) -> Path:
    run_dir = Path(run_dir_str)
    if run_dir_str.endswith("runs/latest"):
        latest_tag_file = Path("runs/latest")
        if latest_tag_file.exists() and latest_tag_file.is_file():
            run_name = latest_tag_file.read_text(encoding="utf-8").strip()
            return Path("runs") / run_name
        d = latest_run_dir(Path("runs"))
        if d is None:
            raise FileNotFoundError("No runs found in 'runs/'. Train first.")
        return d
    return run_dir

def try_shap(model, X, out_path: Path) -> bool:
    try:
        import shap  # noqa
    except Exception:
        return False

    try:
        # SHAP can be heavy; we sample
        X_sample = X.sample(min(2000, len(X)), random_state=42)
        # For Pipeline: preprocessor + model
        pre = model.named_steps["pre"]
        clf = model.named_steps["model"]
        X_trans = pre.transform(X_sample)

        # Use model-agnostic explainer if needed
        explainer = None
        try:
            explainer = shap.Explainer(clf, X_trans)
        except Exception:
            explainer = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:,1], shap.sample(X_trans, 100))
        shap_values = explainer(X_trans)

        plt.figure()
        shap.summary_plot(shap_values, X_trans, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return True
    except Exception:
        return False

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

    # Permutation importance on test set (fast + reliable)
    ensure_dir(CFG.figures_dir)
    r = permutation_importance(model, split.X_test, split.y_test, n_repeats=5, random_state=int(meta["seed"]), n_jobs=-1)

    # Get feature names (post-transform)
    pre = model.named_steps["pre"]
    feat_names = getattr(pre, "get_feature_names_out", None)
    if feat_names is not None:
        names = pre.get_feature_names_out()
    else:
        names = np.array([f"f{i}" for i in range(len(r.importances_mean))])

    imp = pd.DataFrame({
        "feature": names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

    imp.to_csv(CFG.tables_dir / "permutation_importance.csv", index=False)

    top = imp.head(args.top_k)
    plt.figure()
    plt.barh(list(reversed(top["feature"].tolist())), list(reversed(top["importance_mean"].tolist())))
    plt.xlabel("Permutation Importance (decrease in score)")
    plt.title("Top Feature Importances (Permutation)")
    plt.tight_layout()
    plt.savefig(CFG.figures_dir / "permutation_importance.png", dpi=180)
    plt.close()

    # Optional SHAP
    shap_path = CFG.figures_dir / "shap_summary.png"
    shap_ok = try_shap(model, split.X_test, shap_path)

    print("[OK] Saved:")
    print(f"- {CFG.tables_dir / 'permutation_importance.csv'}")
    print(f"- {CFG.figures_dir / 'permutation_importance.png'}")
    if shap_ok:
        print(f"- {shap_path}")
    else:
        print("- SHAP not generated (optional dependency or model compatibility).")

if __name__ == "__main__":
    main()
