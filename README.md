# Explainable Credit Risk Prediction Using Machine Learning

**CS465 Machine Learning Project**

A complete, reproducible machine learning pipeline for predicting credit card default risk using the UCI Credit Card Default dataset. The pipeline covers data preprocessing, model training, evaluation, and explainability.

---

## Results

| Model | CV ROC-AUC |
|---|---|
| Dummy Baseline | 0.500 |
| Logistic Regression | 0.727 |
| Random Forest | 0.777 |
| **Histogram Gradient Boosting** | **0.783** ✅ |

**Best model test set performance:**

| Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision |
|---|---|---|---|---|---|
| 0.817 | 0.652 | 0.366 | 0.469 | 0.777 | 0.551 |

---

## Dataset

**UCI Credit Card Default Dataset** — 30,000 customer records, 23 features, binary target (`default.payment.next.month`).

Download from Kaggle: https://www.kaggle.com/datasets/adilshamim8/credit-risk-benchmark-dataset

Place the downloaded CSV at: `data/raw/dataset.csv`

---

## Project Structure

```
cs465_credit_risk_project/
├── data/
│   └── raw/              # Place dataset.csv here (not committed)
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   └── 02_train_eval.ipynb  # Training and evaluation walkthrough
├── reports/
│   ├── figures/          # ROC curve, confusion matrix, feature importance
│   └── tables/           # Metrics CSV files
├── runs/                 # Saved model artifacts per run
├── src/
│   ├── config.py         # Seed, split ratios, paths
│   ├── data_load.py      # CSV loading
│   ├── preprocess.py     # Imputation, scaling, encoding
│   ├── models.py         # Model definitions
│   ├── train.py          # Training + cross-validation
│   ├── evaluate.py       # Test set evaluation + figures
│   ├── explain.py        # Permutation importance + SHAP
│   └── plotting.py       # Plot helpers
├── paper/
│   └── IEEE_Paper_Draft.docx
└── requirements.txt
```

---

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## How to Run

**Step 1 — Train all models (picks best by CV ROC-AUC):**
```bash
python -m src.train --data_path data/raw/dataset.csv --target default.payment.next.month
```

**Step 2 — Evaluate best model on held-out test set:**
```bash
python -m src.evaluate --run_dir runs/latest
```

**Step 3 — Permutation importance + explainability:**
```bash
python -m src.explain --run_dir runs/latest
```

All outputs are saved to `reports/figures/` and `reports/tables/`.

---

## Key Finding

`PAY_0` (most recent repayment status) is the dominant feature — shuffling it alone drops ROC-AUC by 0.074, roughly **50x more** than any other variable. The model has learned that whether you pay on time matters far more than how much you owe.

---

## Reproducibility

- Fixed random seed: `42`
- Stratified 80/20 train/test split
- 5-fold stratified cross-validation for model selection
- Full scikit-learn Pipeline (no data leakage)