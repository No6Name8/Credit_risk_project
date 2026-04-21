from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Splits
    test_size: float = 0.2
    cv_folds: int = 5

    # Paths
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    tables_dir: Path = Path("reports/tables")
    runs_dir: Path = Path("runs")

CFG = Config()
