import pandas as pd
from pathlib import Path

def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"CSV seems empty: shape={df.shape}")
    return df
