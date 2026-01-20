

import pandas as pd
from config import DATA_PATH

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    if df.empty:
        raise ValueError("Loaded dataset is empty")

    return df