# Odpowiada za wczytywanie surowych danych z pliku CSV (znajduje się w /data)
# Nie wykonuje żadnego przetwarzania ani analizy danych

import pandas as pd

from src.config import DATA_PATH

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found")

    df = pd.read_csv(DATA_PATH)
    return df