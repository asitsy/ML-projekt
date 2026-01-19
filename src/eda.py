# Wykonuje wstępną analizę danych (EDA) dla punktu 4:
# statystyki opisowe, brakujące wartości oraz korelacje

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES)
from src.data_loads import load_data


def basic_data_info(df: pd.DataFrame) -> None:
    
    print("=== BASIC DATA INFO ===")
    print(f"Dataset shape: {df.shape}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    
def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:

    print("\n STATISTICAL_SUMMARY (NUMERIC FEATURES) ")

    numeric_df = df[NUMERIC_FEATURES]
    summary = numeric_df.describe()

    print(summary)
    return summary
    
