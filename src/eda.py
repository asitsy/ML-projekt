# Wykonuje wstępną analizę danych (EDA) dla punktu 4:
# statystyki opisowe, brakujące wartości oraz korelacje

import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES)
from src.data_loads import load_data


def basic_data_info(df: pd.DataFrame) -> None:
    
    print(" BASIC DATA INFO")
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
    
def correlation_matrix(df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    
    numeric_df = df[NUMERIC_FEATURES]
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        cmap ="coolwarm",
        linewidths = 0.5
    )
    plt.title("Correlation Matrix (Numeric Features)")
    
    if save_path:
        plt.savefig(save_path, bbox_inches = "tight")
        
    plt.show()
    return corr

def target_distribution(df: pd.DataFrame) -> None:
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], bins = 30, kde=True)
    plt.title(f"Distribution of {TARGET_COLUMN}")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Count")
    plt.show()