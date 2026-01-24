# Wykonuje wstępną analizę danych (EDA) dla punktu 4:
# statystyki opisowe, brakujące wartości oraz korelacje

import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from config import (DATA_PATH, TARGET_COLUMN, NUMERIC_FEATURES)
from data_loads import load_data
import numpy as np

def basic_data_info(df: pd.DataFrame) -> None:
    
    print(" BASIC DATA INFO ")
    print(f"Dataset shape: {df.shape}") #df.shape повертає рядки та колонки

    print("\nData types:")
    print(df.dtypes) # показує назву кожної колонки і її тип

    print("\nMissing values per column:")
    print(df.isnull().sum()) # показує скільки в нас пропусків (sum скільки у кожній колонці)
    
    
def statistical_summary(df: pd.DataFrame) -> pd.DataFrame: 

    print("\n STATISTICAL_SUMMARY (NUMERIC FEATURES) ")

    numeric_df = df[NUMERIC_FEATURES]
    summary = numeric_df.describe()

    print(summary)
    return summary
    
def correlation_matrix(df: pd.DataFrame, save_path: str | None = None) -> pd.DataFrame:
    
    numeric_df = df[NUMERIC_FEATURES]
    corr = numeric_df.corr() # рахує кореляцію між числовими змінними
    
    plt.figure(figsize=(12, 8)) # тут створюється графік
    sns.heatmap(
        corr,
        cmap ="coolwarm", # редагування стилю
        linewidths = 0.5
    )
    plt.title("Correlation Matrix (Numeric Features)")
    
    if save_path:                               
        plt.savefig(save_path, bbox_inches = "tight")
        
    plt.show()
    return corr

def target_distribution(df: pd.DataFrame) -> None: # аналіз змінної
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], bins = 30, kde=True) # показує розподіл значень
    plt.title(f"Distribution of {TARGET_COLUMN}")
    plt.xlabel(TARGET_COLUMN)
    plt.ylabel("Count")
    plt.show() 

# STREAMLIT DEPLOY
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()

    # створюємо маску для верхнього трикутника, щоб не дублювати
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={"shrink": .8},
        ax=ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    return fig