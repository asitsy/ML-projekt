import pandas as pd  

from data_loads import load_data
from preprocessing import prepare_data
from models import (
    train_baseline_linear,
    train_ridge,
    train_tuning,
    train_random_forest_with_tuning,
)
from evaluation import evaluate_model

def main():
    df = load_data()

    X_train, X_test, y_train, y_test, *_ = prepare_data(df)

    models = [
        train_baseline_linear(X_train, y_train),
        train_ridge(X_train, y_train),
        train_tuning(X_train, y_train),
        train_random_forest_with_tuning(X_train, y_train, cv=2),
    ]

    results = []

    for model_result in models:
        metrics = evaluate_model(
            model_result.model,
            X_test,
            y_test,
        )

        results.append({
            "model": model_result.name,
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
        })

    results_df = pd.DataFrame(results)

    return results_df   

if __name__ == "__main__":
    results = main()
    print(results)