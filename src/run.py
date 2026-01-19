from src.data_loads import load_data
from src.preprocessing import prepare_data
from src.models import (
    train_baseline_linear,
    train_ridge,
    train_tuning,
)
from src.evaluation import evaluate_multiple_models

def main():
    df = load_data()

    X_train, X_test, y_train, y_test, _ = prepare_data(df)

    results = [
        train_baseline_linear(X_train, y_train),
        train_ridge(X_train, y_train),
        train_tuning(X_train, y_train, cv=2),
    ]

    metrics_df = evaluate_multiple_models(results, X_test, y_test)

    print("\nModel evaluation")
    print(metrics_df)

if __name__ == "__main__":
    main()