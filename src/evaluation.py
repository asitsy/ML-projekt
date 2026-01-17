from typing import Dict, List
import pandas as pd

from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.models import ModelResult

def evaluate_model(
    result: ModelResult,
    X_test,
    y_test,
) -> Dict:
    model = result.model
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model_name": result.name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_params": result.best_params,
    }

def evaluate_multiple_models(
    results: List[ModelResult],
    X_test,
    y_test,
) -> pd.DataFrame:
    evaluations = []

    for result in results:
        evaluations.append(evaluate_model(result, X_test, y_test))

    return pd.DataFrame(evaluations).sort_values(by="rmse")

#pytest проходить