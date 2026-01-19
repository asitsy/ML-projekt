import pytest

from src.data_loads import load_data
from src.models import (
    train_baseline_linear,
    train_ridge,
    train_tuning,
    train_random_forest_with_tuning,
)
from src.preprocessing import prepare_data

@pytest.fixture(scope="module")
def prepared_data():
    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    return X_train, X_test, y_train, y_test

def test_train_baseline_linear_integration(prepared_data):
    Xtrain, , ytrain,  = prepared_data

    result = train_baseline_linear(X_train, y_train)

    assert result is not None
    assert result.model is not None
    assert result.name == "Baseline: LinearRegression"
    assert result.best_params is None

def test_train_ridge_integration(prepared_data):
    Xtrain, , ytrain,  = prepared_data

    result = train_ridge(X_train, y_train)

    assert result is not None
    assert result.model is not None
    assert isinstance(result.best_params, dict)
    assert "alpha" in result.best_params

def test_train_ridge_with_tuning_integration(prepared_data):
    Xtrain, , ytrain,  = prepared_data

    result = train_tuning(X_train, y_train)

    assert result is not None
    assert result.model is not None
    assert result.name == "Alternative: Ridge (tuned)"
    assert isinstance(result.best_params, dict)
    assert "alpha" in result.best_params

def test_train_random_forest_with_tuning_integration(prepared_data):
    Xtrain, , ytrain,  = prepared_data

    # обмежуємо дані → швидше
    result = train_random_forest_with_tuning(
        X_train.head(500),
        y_train.head(500),
        cv=2,
    )

    assert result is not None
    assert result.model is not None
    assert isinstance(result.best_params, dict)
