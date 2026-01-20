import numpy as np

from models import (
    ModelResult,
    train_baseline_linear,
    train_ridge,
    train_tuning,
    train_random_forest_with_tuning,
)

def test_model_result():
    model = object()

    result = ModelResult(
        name="model",
        model=model,
        best_params=None,
    )

    assert result.name == "model"
    assert result.model is model
    assert result.best_params is None

def test_train_baseline_linear():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    result = train_baseline_linear(X, y)

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert result.best_params is None

    preds = result.model.predict(X)
    assert preds.shape == (4,)

def test_train_ridge():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = train_ridge(X, y)

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert isinstance(result.best_params, dict)
    assert "alpha" in result.best_params

    preds = result.model.predict(X)
    assert preds.shape == (4,)

def test_train_tuning():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])
    y = np.array([1.1, 2.0, 3.1, 4.0, 5.1])

    result = train_tuning(X, y)

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert isinstance(result.best_params, dict)
    assert "alpha" in result.best_params

    preds = result.model.predict(X)
    assert preds.shape == (5,)

def test_train_random_forest_with_tuning():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = train_random_forest_with_tuning(
        X,
        y,
        cv=2, 
    )

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert isinstance(result.best_params, dict)

    preds = result.model.predict(X)
    assert preds.shape == (6,)
    
    import numpy as np

from models import (
    ModelResult,
    train_baseline_linear,
    train_ridge,
    train_tuning,
    train_random_forest_with_tuning,
)

def test_model_result():
    model = object()

    result = ModelResult(
        name="model",
        model=model,
        best_params=None,
    )

    assert result.name == "model"
    assert result.model is model
    assert result.best_params is None

def test_train_baseline_linear():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    result = train_baseline_linear(X, y)

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert result.best_params is None

    preds = result.model.predict(X)
    assert preds.shape == (4,)

def test_train_ridge():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    result = train_ridge(X, y)

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert result.best_params is None
    assert result.name == "Alternative: Ridge"

    preds = result.model.predict(X)
    assert preds.shape == (4,)

def test_train_tuning():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ])
    y = np.array([1.1, 2.0, 3.1, 4.0, 5.1])

    result = train_tuning(X, y)

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert isinstance(result.best_params, dict)
    assert "alpha" in result.best_params

    preds = result.model.predict(X)
    assert preds.shape == (5,)

def test_train_random_forest_with_tuning():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    result = train_random_forest_with_tuning(
        X,
        y,
        cv=2, 
    )

    assert isinstance(result, ModelResult)
    assert result.model is not None
    assert isinstance(result.best_params, dict)

    preds = result.model.predict(X)
    assert preds.shape == (6,)