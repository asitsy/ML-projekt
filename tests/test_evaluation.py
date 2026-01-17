from src.data_loads import load_data
from src.preprocessing import prepare_data
from src.models import train_baseline_linear
from src.evaluation import evaluate_model

def test_evaluate_model_returns_metrics():

    df = load_data()
    X_train, X_test, y_train, y_test, _ = prepare_data(df)

    result = train_baseline_linear(X_train, y_train)

    metrics = evaluate_model(result, X_test, y_test)

    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics