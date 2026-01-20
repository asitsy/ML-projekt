from data_loads import load_data
from preprocessing import prepare_data

def test_prepare_data_runs():
    df = load_data()

    X_train_processed, X_test_processed, y_train, y_test, preprocessor = prepare_data(df)

    assert X_train_processed is not None
    assert X_test_processed is not None
    assert y_train is not None
    assert y_test is not None
    assert preprocessor is not None

def test_no_missing_values_after_preprocessing():
    df = load_data()

    X_train_processed, X_test_processed, y_train, y_test, _ = prepare_data(df)

    assert not (X_train_processed != X_train_processed).any()
    assert not (X_test_processed != X_test_processed).any()