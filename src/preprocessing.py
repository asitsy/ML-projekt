import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from config import TARGET_COLUMN, NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessing_pipeline() -> ColumnTransformer: # Pipeline для ЧИСЛОВИХ колонок
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Якщо є пропущені значення — замінюємо їх медіаною
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),# Якщо є пропуски — заповнюємо найчастішим значенням
        ("encoder", OneHotEncoder(handle_unknown="ignore")) # Перетворення категорії на One-hot
    ])

    preprocessor = ColumnTransformer(transformers=[ # ColumnTransformer обʼєднує обидва pipeline
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    # Remove rows with missing target values
    df = df.dropna(subset=[TARGET_COLUMN])
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")

    X = df.drop(columns=[TARGET_COLUMN]) # X — всі колонки, окрім цільової
    y = df[TARGET_COLUMN] # y — цільова змінна, яку будемо передбачати

    X_train, X_test, y_train, y_test = train_test_split( # Ділимо дані на тренувальні та тестові
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    preprocessor = build_preprocessing_pipeline()

    X_train_processed = preprocessor.fit_transform(X_train) # Навчаємо preprocessing на TRAIN даних
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor