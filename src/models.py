import numpy as np
from typing import Dict, Any, Optional

#dataclasses дозволяє зручно зберігати результат навчання моделі  
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

@dataclass
class ModelResult:
    name: str # Назва моделі ( для логів, звіти, порівняння) 
    model: Any #навчена модель sklearn
    best_params: Optional[Dict[str, Any]] = None # Це найкращі результати(параметри), тунінга
   # це клас-контейнер для зберігання результатів навчання моделі 
    
# тут прописано функція для імпорта LinearRegression 
# порівняння з іншими моделями
def train_baseline_linear(
    X_train: np.ndarray, # Навчальні ознаки
    y_train: np.ndarray, # Цільова змінна
) -> ModelResult: 
    
    # створення моделі 
    model = LinearRegression()
    
    # - ця строчка яка навчає модель на тренувальних даних   
    model.fit(X_train, y_train)
    
    return ModelResult(
        name ="Baseline: LinearRegression",
        model= model,
    )
    
# Тут для імпорт для ridge ( це альтернативний класифікатор(в нас регресія) )
def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ModelResult:
    # Створення моделі:
    model = Ridge()
    model.fit(X_train, y_train)

    return ModelResult(
        name="Alternative: Ridge",
        model=model,
    )
    
# це функція для імпорта GridSearchCV ( це fine tuning )
def train_tuning (
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5, 
) -> ModelResult: 
    
    # тут ми перевіряємо tuning 
    param_grid = {
        "alpha": [0.1, 1.0, 10.0, 100.0],
}

    grid = GridSearchCV(
        estimator=Ridge(),
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    
    grid.fit(X_train, y_train)
    
    return ModelResult(
        name="Alternative: Ridge (tuned)",
        model=grid.best_estimator_,
        best_params=grid.best_params_,
    )
    
# тут прописано для імпорта RandomForest + tuning (альтернативний класифікатор(в нас регресія))
def train_random_forest_with_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> ModelResult:
    
    #Define the parameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    
    base_model = RandomForestRegressor( # модель машинного навчання для регресії
        random_state=42, # Для відтворюваності результатів
        n_jobs = -1, # Паралельне навчання
    )
    
    grid = GridSearchCV(
        estimator = base_model,
        param_grid = param_grid,
        cv = cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1, 
    )

    grid.fit(X_train, y_train) # - Initialize GridSearchCV
    
    #Це створення і повернення результату навчання моделі
    return ModelResult(
        name="Alternative: RandomForest (tuned)",
        model = grid.best_estimator_,
        best_params = grid.best_params_,
    )

