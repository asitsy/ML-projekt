import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

class ModelResult:
    name: str
    model: Any
    best_params: Optional[Dict[str, Any]] = None
    
def train_baseline_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ModelResult:
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return ModelResult(
        name ="Baseline: LinearRegression",
        model= model,
    )



