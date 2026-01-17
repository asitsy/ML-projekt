from typing import Dict, List
import pandas as pd

# тут імпортуємо метрики регресії зі sklearn.
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
# ModelResult це наша структура з models.py (10), яка містить навчену модель, її назву та кращі параметри
from src.models import ModelResult

def evaluate_model( # оцінює одну навчену модель регресії
    result: ModelResult, # обʼєкт ModelResult, який містить модель та її метадані
    X_test, # тест ознака
    y_test, # справжні значення цільової змінної
) -> Dict:
    model = result.model
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred) # середня величина помилки прогнозу
    mae = mean_absolute_error(y_test, y_pred) # середня різниця між прогнозом і реальними значеннями
    r2 = r2_score(y_test, y_pred) # показує яку частку варіації даних пояснює модель

# повертаємо всі метрики у вигляді словника
    return {
        "model_name": result.name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_params": result.best_params, # кращі параметри
    }

def evaluate_multiple_models( # оцінює кілька моделей та порівнює їх між собою.

    results: List[ModelResult], # список обʼєктів ModelResult ( baseline, альтернативні, tuned моделі )
    X_test, # тест ознака 
    y_test, # справжні значення цільової змінної
) -> pd.DataFrame:
    evaluations = [] # список для збереження результатів оцінки кожної моделі

    for result in results: # проходимо по всіх моделях у списку
        evaluations.append(evaluate_model(result, X_test, y_test)) # і для кожної моделі викликаємо evaluate_model

    # перетворюємо список словників у DataFrame і сортуємо моделі за RMSE, де менше=краще
    return pd.DataFrame(evaluations).sort_values(by="rmse")

#pytest проходить