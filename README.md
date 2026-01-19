# Social Media vs Productivity – Machine Learning Project

## 📌 Opis projektu
Celem projektu jest analiza wpływu korzystania z mediów społecznościowych na produktywność człowieka
oraz zbudowanie modelu uczenia maszynowego przewidującego rzeczywisty poziom produktywności
na podstawie danych demograficznych, behawioralnych i zawodowych.

Projekt został wykonany w ramach modułu **Uczenie maszynowe w Python**.

Dataset:  
https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity

---

## 🧠 Problem badawczy
Czy na podstawie:
- czasu spędzanego w mediach społecznościowych,
- liczby powiadomień,
- poziomu stresu,
- snu, pracy i przerw,
- satysfakcji zawodowej

można **przewidzieć rzeczywistą produktywność użytkownika** (`actual_productivity_score`)?

Jest to problem **regresji**.

---

## 📂 Struktura projektu

# Social Media vs Productivity – Machine Learning Project

## 📌 Opis projektu
Celem projektu jest analiza wpływu korzystania z mediów społecznościowych na produktywność człowieka
oraz zbudowanie modelu uczenia maszynowego przewidującego rzeczywisty poziom produktywności
na podstawie danych demograficznych, behawioralnych i zawodowych.

Projekt został wykonany w ramach modułu **Uczenie maszynowe w Python**.

Dataset:  
https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity

---

## 🧠 Problem badawczy
Czy na podstawie:
- czasu spędzanego w mediach społecznościowych,
- liczby powiadomień,
- poziomu stresu,
- snu, pracy i przerw,
- satysfakcji zawodowej

można **przewidzieć rzeczywistą produktywność użytkownika** (`actual_productivity_score`)?

Jest to problem **regresji**.

---

## 📂 Struktura projektu

├── data/
│ └── social_media_vs_productivity.csv
├── src/
│ ├── config.py
│ ├── data_loads.py
│ ├── eda.py
│ ├── preprocessing.py
│ ├── models.py
│ ├── evaluation.py
│ └── init.py
├── tests/
│ ├── test_data_loads.py
│ ├── test_preprocessing.py
│ ├── test_models.py
│ └── test_evaluation.py
├── README.md
└── requirements.txt

---

## ⚙️ Opis plików

### `config.py`
Plik konfiguracyjny zawierający:
- ścieżkę do danych (`DATA_PATH`)
- listę cech numerycznych i kategorycznych
- nazwę zmiennej docelowej (`TARGET_COLUMN`)

Pozwala centralnie zarządzać strukturą danych bez zmieniania logiki programu.

---

### `data_loads.py`
Odpowiada za:
- wczytanie danych z pliku CSV
- walidację istnienia pliku
- sprawdzenie, czy dataset nie jest pusty

---

### `eda.py` (Exploratory Data Analysis)
Wstępna analiza danych:
- podstawowe informacje o zbiorze (`shape`, `dtypes`)
- liczba brakujących wartości
- statystyki opisowe cech numerycznych
- macierz korelacji
- rozkład zmiennej docelowej

Ten etap pozwala **zrozumieć dane przed modelowaniem**.

---

### `preprocessing.py`
Przygotowanie danych do uczenia maszynowego:
- podział na zbiór treningowy i testowy
- imputacja brakujących wartości
- normalizacja danych numerycznych
- kodowanie cech kategorycznych (One-Hot Encoding)
- zastosowanie `Pipeline` i `ColumnTransformer`

Efektem jest macierz cech gotowa do trenowania modeli.

---

### `models.py`
Definicja i trenowanie modeli:

#### Modele:
- **Baseline**: `LinearRegression`
- **Alternatywny**: `Ridge`
- **Alternatywny z tuningiem**: `Ridge + GridSearchCV`
- **Zaawansowany**: `RandomForestRegressor + GridSearchCV`

Zastosowano:
- `GridSearchCV` do fine-tuningu hiperparametrów
- `dataclass ModelResult` do czytelnego przechowywania wyników modeli

---

### `evaluation.py`
Ocena jakości modeli:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² score

Możliwość porównania wielu modeli w formie tabeli.

---

### `tests/`
Testy jednostkowe (`pytest`) sprawdzające:
- poprawne wczytanie danych
- działanie preprocessing
- trenowanie modeli
- zwracanie metryk

Testy zapewniają poprawność i stabilność rozwiązania.

---

## 📊 Wyniki
Modele są porównywane na podstawie błędu RMSE.
Najlepsze rezultaty osiąga model:
- **RandomForestRegressor (tuned)**

Fine-tuning znacząco poprawia jakość predykcji względem modelu bazowego.

---

## 🛠 Technologie
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- pytest

---

## 👤 Autor

Anastasiia Tsyban @: anasta.tsyban@gmail.com
Volodymyr Poleshko @: volodymyrpoleshko@gmail.com

Projekt wykonany w ramach zajęć akademickich  
kierunek: Informatyka  
moduł: Uczenie maszynowe w Python