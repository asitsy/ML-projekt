from src.data_loads import load_data
from src.preprocessing import prepare_data

df = load_data()

X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)