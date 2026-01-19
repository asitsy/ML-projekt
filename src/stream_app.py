import streamlit as st
import pandas as pd

st.title("ML Model Evaluation Dashboard")

results = pd.read_csv("results.csv")

st.subheader("Model comparison")
st.dataframe(results)

best_model = results.sort_values("rmse").iloc[0]

st.success(
    f"Best model: {best_model['model']}\n"
    f"RMSE: {best_model['rmse']:.3f}, R²: {best_model['r2']:.3f}"
)

st.bar_chart(
    results.set_index("model")[["rmse"]]
)