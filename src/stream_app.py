import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from run import main
from data_loads import load_data
from eda import plot_correlation_matrix


st.set_page_config(
    page_title="ML Project Dashboard",
    layout="centered",
)

st.title("📊 ML Project Dashboard")

st.write(
    "This dashboard presents model evaluation results "
    "Social media Impact."
)

# MODEL EVALUATION
st.header("Model evaluation")

with st.spinner("Running ML pipeline..."):
    results = main()

if results is None or results.empty:
    st.error("No evaluation results available.")
    st.stop()

st.subheader("Model comparison")
st.dataframe(results, use_container_width=True)

best_model = results.sort_values("rmse").iloc[0]

st.success(
    f"🏆 Best model: {best_model['model']}\n\n"
    f"RMSE: {best_model['rmse']:.3f}\n"
    f"R²: {best_model['r2']:.3f}"
)

st.subheader("Evaluation metrics")

st.write("**RMSE (lower is better)**")
st.bar_chart(results.set_index("model")[["rmse"]])

st.write("**R² (higher is better)**")
st.bar_chart(results.set_index("model")[["r2"]])

# EDA CORRELATION MATRIX
st.header("Exploratory Data Analysis")

with st.expander("Show correlation matrix"):
    df = load_data()

    fig = plot_correlation_matrix(df)
    st.pyplot(fig)