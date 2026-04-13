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

# TARGET DISTRIBUTION & SCALE 
st.subheader("Target distribution & scale")

df = load_data()
target = df["actual_productivity_score"]

with st.expander("ℹ Wyjaśnienie wyników i skali danych"):
    target = df["actual_productivity_score"]

    st.write(f"""
    **Najlepszy model:** {best_model['model']}

    ---
    ### 📉 Metryki modelu

    **RMSE = {best_model['rmse']:.3f}**

    To jest średni błąd modelu **na danych zmiennej docelowej**  
    `actual_productivity_score`, która w tym zbiorze danych ma skalę **od {target.min():.1f} do {target.max():.1f}**  
    (typowo przyjmuje wartości **1–10**).

    Oznacza to, że model myli się średnio o około **{best_model['rmse']:.2f} jednostki** na tej skali.

    ---
    **R² = {best_model['r2']:.3f}**

    To wskaźnik dopasowania modelu.  
    Wynik **{best_model['r2']*100:.1f}%** oznacza, że model potrafi **wyjaśnić większość zmienności w danych**.

    ---
    ### 🧾 Co to znaczy w praktyce?

    Model przewiduje wartości **dokładnie i stabilnie**, a jego błędy są **niewielkie** w stosunku do skali 1–10.
    Dzięki temu można uznać model za **dobrze dopasowany i użyteczny**.

    ---
    ### 📊 Analiza zmiennej docelowej: `actual_productivity_score`
    """)

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(target, bins=20, color="#2980b9")
    ax.set_title("Rozkład `actual_productivity_score` (skala 0–10)")
    ax.set_xlabel("Wartość")
    ax.set_ylabel("Liczba próbek")
    st.pyplot(fig)

    # Stats
    st.write(f"""
    **Charakterystyka statystyczna:**

    • **Zakres:** {target.min():.1f} – {target.max():.1f}  
    • **Średnia (mean):** {target.mean():.2f}  
    • **Odchylenie standardowe (std):** {target.std():.2f}
    """)

    st.write("""
    **Interpretacja:**  
    Rozkład pokazuje, że wartości zmiennej `actual_productivity_score` są rozproszone w skali 1–10.  
    Ponieważ RMSE ≈ 0.5, model myli się średnio o **pół punktu na tej skali**, co jest **małym błędem**.
    """)

# METRIC BAR CHARTS
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

if __name__ == "__main__":
    main()