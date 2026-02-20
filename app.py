import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Energy Production Forecast App",
    layout="wide",
)

# ---------- CONSTANTS ----------
FEATURES = [
    "Year",
    "Coal and peat",
    "Natural gas",
    "Oil",
    "Onshore wind energy",
    "Renewable hydropower",
    "Solar photovoltaic",
    "Solid biofuels",
    "Total non-renewable",
    "Total renewable",
    "Total value",
]

ALGO_DISPLAY_TO_KEY = {
    "Linear Regression": "linear",
    "Random Forest (Final)": "rf",
    "Tuned Random Forest": "tuned",
}

TARGET_KEYS = ["nonrenewable", "renewable", "total"]


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    models = {
        "linear": {
            "nonrenewable": joblib.load("linear_nonrenewable.pkl"),
            "renewable": joblib.load("linear_renewable.pkl"),
            "total": joblib.load("linear_total.pkl"),
        },
        "rf": {
            "nonrenewable": joblib.load("rf_nonrenewable.pkl"),
            "renewable": joblib.load("rf_renewable.pkl"),
            "total": joblib.load("rf_total.pkl"),
        },
        "tuned": {
            "nonrenewable": joblib.load("tuned_nonrenewable.pkl"),
            "renewable": joblib.load("tuned_renewable.pkl"),
            "total": joblib.load("tuned_total.pkl"),
        },
    }
    return models


models = load_models()


# ---------- SIDEBAR ----------
st.sidebar.title("Energy Forecast – Controls")

selected_algo_display = st.sidebar.selectbox(
    "Choose Model Type",
    ["Linear Regression", "Random Forest (Final)", "Tuned Random Forest"],
    index=1,  # Default to RF
)

selected_algo_key = ALGO_DISPLAY_TO_KEY[selected_algo_display]

st.sidebar.markdown("---")
st.sidebar.write("This app predicts **next-year**:")
st.sidebar.write("- Total non-renewable energy")
st.sidebar.write("- Total renewable energy")
st.sidebar.write("- Total energy (total value)")

st.sidebar.markdown("---")
st.sidebar.caption("Made by MUHAMMED NIHAL V P 💻")


# ---------- MAIN LAYOUT ----------
st.title("⚡ Energy Production Forecast Web App")

tab_overview, tab_predict, tab_compare, tab_about = st.tabs(
    ["Overview", "Interactive Prediction", "Model Comparison", "About Project"]
)

# ---------- OVERVIEW TAB ----------
with tab_overview:
    st.subheader("Project Overview")

    st.markdown(
        """
This app forecasts **next-year energy production** using three ML algorithms:

- **Linear Regression**
- **Random Forest** (your final chosen model)
- **Tuned Random Forest**

### Inputs (Current Year)
The models use the following features for the **current year**:

- Year  
- Coal and peat  
- Natural gas  
- Oil  
- Onshore wind energy  
- Renewable hydropower  
- Solar photovoltaic  
- Solid biofuels  
- Total non-renewable  
- Total renewable  
- Total value  

The app predicts:
- **Total non-renewable energy – next year**
- **Total renewable energy – next year**
- **Total energy (total value) – next year**
"""
    )

    st.markdown("### How to use this app")
    st.markdown(
        """
1. Go to **Interactive Prediction** tab  
2. Enter current year values  
3. Select model type in the sidebar  
4. Click **Predict** to see next-year forecast  
5. Use **Model Comparison** tab to compare all three models for the same input  
"""
    )


# ---------- PREDICTION TAB ----------
with tab_predict:
    st.subheader("🔮 Interactive Prediction")

    st.markdown(
        f"Current selected model: **{selected_algo_display}**"
    )

    with st.form("prediction_form"):
        st.markdown("#### Enter current year values")

        cols1 = st.columns(3)
        cols2 = st.columns(4)
        cols3 = st.columns(4)

        # Store values in the same order as FEATURES
        inputs = {}

        # Row 1
        with cols1[0]:
            inputs["Year"] = st.number_input("Year", min_value=1900, max_value=2100, value=2025, step=1)
        with cols1[1]:
            inputs["Coal and peat"] = st.number_input("Coal and peat", value=0.0)
        with cols1[2]:
            inputs["Natural gas"] = st.number_input("Natural gas", value=0.0)

        # Row 2
        with cols2[0]:
            inputs["Oil"] = st.number_input("Oil", value=0.0)
        with cols2[1]:
            inputs["Onshore wind energy"] = st.number_input("Onshore wind energy", value=0.0)
        with cols2[2]:
            inputs["Renewable hydropower"] = st.number_input("Renewable hydropower", value=0.0)
        with cols2[3]:
            inputs["Solar photovoltaic"] = st.number_input("Solar photovoltaic", value=0.0)

        # Row 3
        with cols3[0]:
            inputs["Solid biofuels"] = st.number_input("Solid biofuels", value=0.0)
        
        # ---------- AUTO CALCULATE TOTALS ----------
        total_nonrenewable = (
            inputs["Coal and peat"] +
            inputs["Natural gas"] +
            inputs["Oil"]
        )

        total_renewable = (
            inputs["Onshore wind energy"] +
            inputs["Renewable hydropower"] +
            inputs["Solar photovoltaic"] +
            inputs["Solid biofuels"]
        )

        total_value = total_nonrenewable + total_renewable

        # Save into inputs so model still works
        inputs["Total non-renewable"] = total_nonrenewable
        inputs["Total renewable"] = total_renewable
        inputs["Total value"] = total_value

        # Show computed values to user
        st.info(f"⚡ Total Non-Renewable: {total_nonrenewable:.2f}")
        st.info(f"🌱 Total Renewable: {total_renewable:.2f}")
        st.info(f"🔋 Total Energy Value: {total_value:.2f}")


        submitted = st.form_submit_button("Predict Next Year")

    if submitted:
        # Arrange in correct order
        x_list = [inputs[feat] for feat in FEATURES]
        X = np.array(x_list).reshape(1, -1)

        model_non = models[selected_algo_key]["nonrenewable"]
        model_ren = models[selected_algo_key]["renewable"]
        model_tot = models[selected_algo_key]["total"]

        pred_non = float(model_non.predict(X)[0])
        pred_ren = float(model_ren.predict(X)[0])
        pred_tot = float(model_tot.predict(X)[0])

        next_year = int(inputs["Year"] + 1)

        st.success(f"Predictions for year **{next_year}** using **{selected_algo_display}**:")

        results_df = pd.DataFrame(
            {
                "Metric": [
                    "Total non-renewable (next year)",
                    "Total renewable (next year)",
                    "Total energy – total value (next year)",
                ],
                "Predicted value": [pred_non, pred_ren, pred_tot],
            }
        )

        st.table(results_df)

        st.bar_chart(
            data=results_df.set_index("Metric")["Predicted value"],
            use_container_width=True,
        )


# ---------- MODEL COMPARISON TAB ----------
with tab_compare:
    st.subheader("📊 Compare Models for the Same Input")

    st.markdown(
        """
Use the same current-year values from the **Interactive Prediction** tab  
and compare predictions from all three models.
"""
    )

    with st.form("compare_form"):
        st.markdown("#### Enter current year values for comparison")

        cols1 = st.columns(3)
        cols2 = st.columns(4)
        cols3 = st.columns(4)

        comp_inputs = {}

        # Row 1
        with cols1[0]:
            comp_inputs["Year"] = st.number_input("Year ", min_value=1900, max_value=2100, value=2025, step=1, key="c_year")
        with cols1[1]:
            comp_inputs["Coal and peat"] = st.number_input("Coal and peat ", value=0.0, key="c_coal")
        with cols1[2]:
            comp_inputs["Natural gas"] = st.number_input("Natural gas ", value=0.0, key="c_gas")

        # Row 2
        with cols2[0]:
            comp_inputs["Oil"] = st.number_input("Oil ", value=0.0, key="c_oil")
        with cols2[1]:
            comp_inputs["Onshore wind energy"] = st.number_input("Onshore wind energy ", value=0.0, key="c_wind")
        with cols2[2]:
            comp_inputs["Renewable hydropower"] = st.number_input("Renewable hydropower ", value=0.0, key="c_hydro")
        with cols2[3]:
            comp_inputs["Solar photovoltaic"] = st.number_input("Solar photovoltaic ", value=0.0, key="c_solar")

        # Row 3
        with cols3[0]:
            comp_inputs["Solid biofuels"] = st.number_input("Solid biofuels ", value=0.0, key="c_bio")

        # ---------- AUTO CALCULATE TOTALS (COMPARISON) ----------
        total_nonrenewable = (
            comp_inputs["Coal and peat"] +
            comp_inputs["Natural gas"] +
            comp_inputs["Oil"]
        )

        total_renewable = (
            comp_inputs["Onshore wind energy"] +
            comp_inputs["Renewable hydropower"] +
            comp_inputs["Solar photovoltaic"] +
            comp_inputs["Solid biofuels"]
        )

        total_value = total_nonrenewable + total_renewable

        # Inject back for model usage
        comp_inputs["Total non-renewable"] = total_nonrenewable
        comp_inputs["Total renewable"] = total_renewable
        comp_inputs["Total value"] = total_value

        # Display totals to user
        st.info(f"⚡ Total Non-Renewable: {total_nonrenewable:.2f}")
        st.info(f"🌱 Total Renewable: {total_renewable:.2f}")
        st.info(f"🔋 Total Energy Value: {total_value:.2f}")


        comp_submitted = st.form_submit_button("Compare All Models")

    if comp_submitted:
        x_list = [comp_inputs[feat] for feat in FEATURES]
        Xc = np.array(x_list).reshape(1, -1)
        next_year_c = int(comp_inputs["Year"] + 1)

        rows = []
        for algo_display, algo_key in ALGO_DISPLAY_TO_KEY.items():
            m_non = models[algo_key]["nonrenewable"]
            m_ren = models[algo_key]["renewable"]
            m_tot = models[algo_key]["total"]

            p_non = float(m_non.predict(Xc)[0])
            p_ren = float(m_ren.predict(Xc)[0])
            p_tot = float(m_tot.predict(Xc)[0])

            rows.append(
                {
                    "Model": algo_display,
                    "Total non-renewable (next)": p_non,
                    "Total renewable (next)": p_ren,
                    "Total value (next)": p_tot,
                }
            )

        comp_df = pd.DataFrame(rows)

        st.markdown(f"### Predictions for year **{next_year_c}**")
        st.dataframe(comp_df, use_container_width=True)

        #st.markdown("#### Total value – next year (all models)")
        import plotly.express as px

        fig = px.bar(
            comp_df,
            x="Model",
            y="Total value (next)",
            text="Total value (next)",
            title="Predicted Total Value Comparison",
        )

        fig.update_layout(
            xaxis_tickangle=0,   # 👈 makes labels horizontal
            height=400
        )

        fig.update_traces(texttemplate='%{text:.2f}', textposition="outside")

        st.plotly_chart(fig, use_container_width=True)


        #st.markdown("#### Renewable vs Non-renewable – next year")
        rn_df = comp_df.set_index("Model")[["Total non-renewable (next)", "Total renewable (next)"]]
        fig2 = px.bar(
            rn_df.reset_index(),
            x="Model",
            y=["Total non-renewable (next)", "Total renewable (next)"],
            barmode="group",
            title="Renewable vs Non-renewable Prediction Comparison",
        )

        # 👉 Force horizontal labels
        fig2.update_layout(
            xaxis_tickangle=0,   # Horizontal
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)


    st.subheader("📈 Actual vs Predicted Comparison")

    st.markdown("### 🌲 Random Forest – Actual vs Predicted")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("rf_nonrenewable_actual_vs_pred.png", caption="RF – Non-renewable", width=300)

    with col2:
        st.image("rf_renewable_actual_vs_pred.png", caption="RF – Renewable", width=300)

    with col3:
        st.image("rf_total_actual_vs_pred.png", caption="RF – Total Value", width=300)

    st.markdown("### 📏 Linear Regression – Actual vs Predicted")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("lr_nonrenewable_actual_vs_pred.png", caption="Linear – Non-renewable", width=300)

    with col2:
        st.image("lr_renewable_actual_vs_pred.png", caption="Linear – Renewable", width=300)

    with col3:
        st.image("lr_total_actual_vs_pred.png", caption="Linear – Total Value", width=300)

    st.info(
        "Although Linear Regression visually appears smoother in scatter plots, "
        "Random Forest achieved higher R² scores and better captured nonlinear energy patterns. "
        "This highlights the importance of combining visual and statistical evaluation."
    )


# ---------- ABOUT TAB ----------
with tab_about:
    st.subheader("ℹ About This Project")

    st.markdown(
        """
This project analyses and forecasts **annual energy production** split into:

- **Total non-renewable energy**
- **Total renewable energy**
- **Total energy (total value)**

Machine learning models used:

1. **Linear Regression** – simple baseline model  
2. **Random Forest** – ensemble tree-based model (chosen as final model)  
3. **Tuned Random Forest** – hyperparameter-tuned variant for experimentation  

Trained **separate models** for each target:
- One model for *Total non-renewable (next year)*
- One model for *Total renewable (next year)*
- One model for *Total value (next year)*  
for all three algorithms.



"""
    )

    st.markdown("---")
    st.caption("Deployed as a Streamlit app for interactive demo of ML models.")
