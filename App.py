import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "employee_churn_model.pkl"

st.set_page_config(page_title="Employee Churn Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Employee Churn Prediction App")
st.write("Predict whether an employee is likely to leave based on HR metrics.")

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

if model is None:
    st.error(
        "Model file not found. Please run `python train_model.py` first to train and save the model "
        f"as `{MODEL_PATH}` in the project folder."
    )
    st.stop()

with st.expander("ℹ️ Input feature notes", expanded=False):
    st.markdown(
        """
        **Numerical**  
        - `satisfaction_level`: 0.0–1.0  
        - `last_evaluation`: 0.0–1.0  
        - `number_project`: integer, typical range 1–10  
        - `average_montly_hours`: 50–350  
        - `time_spend_company`: years, typical range 1–20  
        - `Work_accident`: 0 = No, 1 = Yes  
        - `promotion_last_5years`: 0 = No, 1 = Yes  

        **Categorical**  
        - `departments`: one of common HR dataset departments  
        - `salary`: one of `low`, `medium`, `high`
        """
    )

st.header("🔢 Single Prediction")
col1, col2 = st.columns(2)
with col1:
    satisfaction_level = st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    last_evaluation = st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
    number_project = st.number_input("Number of Projects", min_value=1, max_value=20, value=3, step=1)
    average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=160, step=1)
with col2:
    time_spend_company = st.number_input("Time Spent at Company (years)", min_value=1, max_value=40, value=3, step=1)
    Work_accident = st.selectbox("Work Accident", options=[0, 1], index=0)
    promotion_last_5years = st.selectbox("Promotion in Last 5 Years", options=[0, 1], index=0)

departments = st.selectbox(
    "Department",
    options=['sales','technical','support','IT','hr','product_mng','marketing','RandD','accounting','management']
)
salary = st.selectbox("Salary Level", options=['low','medium','high'])

if st.button("Predict"):
    input_row = pd.DataFrame({
        'satisfaction_level': [satisfaction_level],
        'last_evaluation': [last_evaluation],
        'number_project': [number_project],
        'average_montly_hours': [average_montly_hours],
        'time_spend_company': [time_spend_company],
        'Work_accident': [Work_accident],
        'promotion_last_5years': [promotion_last_5years],
        'departments': [departments],
        'salary': [salary]
    })
    pred = model.predict(input_row)[0]
    proba = None
    try:
        proba = float(model.predict_proba(input_row)[0][1])
    except Exception:
        pass
    st.success("⚠️ Likely to leave" if pred == 1 else "✅ Likely to stay")
    if proba is not None:
        st.write(f"Confidence (leave probability): **{proba:.2%}**")

st.divider()
st.header("📁 Batch Prediction (CSV)")
st.write("Upload a CSV with the same columns as training data (including `departments` and `salary`).")
uploaded = st.file_uploader("Choose CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    # Try the common column name fix
    if 'Departments ' in df.columns and 'departments' not in df.columns:
        df = df.rename(columns={'Departments ': 'departments'})
    try:
        preds = model.predict(df.drop(columns=[]))  # pipeline handles preprocessing
        out = df.copy()
        out['prediction'] = preds
        st.dataframe(out.head(50))
        # Download
        csv_bytes = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")
    except Exception as e:
        st.error(f"Prediction failed. Check your columns. Details: {e}")

st.caption("Tip: Train locally with `python train_model.py` to generate `employee_churn_model.pkl`, then run the app with `streamlit run app.py`.")
