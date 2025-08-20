# Employee Retention Prediction Web App

A simple **Machine Learning + Streamlit** web application that predicts whether an employee is likely to leave (churn) or stay in the company based on HR dataset features.

---

## 🚀 Features
- Upload HR dataset (`HR_dataset2.csv`) for training.
- Machine Learning pipeline with preprocessing & RandomForest model.
- Model evaluation: accuracy, precision, recall.
- Streamlit-powered interactive web app.
- Predict employee churn with trained model.

---

## 📂 Project Structure
employee-retention-prediction/
│── app.py # Streamlit frontend
│── model.py # ML training pipeline
│── HR_dataset2.csv # Dataset (sample)
│── employee_churn_model.pkl# Saved trained model
│── requirements.txt # Project dependencies
│── README.md # Documentation