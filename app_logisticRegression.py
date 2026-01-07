import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction using Logistic Regression")
st.write("This app predicts whether a customer is likely to churn.")

# --------------------------------------------------
# Load Dataset (Direct)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")

df = load_data()

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
st.subheader("🧹 Data Preprocessing")

df = df.copy()

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

st.success("Data preprocessing completed")

# --------------------------------------------------
# Features & Target
# --------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.success("Logistic Regression model trained successfully")

# --------------------------------------------------
# Model Evaluation
# --------------------------------------------------
st.subheader("📈 Model Evaluation")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# Classification Report
st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# --------------------------------------------------
# Churn Analysis
# --------------------------------------------------
tn, fp, fn, tp = cm.ravel()

st.subheader("🔍 Churn Analysis")
st.write(f"✔ Correctly identified churn customers: **{tp}**")
st.write(f"✔ Correctly identified non-churn customers: **{tn}**")

# --------------------------------------------------
# Single Customer Prediction
# --------------------------------------------------
st.subheader("🧑‍💼 Predict Churn for a New Customer")

input_data = {}
cols = X.columns

col1, col2 = st.columns(2)

for i, col in enumerate(cols):
    if i % 2 == 0:
        input_data[col] = col1.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
    else:
        input_data[col] = col2.number_input(
            col,
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )

if st.button("Predict Churn"):
    new_customer = pd.DataFrame([input_data])
    new_customer_scaled = scaler.transform(new_customer)

    prediction = model.predict(new_customer_scaled)[0]
    probability = model.predict_proba(new_customer_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay (Probability: {1 - probability:.2f})")
