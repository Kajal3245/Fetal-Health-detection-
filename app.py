import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fetal Health AI", layout="wide")

st.title("🩺 AI Fetal Health Prediction System")
st.warning("⚠️ This tool is for educational purposes only.")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join("model", "model.pkl")
        return pickle.load(open(model_path, "rb"))
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# -----------------------------
# LOAD DATA (for feature names)
# -----------------------------
df = pd.read_csv("data/fetal_health.csv")
feature_names = df.drop("fetal_health", axis=1).columns

# -----------------------------
# INPUT FIELDS (AUTO GENERATE)
# -----------------------------
st.header("Enter Clinical Parameters")

inputs = []

for feature in feature_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict"):

    features = np.array([inputs])
    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Normal")
    elif prediction == 2:
        st.warning("⚠️ Suspect")
    else:
        st.error("🚨 Pathological")