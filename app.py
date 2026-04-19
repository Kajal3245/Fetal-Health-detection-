import streamlit as st
import pickle
import os

st.set_page_config(page_title="Fetal Health AI", layout="wide")

# -----------------------------
# SAFE MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), "model", "model.pkl")
        model = pickle.load(open(model_path, "rb"))
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()