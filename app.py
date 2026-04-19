import streamlit as st
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ---------------- #
st.set_page_config(page_title="Fetal Health AI", layout="wide")

st.title("🩺 AI Fetal Health Prediction System")

st.info("⚠️ This tool assists in fetal health prediction using Machine Learning. Not a replacement for medical advice.")

# ---------------- LOAD MODELS ---------------- #
model = pickle.load(open("model/model.pkl", "rb"))
accuracy = pickle.load(open("model/accuracy.pkl", "rb"))
cm = pickle.load(open("model/cm.pkl", "rb"))
svm_acc = pickle.load(open("model/svm_acc.pkl", "rb"))

# ---------------- INPUT FEATURES ---------------- #
st.markdown("## Enter clinical parameters:")

feature_names = [
    "baseline value", "accelerations", "fetal_movement",
    "uterine_contractions", "light_decelerations", "severe_decelerations",
    "prolongued_decelerations", "abnormal_short_term_variability",
    "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability",
    "histogram_width", "histogram_min", "histogram_max",
    "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median",
    "histogram_variance", "histogram_tendency"
]

features = []

col1, col2 = st.columns(2)

for i, name in enumerate(feature_names):
    if i % 2 == 0:
        val = col1.number_input(name, value=0.0)
    else:
        val = col2.number_input(name, value=0.0)
    features.append(val)

# ---------------- PREDICTION ---------------- #
if st.button("Predict"):
    result = model.predict([features])[0]

    if result == 1:
        st.success("✅ Normal")
    elif result == 2:
        st.warning("⚠️ Suspect")
    else:
        st.error("❌ Pathological")

    # 📈 Input Graph
    st.subheader("📈 Input Feature Visualization")
    fig, ax = plt.subplots()
    ax.plot(features, marker='o')
    ax.set_title("Feature Values")
    st.pyplot(fig)

# ---------------- MODEL PERFORMANCE ---------------- #
st.markdown("## 📊 Model Performance")

st.write(f"Random Forest Accuracy: {round(accuracy*100,2)}%")
st.write(f"SVM Accuracy: {round(svm_acc*100,2)}%")

# ---------------- CONFUSION MATRIX ---------------- #
st.markdown("### Confusion Matrix")

fig2, ax2 = plt.subplots()
ax2.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax2.text(j, i, cm[i, j], ha="center", va="center")

st.pyplot(fig2)

# ---------------- FEATURE IMPORTANCE ---------------- #
st.markdown("### 🔍 Feature Importance")

importances = model.feature_importances_

fig3, ax3 = plt.subplots()
ax3.barh(feature_names, importances)

st.pyplot(fig3)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.caption("Developed by Kajal | Machine Learning Project")