import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("data/fetal_health.csv")

# Split data
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- RANDOM FOREST ---------------- #
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
cm = confusion_matrix(y_test, rf_pred)

# ---------------- SVM ---------------- #
svm_model = SVC()
svm_model.fit(X_train, y_train)

svm_accuracy = svm_model.score(X_test, y_test)

# ---------------- SAVE EVERYTHING ---------------- #
pickle.dump(rf_model, open("model/model.pkl", "wb"))
pickle.dump(rf_accuracy, open("model/accuracy.pkl", "wb"))
pickle.dump(cm, open("model/cm.pkl", "wb"))
pickle.dump(svm_model, open("model/svm.pkl", "wb"))
pickle.dump(svm_accuracy, open("model/svm_acc.pkl", "wb"))

print("✅ EVERYTHING SAVED SUCCESSFULLY!")