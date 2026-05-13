import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE


# =========================
# Load Dataset
# =========================

data = pd.read_csv("data/processed/processed_paysim.csv")


# =========================
# Features & Target
# =========================

X = data.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = data['isFraud']


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# Handle Imbalance
# =========================

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# =========================
# Scaling
# =========================

scaler = StandardScaler()

X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)


# =========================
# Model Training
# =========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train_res_scaled, y_train_res)


# =========================
# Predictions
# =========================

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# =========================
# Evaluation
# =========================

print("\n===== Logistic Regression =====\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))