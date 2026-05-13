import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


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
# Handle Imbalance (SMOTE)
# =========================

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


# =========================
# Model Training (XGBoost)
# =========================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])),
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)


# =========================
# Predictions
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# =========================
# Evaluation
# =========================

print("\n===== XGBoost Classifier =====\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))