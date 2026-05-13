import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE


def run_model(X_train, X_test, y_train, y_test):

    # =====================
    # SMOTE
    # =====================
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # =====================
    # Scaling
    # =====================
    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test = scaler.transform(X_test)

    # =====================
    # Model
    # =====================
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    # =====================
    # Predict
    # =====================
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # =====================
    # Metrics
    # =====================
    return {
        "model_name": "Logistic Regression",
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }