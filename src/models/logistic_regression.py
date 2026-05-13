import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.evaluation.evaluate_model import confusion_matrix, roc_curve

def run_model(X_train, X_test, y_train, y_test):

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_res = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # =====================
    # SAVE MODEL + SCALER
    # =====================
    os.makedirs("output/models", exist_ok=True)

    joblib.dump(model, "output/models/logistic_regression.pkl")
    joblib.dump(scaler, "output/models/logistic_scaler.pkl")

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "model_name": "Logistic Regression",
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": auc,
        "confusion_matrix": cm,
        "roc": (fpr, tpr, auc)
    }