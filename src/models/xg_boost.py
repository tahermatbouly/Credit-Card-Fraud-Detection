import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from src.evaluation.evaluate_model import confusion_matrix, roc_curve

def run_model(X_train, X_test, y_train, y_test):

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    os.makedirs("output/models", exist_ok=True)
    joblib.dump(model, "output/models/xgboost.pkl")

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    return {
        "model_name": "XGBoost",
        "model": model,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": auc,
        "confusion_matrix": cm,
        "roc": (fpr, tpr, auc)
    }