import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)


# =========================
# Main Evaluation Function
# =========================

def evaluate_model(model_name, model, X_test, y_test):
    """
    Evaluates a trained ML model and prints key metrics.
    Works for classification models with predict + predict_proba.
    """

    print(f"\n==============================")
    print(f"📊 Evaluation: {model_name}")
    print(f"==============================\n")

    # =========================
    # Predictions
    # =========================
    y_pred = model.predict(X_test)

    # Some models may not support predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = None

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # =========================
    # Classification Report
    # =========================
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # =========================
    # Extra Metrics (Fraud-focused)
    # =========================
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nExtra Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")

    # =========================
    # ROC Curve
    # =========================
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        plt.figure()
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()
        plt.show()

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }