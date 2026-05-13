from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(name: str, model: Any, X_test: Any, y_test: Any) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    roc = float(roc_auc_score(y_test, y_prob))

    print(f"\n===== {name} =====\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC-AUC Score:")
    print(roc)

    return {"name": name, "roc_auc": roc}
